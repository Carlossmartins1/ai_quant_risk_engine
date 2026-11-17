"""
Tests de la Fase 6: plano de control (cola de comandos exactly-once), capa de datos del dashboard,
la app Plotly Dash (layout + callbacks reales vía HTTP), integración con el orquestador, migraciones y
healthcheck.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import shutil
import threading
from datetime import datetime, timedelta, timezone

import pytest

import plotly.utils

from ai_agents.control_plane import ControlPlane, ControlState, SimulationCommand
from ai_agents.orchestrator import Orchestrator, OrchestratorConfig
from ai_agents.trader_agents import AgentPersona, TradingAgent
from dashboard.data import DashboardRepository, MarketTape, feed_view, fmt_usd
from risk_engine.liquidation_executor import LiquidationExecutor
from tests.helpers import make_store, seed_account, seed_insurance_fund

pytestmark = pytest.mark.integration
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _liquidate_whale(db, *, mark=89_000):
    seed_insurance_fund(db, 50_000)
    uid = seed_account(db, "whale", "WHALE_AGGRESSIVE", 4_750.0, [("BTC", 1.0, 95_000.0, 20, 0.025)])
    store = make_store({"BTC": {"mark": mark, "bids": [(88_950, 0.4), (88_900, 0.3), (88_800, 1.0)]}})
    LiquidationExecutor(db, store).liquidate_account(uid)
    return uid


# ============================================================================= plano de control
def test_command_queue_is_exactly_once_across_consumers(clean_db):
    producer = ControlPlane(clean_db, instance_id="dashboard")
    ids = [producer.enqueue(SimulationCommand.TRIGGER_BLACK_SWAN, requested_by=f"op{i}") for i in range(40)]
    consumed: list[list[SimulationCommand]] = []
    barrier = threading.Barrier(4)

    def consumer(i):
        cp = ControlPlane(clean_db, instance_id=f"orch-{i}")
        barrier.wait()
        while batch := cp.consume(limit=1):
            consumed.append(batch)

    threads = [threading.Thread(target=consumer, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(len(b) for b in consumed) == len(ids)                 # ni perdidos ni duplicados
    state = producer.state()
    assert state.pending_commands == 0
    with clean_db.transaction(readonly=True) as cur:
        cur.execute("SELECT count(DISTINCT consumed_by) AS n, count(*) FILTER (WHERE consumed_at IS NULL) AS p "
                    "FROM simulation_commands")
        row = cur.fetchone()
    assert row["p"] == 0 and row["n"] >= 2                           # repartido entre réplicas (SKIP LOCKED)


def test_publish_state_is_json_safe_and_tracks_heartbeat(clean_db):
    cp = ControlPlane(clean_db, instance_id="orch-x")
    assert not cp.state().is_online()
    cp.publish("BLACK_SWAN", {"BTC": {"mark": 90_000.0, "spread_bps": math.nan}}, {"x": math.inf})
    st = cp.state()
    assert st.is_online() and st.current_mode == "BLACK_SWAN" and st.orchestrator_id == "orch-x"
    assert st.market_state["BTC"]["spread_bps"] is None and st.stats["x"] is None
    cp.mark_offline()
    assert cp.state().current_mode == "OFFLINE"


def test_orchestrator_consumes_dashboard_command(clean_db):
    ControlPlane(clean_db).enqueue(SimulationCommand.TRIGGER_BLACK_SWAN, requested_by="risk-operator")
    agents = [TradingAgent(AgentPersona.whale(directional_bias="LONG"))]
    cfg = OrchestratorConfig(tick_interval_s=0.05, risk_interval_s=0.05, decision_interval_s=0.5,
                             control_interval_s=0.1, stress_paths=200, seed=4)
    report = Orchestrator(clean_db, config=cfg, agents=agents).run(2.5)
    assert "BLACK_SWAN" in [m for _, m in report.mode_changes]
    with clean_db.transaction(readonly=True) as cur:
        cur.execute("SELECT consumed_by FROM simulation_commands")
        assert cur.fetchone()["consumed_by"] is not None
        cur.execute("SELECT current_mode, market_state FROM simulation_control")
        row = cur.fetchone()
    assert row["current_mode"] == "OFFLINE" and set(row["market_state"]) == {"BTC", "ETH"}
    assert "funding_hourly" in row["market_state"]["BTC"]
    with clean_db.transaction(readonly=True) as cur:
        cur.execute("SELECT stats FROM simulation_control")
        stats = cur.fetchone()["stats"]
    assert stats["feed"] == {"configured": False, "connected": False, "symbols": {}}   # sin --live
    risk = stats["risk"]
    if risk:                                                       # ventana de retornos suficiente
        assert risk["es"] >= risk["var"] >= 0 and risk["alpha"] == 0.99


def test_reset_agents_command_recapitalizes_accounts(clean_db):
    seed_insurance_fund(clean_db, 50_000)
    agents = [TradingAgent(AgentPersona.whale(directional_bias="LONG")), TradingAgent(AgentPersona.arbitrageur())]
    orch = Orchestrator(clean_db, config=OrchestratorConfig(seed=4), agents=agents)
    ids = orch.bootstrap()
    whale, arb = ids["whale_trader"], ids["arbitrageur"]
    with clean_db.transaction() as cur:  # whale liquidado y arbitrajista con posición abierta y PnL acumulado
        cur.execute("UPDATE users_collateral SET deposited_collateral = 0, account_status = 'LIQUIDATED' "
                    "WHERE user_id = %s", (whale,))
        cur.execute("UPDATE users_collateral SET deposited_collateral = 76000 WHERE user_id = %s", (arb,))
        cur.execute("""INSERT INTO open_positions (user_id, symbol, quantity, entry_price, leverage, maintenance_margin_rate)
                       VALUES (%s, 'BTC', 0.5, 95000, 3, 0.025)""", (arb,))
    orch._liquidated_at[whale] = 1.0

    ControlPlane(clean_db).enqueue(SimulationCommand.RESET_AGENTS, requested_by="risk-operator")
    orch._control_tick()

    with clean_db.transaction(readonly=True) as cur:
        cur.execute("SELECT agent_name, account_status::text AS s, deposited_collateral::float AS c "
                    "FROM users_collateral ORDER BY user_id")
        accounts = {r["agent_name"]: (r["s"], r["c"]) for r in cur.fetchall()}
        cur.execute("SELECT count(*) AS n FROM open_positions WHERE status = 'OPEN'")
        open_positions = cur.fetchone()["n"]
        cur.execute("SELECT balance::float AS b FROM insurance_fund")
        fund = cur.fetchone()["b"]
    assert accounts == {"whale_trader": ("ACTIVE", 10_000.0), "arbitrageur": ("ACTIVE", 100_000.0)}
    assert open_positions == 0 and fund == 50_000.0 and not orch._liquidated_at


# ============================================================================= capa de datos
def test_repository_views(clean_db):
    uid = _liquidate_whale(clean_db)
    seed_account(clean_db, "arb", "ARBITRAGEUR_NEUTRAL", 100_000.0,
                 [("BTC", 1.0, 95_000.0, 3, 0.025), ("ETH", -27.0, 3_500.0, 3, 0.02)])
    risky = seed_account(clean_db, "risky", "WHALE_AGGRESSIVE", 3_000.0, [("BTC", 1.0, 95_000.0, 20, 0.025)])
    repo = DashboardRepository(clean_db)

    fund = repo.fund_overview()
    assert fund.balance == pytest.approx(48_640) and fund.deficits_absorbed == pytest.approx(1_360)
    assert fund.capital_injected == pytest.approx(50_000) and fund.liquidations == 1
    assert fund.net_pnl == pytest.approx(-1_360)

    ledger = repo.fund_ledger()
    assert ledger["entry_type"].tolist() == ["CAPITAL_INJECTION", "BANKRUPTCY_DEFICIT"]
    assert ledger["agent_name"].tolist()[1] == "whale"

    liq = repo.liquidations()
    row = liq.iloc[0]
    assert row["user_id"] == uid and row["deficit"] == pytest.approx(1_360) and row["positions"] == 1
    assert row["wavg_slippage_bps"] == pytest.approx(abs(88_890 - 89_000) / 89_000 * 1e4, rel=1e-3)
    assert row["execution_source"] == "L2_BOOK" and row["regime"] == "LIVE_TESTNET"

    # Sin orquestador vivo: fallback a last_mark_price (sin posiciones abiertas con mark aún => "none")
    health = repo.accounts_health()
    assert health.marks_source in ("none", "last_mark_price") and not health.marks_live
    # Con orquestador publicando marks: health ratios con los mismos precios que el motor
    ControlPlane(clean_db).publish("NORMAL", {"BTC": {"mark": 93_000.0}, "ETH": {"mark": 3_400.0}})
    health = repo.accounts_health()
    t = health.table.set_index("agent_name")
    assert health.marks_live and health.marks == {"BTC": 93_000.0, "ETH": 3_400.0}
    assert t.loc["risky", "health_ratio"] == pytest.approx((3_000 - 2_000) / (93_000 * 0.025))
    assert t.loc["risky", "status"] == "🔴 Liquidable" and t.loc["arb", "status"] == "🟢 Sana"
    assert t.loc["arb", "positions"] == "+1 BTC, -27 ETH"
    assert health.table["agent_name"].iloc[0] == "risky"            # peor salud primero
    assert risky in health.table["user_id"].tolist() and uid not in health.table["user_id"].tolist()


def test_fmt_usd():
    assert fmt_usd(-1234.5) == "-$1,234.50" and fmt_usd(150_527.3, compact=True) == "$150,527"
    assert fmt_usd(None) == "—" and fmt_usd(math.nan) == "—"


# ============================================================================= feed / tape
def _state(mode="NORMAL", *, at=None, sim=95_000.0, live=None, connected=True):
    feed = {"configured": live is not None, "connected": connected,
            "symbols": {} if live is None else {"BTC": {"mark": live, "oracle": live, "funding_hourly": 1.25e-5,
                                                        "latency_ms": 42.0, "age_ms": 120.0}}}
    return ControlState(mode, "orch", at or datetime.now(timezone.utc), 0.5,
                        {"BTC": {"mark": sim, "oracle": sim, "basis_bps": 0.0, "funding_hourly": 1e-5}},
                        {"feed": feed}, 0)


def test_feed_view_and_market_tape():
    view = feed_view(_state(sim=90_000.0, live=100_000.0)).set_index("symbol")
    assert view.loc["BTC", "deviation_bps"] == pytest.approx(-1_000.0)              # crash simulado −10 %
    assert view.loc["BTC", "latency_ms"] == 42.0
    assert feed_view(_state())["live_mark"].isna().all()                              # sin feed real

    tape = MarketTape(maxlen=3)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tape.record(_state(at=t0, live=95_000.0))
    tape.record(_state(at=t0, live=95_000.0))                                        # mismo heartbeat: dedupe
    tape.record(_state("BLACK_SWAN", at=t0 + timedelta(seconds=1), sim=85_500.0, live=95_000.0))
    df = tape.frame()
    assert len(df) == 2 and df["mode"].tolist() == ["NORMAL", "BLACK_SWAN"]
    assert df["deviation_bps"].iloc[-1] == pytest.approx(-1_000.0)


# ============================================================================= app Dash
@pytest.fixture
def app_env(clean_db, monkeypatch):
    from tests.conftest import TEST_DB, _conn_kwargs

    kw = _conn_kwargs()
    for k, v in {"POSTGRES_HOST": kw["host"], "POSTGRES_PORT": str(kw["port"]), "POSTGRES_USER": kw["user"],
                 "POSTGRES_PASSWORD": kw["password"], "POSTGRES_DB": TEST_DB}.items():
        monkeypatch.setenv(k, v)
    # Vacía (no borrada): DBConfig.from_env() carga .env sin override y la reintroduciría
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "")
    monkeypatch.setenv("DASHBOARD_OPERATOR_NAME", "")
    yield clean_db


def _dumps(component) -> str:
    return json.dumps(component, cls=plotly.utils.PlotlyJSONEncoder, ensure_ascii=False)


def _fire(client, trigger: str, password: str | None = None):
    """Dispara el callback ``send_command`` igual que el navegador tras confirmar el diálogo."""
    outputs = [{"id": "cmd-toast", "property": p} for p in ("children", "icon", "is_open")]
    inputs = [{"id": t, "property": "submit_n_clicks", "value": 1 if t == trigger else None}
              for t in ("confirm-normal", "confirm-black-swan", "confirm-reset")]
    body = {"output": "..cmd-toast.children...cmd-toast.icon...cmd-toast.is_open..", "outputs": outputs,
            "inputs": inputs, "changedPropIds": [f"{trigger}.submit_n_clicks"],
            "state": [{"id": "operator-pwd", "property": "value", "value": password}]}
    resp = client.post("/_dash-update-component", json=body)
    assert resp.status_code == 200, resp.data
    return resp.get_json()["response"]["cmd-toast"]


def _commands(db):
    with db.transaction(readonly=True) as cur:
        cur.execute("SELECT command::text AS c, requested_by FROM simulation_commands ORDER BY command_id")
        return [(r["c"], r["requested_by"]) for r in cur.fetchall()]


def test_dash_app_renders_and_triggers_commands(app_env):
    from dashboard.app import controls_view, create_app, main_view

    _liquidate_whale(app_env)
    ControlPlane(app_env).publish("NORMAL", {"BTC": {"mark": 89_000.0, "oracle": 89_010.0, "basis_bps": -1.1,
                                                     "liquidity": 0.8, "spread_bps": 2.0}},
                                  {"risk": {"var": 1_234.5, "es": 1_500.0, "alpha": 0.99, "n_scenarios": 200}})
    repo = DashboardRepository(app_env)
    app = create_app(repo)
    client = app.server.test_client()
    assert client.get("/").status_code == 200 and client.get("/healthz").status_code == 200
    layout = client.get("/_dash-layout").get_data(as_text=True)
    for cid in ("tick", "confirm-black-swan", "btn-reset", "kpi-row", "fund-graph", "price-graph", "hl-brand",
                "acc-table", "liq-table", "adl-table", "feed-table"):
        assert f'"{cid}"' in layout, cid

    view = main_view(repo, MarketTape(), "/assets/hyperliquid.svg")
    kpis = _dumps(view.kpis)
    for text in ("Insurance Fund", "Cobertura / OI", "Déficit absorbido", "$1,360.00", "VaR 99%", "$1,234.50",
                 "ES 99%", "Liquidaciones"):
        assert text in kpis, text
    assert "BTC-PERP" in _dumps(view.tickers) and "89,000.00" in _dumps(view.tickers)
    assert view.alert is None and view.fund_figure.data                          # fondo positivo: sin ADL

    controls = controls_view(repo, None)
    assert controls.normal_disabled and not controls.black_swan_disabled        # ya en NORMAL
    toast = _fire(client, "confirm-black-swan")
    assert toast["is_open"] and "Cisne Negro" in toast["children"] and toast["icon"] == "success"
    _fire(client, "confirm-reset")
    assert _commands(app_env) == [("TRIGGER_BLACK_SWAN", "dashboard"), ("RESET_AGENTS", "dashboard")]
    cmds = controls_view(repo, None).commands
    assert [r["label"].split(" ", 1)[1] for r in cmds] == ["Recapitalizar", "Cisne Negro"]
    assert all(r["status"] == "⏳ en cola" for r in cmds)                       # sin orquestador que consuma


def test_dash_controls_require_operator_password(app_env, monkeypatch):
    from dashboard.app import controls_view, create_app

    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "s3cret")
    repo = DashboardRepository(app_env)
    client = create_app(repo).server.test_client()
    v = controls_view(repo, None)
    assert v.normal_disabled and v.black_swan_disabled and v.reset_disabled
    toast = _fire(client, "confirm-black-swan", "wrong")                          # validado en el servidor
    assert toast["icon"] == "danger" and "incorrecta" in toast["children"] and _commands(app_env) == []
    assert not controls_view(repo, "s3cret").reset_disabled
    _fire(client, "confirm-black-swan", "s3cret")
    assert _commands(app_env) == [("TRIGGER_BLACK_SWAN", "operator")]


def test_dash_tabs_and_adl_events(app_env):
    from dashboard.app import TAB_OUTPUTS, main_view, tab_view

    seed_insurance_fund(app_env, 1_000)                                           # fondo < déficit => ADL
    uid = seed_account(app_env, "whale", "WHALE_AGGRESSIVE", 4_750.0, [("BTC", 1.0, 95_000.0, 20, 0.025)])
    store = make_store({"BTC": {"mark": 89_000, "bids": [(88_950, 0.4), (88_900, 0.3), (88_800, 1.0)]}})
    LiquidationExecutor(app_env, store).liquidate_account(uid)
    seed_account(app_env, "arb", "ARBITRAGEUR_NEUTRAL", 100_000.0, [("BTC", 1.0, 95_000.0, 3, 0.025)])
    ControlPlane(app_env).publish("BLACK_SWAN", {"BTC": {"mark": 85_500.0}, "ETH": {"mark": 3_400.0}})
    repo = DashboardRepository(app_env)

    adl = repo.adl_events()
    assert len(adl) == 1 and adl.iloc[0]["shortfall"] == pytest.approx(360) and adl.iloc[0]["agent_name"] == "whale"
    view = main_view(repo, MarketTape(), "/logo.svg")
    assert "ADL" in _dumps(view.alert) and "ADL REQUERIDO" in _dumps(view.kpis)

    tape = MarketTape()
    for tab, outputs in TAB_OUTPUTS.items():
        values = tab_view(repo, tape, tab)
        assert set(values) == {cid for cid, _ in outputs}, tab
        _dumps(values)                                                            # JSON-serializable
    acc = tab_view(repo, tape, "accounts")
    assert [r["agent_name"] for r in acc["acc-table"]] == ["arb"]
    assert acc["pos-table"][0]["symbol"] == "BTC" and acc["pos-table"][0]["mark_price"] == 85_500.0
    assert tab_view(repo, tape, "adl")["adl-table"][0]["shortfall"] == pytest.approx(360)
    assert tab_view(repo, tape, "liquidations")["liq-table"][0]["deficit"] == pytest.approx(1_360)


# ============================================================================= migraciones / healthcheck
def test_migrations_idempotent_and_tamper_detection(db, tmp_path, monkeypatch):
    from database import migrate as mig
    from tests.conftest import TEST_DB, _conn_kwargs
    from database.connection import DBConfig

    kw = _conn_kwargs()
    cfg = DBConfig(host=kw["host"], port=kw["port"], user=kw["user"], password=kw["password"], dbname=TEST_DB)
    assert mig.migrate(cfg) == []                                  # ya aplicadas por la fixture
    tampered = tmp_path / "migrations"
    shutil.copytree(mig.MIGRATIONS_DIR, tampered)
    f = tampered / "002_simulation_control.sql"
    f.write_text(f.read_text() + "\n-- cambio no autorizado\n")
    monkeypatch.setattr(mig, "MIGRATIONS_DIR", tampered)
    with pytest.raises(mig.MigrationError):
        mig.migrate(cfg)


def test_container_healthcheck(app_env, monkeypatch):
    spec = importlib.util.spec_from_file_location("healthcheck", ROOT / "deployment" / "scripts" / "healthcheck.py")
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    monkeypatch.setenv("SERVICE_ROLE", "orchestrator")
    assert hc.main() == 1                                          # sin heartbeat
    ControlPlane(app_env).publish("NORMAL", {})
    assert hc.main() == 0
    monkeypatch.setenv("SERVICE_ROLE", "shipper")
    assert hc.main() == 0
    monkeypatch.setenv("POSTGRES_PORT", "1")
    assert hc.main() == 1                                          # BD inaccesible
