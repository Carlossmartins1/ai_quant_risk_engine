"""
Tests de la Fase 6: plano de control (cola de comandos exactly-once), capa de datos del dashboard,
la app Streamlit completa (``AppTest``), integración con el orquestador, migraciones y healthcheck.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import shutil
import threading

import pytest

from ai_agents.control_plane import ControlPlane, SimulationCommand
from ai_agents.orchestrator import Orchestrator, OrchestratorConfig
from ai_agents.trader_agents import AgentPersona, TradingAgent
from dashboard.data import DashboardRepository, fmt_usd
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


# ============================================================================= app Streamlit
@pytest.fixture
def app_env(clean_db, monkeypatch):
    from tests.conftest import TEST_DB, _conn_kwargs

    kw = _conn_kwargs()
    for k, v in {"POSTGRES_HOST": kw["host"], "POSTGRES_PORT": str(kw["port"]), "POSTGRES_USER": kw["user"],
                 "POSTGRES_PASSWORD": kw["password"], "POSTGRES_DB": TEST_DB}.items():
        monkeypatch.setenv(k, v)
    # Vacía (no borrada): DBConfig.from_env() carga .env sin override y la reintroduciría
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "")
    import streamlit as st

    st.cache_resource.clear()
    yield clean_db
    st.cache_resource.clear()


def test_streamlit_app_renders_and_triggers_black_swan(app_env):
    from streamlit.testing.v1 import AppTest

    _liquidate_whale(app_env)
    ControlPlane(app_env).publish("NORMAL", {"BTC": {"mark": 89_000.0, "oracle": 89_010.0, "basis_bps": -1.1,
                                                     "liquidity": 0.8, "spread_bps": 2.0}})
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    labels = {m.label for m in at.metric}
    assert {"Insurance Fund", "Déficit absorbido", "Liquidaciones", "Mark", "Oracle"} <= labels
    assert next(m for m in at.metric if m.label == "Déficit absorbido").value == "$1,360.00"

    bs = next(b for b in at.button if b.key == "btn_black_swan")
    assert bs.disabled                                            # exige confirmación explícita
    at.checkbox(key="confirm_bs").check().run()
    next(b for b in at.button if b.key == "btn_black_swan").click().run()
    assert not at.exception
    with app_env.transaction(readonly=True) as cur:
        cur.execute("SELECT command::text AS c, requested_by FROM simulation_commands")
        rows = cur.fetchall()
    assert [(r["c"], r["requested_by"]) for r in rows] == [("TRIGGER_BLACK_SWAN", "dashboard")]
    assert at.checkbox(key="confirm_bs").value is False             # la confirmación se rearma


def test_streamlit_controls_require_operator_password(app_env, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "s3cret")
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    assert all(b.disabled for b in at.button if b.key in ("btn_normal", "btn_black_swan"))
    at.text_input(key="operator_pwd").input("wrong").run()
    assert any("incorrecta" in e.value for e in at.error)
    at.text_input(key="operator_pwd").input("s3cret").run()
    assert not next(b for b in at.button if b.key == "btn_normal").disabled


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
