"""Auto-Deleveraging: planificador puro (water-filling) + transacción ACID + integración con el liquidador."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import psycopg2
import pytest

from risk_engine.adl import AdlEngine, plan_adl
from risk_engine.liquidation_executor import LiquidationConfig, LiquidationExecutor
from tests.helpers import make_store, seed_account, seed_insurance_fund


def _cands(rows):
    return pd.DataFrame(rows, columns=["user_id", "unrealized_pnl", "maintenance_margin", "account_equity", "account_mm"])


# ============================================================================= planificador puro
def test_common_fraction_pro_rata_to_unrealized_profit():
    plan = plan_adl(30_000, _cands([(1, 20_000, 1_000, 50_000, 1_000), (2, 40_000, 2_000, 90_000, 2_000)]))
    assert plan.phi == pytest.approx(0.5) and plan.status == "COVERED"
    assert plan.haircuts.tolist() == pytest.approx([10_000, 20_000])          # misma φ: pro-rata a u_i
    assert plan.recovered == pytest.approx(30_000) and plan.residual == 0


def test_partial_when_profits_cannot_cover_the_shortfall():
    plan = plan_adl(100_000, _cands([(1, 20_000, 0, 30_000, 0), (2, 10_000, 0, 20_000, 0)]))
    assert plan.phi == 1.0 and plan.status == "PARTIAL"
    assert plan.recovered == pytest.approx(30_000) and plan.residual == pytest.approx(70_000)


def test_hedged_account_is_capped_by_health_and_others_pay_more():
    """Cuenta 1: pierna ganadora 20k + perdedora (E−1.1·MM = 4k, U−1.1·MMw = 18.9k) => φmax ≈ 0.2116."""
    rows = [(1, 20_000, 1_000, 15_000, 10_000), (2, 20_000, 1_000, 60_000, 1_000)]
    plan = plan_adl(16_000, _cands(rows))
    phi_max_1 = (15_000 - 1.1 * 10_000) / (20_000 - 1.1 * 1_000)
    assert plan.account_phi[1] == pytest.approx(phi_max_1)
    assert plan.account_phi[2] == pytest.approx((16_000 - phi_max_1 * 20_000) / 20_000)
    assert plan.recovered == pytest.approx(16_000)
    # post-ADL: la cuenta cubierta conserva health ≥ 1.10
    e_after = 15_000 - plan.account_phi[1] * 20_000
    mm_after = 10_000 - plan.account_phi[1] * 1_000
    assert e_after / mm_after == pytest.approx(1.10)


def test_account_below_health_floor_is_never_touched():
    plan = plan_adl(1_000, _cands([(1, 5_000, 100, 10_500, 10_000), (2, 5_000, 100, 50_000, 100)]))
    assert 1 not in plan.account_phi and plan.account_phi[2] == pytest.approx(0.2)


def test_planner_validation():
    assert plan_adl(10, _cands([])).status == "NO_COUNTERPARTIES"
    with pytest.raises(ValueError):
        plan_adl(0, _cands([]))
    with pytest.raises(ValueError):
        plan_adl(10, _cands([(1, -5, 0, 1, 0)]))


# ============================================================================= transacción ACID
MARKS = {"BTC": {"mark": 100_000.0}, "ETH": {"mark": 4_000.0}}


def _fund_negative(db, balance):
    seed_insurance_fund(db, 1.0)
    with db.transaction() as cur:
        cur.execute("UPDATE insurance_fund SET balance = %s WHERE fund_id = 1", (balance,))


def _state(db):
    with db.transaction(readonly=True) as cur:
        cur.execute("SELECT balance FROM insurance_fund WHERE fund_id = 1")
        fund = cur.fetchone()["balance"]
        cur.execute("SELECT position_id, user_id, quantity, status FROM open_positions ORDER BY position_id")
        pos = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT user_id, deposited_collateral FROM users_collateral ORDER BY user_id")
        coll = {r["user_id"]: r["deposited_collateral"] for r in cur.fetchall()}
    return fund, pos, coll


@pytest.mark.integration
def test_adl_restores_fund_by_haircutting_winners_pro_rata(clean_db):
    _fund_negative(clean_db, -25_000)
    w1 = seed_account(clean_db, "w1", "WHALE_AGGRESSIVE", 50_000, [("BTC", 2.0, 90_000, 10, 0.0125)])   # u = 20 000
    w2 = seed_account(clean_db, "w2", "ARBITRAGEUR_NEUTRAL", 50_000, [("ETH", 30.0, 3_000, 5, 0.02)])    # u = 30 000
    loser = seed_account(clean_db, "l", "MARKET_MANIPULATOR", 1e6, [("BTC", -1.0, 95_000, 5, 0.0125)])  # u < 0

    out = AdlEngine(clean_db, make_store(MARKS)).run(trigger_batch_id=None)
    assert out.status == "COVERED" and out.phi == pytest.approx(0.5)
    assert out.recovered == Decimal("25000") and out.residual == 0 and out.fund_after == 0
    assert {f.user_id for f in out.fills} == {w1, w2}

    fund, pos, coll = _state(clean_db)
    assert fund == 0
    qty = {p["user_id"]: p["quantity"] for p in pos}
    assert qty[w1] == Decimal("1") and qty[w2] == Decimal("15") and qty[loser] == Decimal("-1")   # φ = 0.5
    assert coll[w1] == Decimal("50000") and coll[w2] == Decimal("50000")    # el principal no se toca
    with clean_db.transaction(readonly=True) as cur:
        cur.execute("SELECT entry_type, amount, balance_after FROM insurance_fund_ledger ORDER BY entry_id DESC LIMIT 1")
        led = cur.fetchone()
        cur.execute("SELECT status, shortfall, recovered, counterparties, positions FROM adl_events")
        ev = cur.fetchone()
        cur.execute("SELECT sum(haircut) AS h, count(*) AS n FROM adl_fills")
        fills = cur.fetchone()
    assert led["entry_type"] == "ADL_RECOVERY" and led["amount"] == Decimal("25000") and led["balance_after"] == 0
    assert ev["status"] == "COVERED" and ev["counterparties"] == 2 and ev["positions"] == 2
    assert fills["h"] == Decimal("25000") and fills["n"] == 2


@pytest.mark.integration
def test_adl_closes_positions_fully_when_profits_fall_short(clean_db):
    _fund_negative(clean_db, -50_000)
    w = seed_account(clean_db, "w", "WHALE_AGGRESSIVE", 50_000, [("BTC", 1.0, 90_000, 10, 0.0125)])     # u = 10 000
    out = AdlEngine(clean_db, make_store(MARKS)).run(trigger_batch_id=None)
    assert out.status == "PARTIAL" and out.recovered == Decimal("10000") and out.residual == Decimal("40000")
    fund, pos, coll = _state(clean_db)
    assert fund == Decimal("-40000") and pos[0]["status"] == "CLOSED" and coll[w] == Decimal("50000")


@pytest.mark.integration
def test_no_counterparties_is_audited_only_when_triggered(clean_db):
    import uuid

    _fund_negative(clean_db, -1_000)
    seed_account(clean_db, "l", "MARKET_MANIPULATOR", 1e6, [("BTC", -1.0, 95_000, 5, 0.0125)])
    engine = AdlEngine(clean_db, make_store(MARKS))
    assert engine.run() is None                                           # reintento periódico: nada que registrar
    out = engine.run(trigger_batch_id=uuid.uuid4())
    assert out.status == "NO_COUNTERPARTIES" and out.recovered == 0 and out.fund_after == Decimal("-1000")
    with clean_db.transaction(readonly=True) as cur:
        cur.execute("SELECT count(*) AS n FROM adl_events WHERE status = 'NO_COUNTERPARTIES'")
        assert cur.fetchone()["n"] == 1


@pytest.mark.integration
def test_adl_is_noop_when_fund_is_solvent_and_audit_is_append_only(clean_db):
    seed_insurance_fund(clean_db, 1_000)
    seed_account(clean_db, "w", "WHALE_AGGRESSIVE", 50_000, [("BTC", 1.0, 90_000, 10, 0.0125)])
    assert AdlEngine(clean_db, make_store(MARKS)).run() is None
    _fund_negative(clean_db, -100)
    AdlEngine(clean_db, make_store(MARKS)).run()
    with pytest.raises(psycopg2.Error, match="append-only"):
        with clean_db.transaction() as cur:
            cur.execute("UPDATE adl_events SET recovered = 0")


@pytest.mark.integration
def test_locked_counterparty_is_skipped_not_waited(clean_db):
    """SKIP LOCKED: una cuenta bloqueada por otro proceso no se espera; el ADL usa las demás."""
    _fund_negative(clean_db, -5_000)
    busy = seed_account(clean_db, "busy", "WHALE_AGGRESSIVE", 50_000, [("BTC", 2.0, 90_000, 10, 0.0125)])
    free = seed_account(clean_db, "free", "ARBITRAGEUR_NEUTRAL", 50_000, [("ETH", 30.0, 3_000, 5, 0.02)])
    with clean_db.transaction() as cur:
        cur.execute("SELECT 1 FROM users_collateral WHERE user_id = %s FOR UPDATE", (busy,))
        out = AdlEngine(clean_db, make_store(MARKS)).run()
    assert out.status == "COVERED" and {f.user_id for f in out.fills} == {free}
    assert out.fills[0].account_phi == pytest.approx(5_000 / 30_000)


# ============================================================================= integración con el liquidador
@pytest.mark.integration
def test_liquidation_that_depletes_the_fund_triggers_adl_in_the_same_cycle(clean_db):
    """
    Reproducción del incidente: un corto apalancado liquidado muy por encima de su precio de bancarrota
    deja el fondo en negativo. El mismo ciclo ejecuta el ADL sobre el largo ganador y repone el fondo.
    """
    seed_insurance_fund(clean_db, 500)
    short = seed_account(clean_db, "manip", "MARKET_MANIPULATOR", 10_000, [("BTC", -1.0, 90_000, 15, 0.0125)])
    long_ = seed_account(clean_db, "whale", "WHALE_AGGRESSIVE", 100_000, [("BTC", 1.0, 80_000, 10, 0.0125)])
    store = make_store({"BTC": {"mark": 100_000.0, "asks": [(100_500, 0.2)]}})   # libro fino: el backstop ejecuta
    ex = LiquidationExecutor(clean_db, store, LiquidationConfig(max_price_age_s=None))
    report = ex.run_cycle()

    (liq,) = report.liquidated
    assert liq.user_id == short and liq.deficit == Decimal("900") and liq.insurance_fund_balance == Decimal("-400")
    (adl,) = report.adl
    assert adl.trigger_batch_id == liq.batch_id and adl.status == "COVERED"
    assert adl.recovered == -liq.insurance_fund_balance and adl.fund_after == 0
    assert {f.user_id for f in adl.fills} == {long_}
    fund, pos, coll = _state(clean_db)
    assert fund == 0 and coll[long_] == Decimal("100000")
    remaining = [p for p in pos if p["user_id"] == long_][0]["quantity"]
    assert Decimal("0") < remaining < Decimal("1")
    assert ex.run_cycle().adl == ()                                        # fondo repuesto: no se repite


@pytest.mark.integration
def test_adl_can_be_disabled(clean_db):
    seed_insurance_fund(clean_db, 500)
    seed_account(clean_db, "manip", "MARKET_MANIPULATOR", 10_000, [("BTC", -1.0, 90_000, 15, 0.0125)])
    seed_account(clean_db, "whale", "WHALE_AGGRESSIVE", 100_000, [("BTC", 1.0, 80_000, 10, 0.0125)])
    store = make_store({"BTC": {"mark": 100_000.0, "asks": [(100_500, 0.2)]}})
    report = LiquidationExecutor(clean_db, store, LiquidationConfig(max_price_age_s=None, adl_enabled=False)).run_cycle()
    assert report.liquidated and report.adl == () and _state(clean_db)[0] < 0
