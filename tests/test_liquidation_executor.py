"""
Tests de integración del motor de liquidación contra PostgreSQL real (esquema de la Fase 1).

Cubren: liquidación con déficit absorbido por el Insurance Fund, liquidación con fee, cartera
cross multi-posición, carreras entre workers concurrentes, fail-safe por precios rancios,
backstop por libro insuficiente / haircut de liquidez y modo live con cliente de exchange simulado.
"""

from __future__ import annotations

import threading
from decimal import Decimal

import pytest

from exchange_connector.hyperliquid_client import OrderKind, OrderResult, OrderStatus, Side
from risk_engine.liquidation_executor import (
    ExecutionMode,
    LiquidationConfig,
    LiquidationExecutor,
    LiquidationStatus,
    MarketRegime,
)
from tests.helpers import make_store, seed_account, seed_insurance_fund

pytestmark = pytest.mark.integration

D = Decimal
BTC_DEEP_BIDS = [(92_490, 5.0)]


def fetch(db, sql, params=()):
    with db.transaction(readonly=True) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def account_row(db, uid):
    return fetch(db, "SELECT * FROM users_collateral WHERE user_id = %s", (uid,))[0]


def fund_balance(db) -> Decimal:
    return fetch(db, "SELECT balance FROM insurance_fund WHERE fund_id = 1")[0]["balance"]


def whale(db, collateral=4_750.0):
    return seed_account(db, "whale", "WHALE_AGGRESSIVE", collateral, [("BTC", 1.0, 95_000.0, 20, 0.025)])


# ----------------------------------------------------------------------------- casos base
def test_healthy_account_untouched(clean_db):
    uid = whale(clean_db)
    ex = LiquidationExecutor(clean_db, make_store({"BTC": {"mark": 95_000, "bids": BTC_DEEP_BIDS}}))
    assert ex.run_cycle().outcomes == ()
    out = ex.liquidate_account(uid)
    assert out.status is LiquidationStatus.NOT_LIQUIDATABLE
    assert account_row(clean_db, uid)["account_status"] == "ACTIVE"
    assert fetch(clean_db, "SELECT count(*) AS n FROM liquidation_events")[0]["n"] == 0


def test_bankruptcy_deficit_absorbed_by_insurance_fund(clean_db):
    """
    Gap bajista: mark 89 000 (equity −1 250 ≤ MM 2 225). El libro absorbe 1 BTC a VWAP 88 890:
    PnL realizado −6 110 => equity tras cierre −1 360 => déficit 1 360 al Insurance Fund, fee 0.
    """
    seed_insurance_fund(clean_db, 50_000)
    uid = whale(clean_db)
    store = make_store({"BTC": {"mark": 89_000, "oracle": 89_020,
                                "bids": [(88_950, 0.4), (88_900, 0.3), (88_800, 1.0)]}})
    report = LiquidationExecutor(clean_db, store).run_cycle()

    assert [o.user_id for o in report.liquidated] == [uid]
    out = report.liquidated[0]
    fill = out.fills[0]
    assert fill.execution_price == D("88890") and fill.source == "L2_BOOK"
    assert fill.bankruptcy_price == D("90250")
    assert fill.realized_pnl == D("-6110")
    assert out.equity_before == D("-1250") and out.maintenance_margin_before == D("2225")
    assert out.deficit == D("1360") and out.liquidation_fee == 0
    assert out.final_collateral == 0 and out.insurance_fund_balance == D("48640")

    # Persistencia
    acc = account_row(clean_db, uid)
    assert acc["account_status"] == "LIQUIDATED" and acc["deposited_collateral"] == 0
    pos = fetch(clean_db, "SELECT status, closed_at, last_mark_price FROM open_positions WHERE user_id = %s", (uid,))[0]
    assert pos["status"] == "LIQUIDATED" and pos["closed_at"] is not None and pos["last_mark_price"] == 89_000
    ev = fetch(clean_db, "SELECT * FROM liquidation_events")[0]
    assert str(ev["liquidation_batch_id"]) == str(out.batch_id)
    assert ev["deficit"] == D("1360") and ev["insurance_fund_delta"] == D("-1360")
    assert ev["execution_price"] == D("88890") and ev["oracle_price"] == D("89020")
    assert ev["margin_ratio_before"] < 0 and ev["market_regime"] == "LIVE_TESTNET"
    assert ev["metadata"]["source"] == "L2_BOOK"
    assert fund_balance(clean_db) == D("48640")
    ledger = fetch(clean_db, "SELECT entry_type, amount, balance_after, liquidation_event_id FROM insurance_fund_ledger ORDER BY entry_id")
    assert [(r["entry_type"], r["amount"], r["balance_after"]) for r in ledger] == [
        ("CAPITAL_INJECTION", D("50000"), D("50000")),
        ("BANKRUPTCY_DEFICIT", D("-1360"), D("48640")),
    ]
    assert ledger[1]["liquidation_event_id"] == ev["event_id"]
    # Re-ejecutar el ciclo no vuelve a liquidar (idempotencia operativa)
    assert LiquidationExecutor(clean_db, store).run_cycle().outcomes == ()


def test_positive_equity_liquidation_charges_fee(clean_db):
    """mark 92 500: equity 2 250 ≤ MM 2 312.5. Ejecución 92 490 => equity 2 240, fee 0.5 % = 462.45."""
    uid = whale(clean_db)
    out = LiquidationExecutor(clean_db, make_store({"BTC": {"mark": 92_500, "bids": BTC_DEEP_BIDS}})).liquidate_account(uid)
    assert out.status is LiquidationStatus.LIQUIDATED
    assert out.liquidation_fee == D("462.45") and out.deficit == 0
    assert out.final_collateral == D("1777.55")
    assert account_row(clean_db, uid)["deposited_collateral"] == D("1777.55")
    assert fund_balance(clean_db) == D("462.45")
    ev = fetch(clean_db, "SELECT liquidation_fee, insurance_fund_delta FROM liquidation_events")[0]
    assert ev["liquidation_fee"] == ev["insurance_fund_delta"] == D("462.45")


def test_cross_margin_multi_position_allocation(clean_db):
    seed_insurance_fund(clean_db, 10_000)
    uid = seed_account(clean_db, "cross", "WHALE_AGGRESSIVE", 6_000.0,
                       [("BTC", 1.0, 95_000.0, 20, 0.025), ("ETH", 10.0, 3_500.0, 20, 0.02)])
    store = make_store({
        "BTC": {"mark": 90_000, "bids": [(89_900, 2.0)]},
        "ETH": {"mark": 3_300, "bids": [(3_290, 50.0)]},
    })
    out = LiquidationExecutor(clean_db, store).liquidate_account(uid)
    # PnL: BTC −5 100, ETH −2 100 => equity_after = 6 000 − 7 200 = −1 200
    assert out.deficit == D("1200") and len(out.fills) == 2
    evs = fetch(clean_db, "SELECT * FROM liquidation_events ORDER BY position_id")
    assert len({str(e["liquidation_batch_id"]) for e in evs}) == 1
    assert sum(e["deficit"] for e in evs) == D("1200")
    assert all(e["insurance_fund_delta"] == e["liquidation_fee"] - e["deficit"] for e in evs)
    # reparto pro-rata por nocional ejecutado (89 900 vs 32 900)
    assert evs[0]["deficit"] == pytest.approx(D("1200") * D("89900") / D("122800"), abs=D("0.00000001"))
    assert fund_balance(clean_db) == D("8800")


# ----------------------------------------------------------------------------- concurrencia y fail-safes
def test_concurrent_workers_liquidate_exactly_once(clean_db):
    seed_insurance_fund(clean_db, 50_000)
    uid = whale(clean_db)
    store = make_store({"BTC": {"mark": 89_000, "bids": [(88_900, 5.0)]}})
    executors = [LiquidationExecutor(clean_db, store) for _ in range(8)]
    barrier = threading.Barrier(len(executors))
    results = []

    def worker(ex):
        barrier.wait()
        results.append(ex.liquidate_account(uid))

    threads = [threading.Thread(target=worker, args=(e,)) for e in executors]
    [t.start() for t in threads]
    [t.join() for t in threads]
    statuses = [r.status for r in results]
    assert statuses.count(LiquidationStatus.LIQUIDATED) == 1
    assert set(statuses) <= {LiquidationStatus.LIQUIDATED, LiquidationStatus.SKIPPED_LOCKED,
                             LiquidationStatus.ACCOUNT_INACTIVE}
    assert fetch(clean_db, "SELECT count(*) AS n FROM liquidation_events")[0]["n"] == 1
    assert fetch(clean_db, "SELECT count(*) AS n FROM insurance_fund_ledger WHERE entry_type = 'BANKRUPTCY_DEFICIT'")[0]["n"] == 1
    assert fund_balance(clean_db) == D("50000") - D("1350")  # exec 88 900 => equity −1 350


def test_stale_prices_block_liquidation(clean_db):
    uid = whale(clean_db)
    store = make_store({"BTC": {"mark": 80_000, "bids": BTC_DEEP_BIDS}}, connected=False)
    ex = LiquidationExecutor(clean_db, store)
    assert ex.run_cycle().outcomes == ()                       # el screening tampoco la marca
    assert ex.liquidate_account(uid).status is LiquidationStatus.STALE_PRICES
    assert account_row(clean_db, uid)["account_status"] == "ACTIVE"


def test_recovery_between_screening_and_lock(clean_db):
    uid = whale(clean_db)
    ex = LiquidationExecutor(clean_db, make_store({"BTC": {"mark": 92_000, "bids": BTC_DEEP_BIDS}}))
    assert uid in ex.calculator.snapshot().liquidatable().index
    with clean_db.transaction() as cur:  # depósito de emergencia antes de que el liquidador adquiera el lock
        cur.execute("UPDATE users_collateral SET deposited_collateral = 20000 WHERE user_id = %s", (uid,))
    assert ex.liquidate_account(uid).status is LiquidationStatus.NOT_LIQUIDATABLE


def test_frozen_account_is_never_touched(clean_db):
    uid = seed_account(clean_db, "frozen", "WHALE_AGGRESSIVE", 100.0, [("BTC", 1.0, 95_000.0, 20, 0.025)], status="FROZEN")
    ex = LiquidationExecutor(clean_db, make_store({"BTC": {"mark": 80_000, "bids": BTC_DEEP_BIDS}}))
    assert ex.run_cycle().outcomes == ()
    assert ex.liquidate_account(uid).status is LiquidationStatus.ACCOUNT_INACTIVE


# ----------------------------------------------------------------------------- liquidez / estrés
def test_thin_book_backstop_and_stress_haircut(clean_db):
    store = make_store({"BTC": {"mark": 89_000, "bids": [(88_900, 0.5)]}})
    a = whale(clean_db)
    normal = LiquidationExecutor(clean_db, store, LiquidationConfig(backstop_discount_bps=100)).liquidate_account(a)
    f = normal.fills[0]
    assert f.source == "L2_BOOK+BACKSTOP" and f.book_fill_fraction == pytest.approx(0.5)
    assert f.execution_price == D("88505")  # 0.5·88 900 + 0.5·(89 000·0.99)

    b = seed_account(clean_db, "whale2", "WHALE_AGGRESSIVE", 4_750.0, [("BTC", 1.0, 95_000.0, 20, 0.025)])
    cfg = LiquidationConfig(liquidity_factor=0.5, regime=MarketRegime.STRESS_MONTE_CARLO)
    stressed = LiquidationExecutor(clean_db, store, cfg).liquidate_account(b)
    g = stressed.fills[0]
    assert g.book_fill_fraction == pytest.approx(0.25) and g.execution_price < f.execution_price
    assert stressed.deficit > normal.deficit
    ev = fetch(clean_db, "SELECT market_regime, metadata FROM liquidation_events WHERE user_id = %s", (b,))[0]
    assert ev["market_regime"] == "STRESS_MONTE_CARLO" and ev["metadata"]["liquidity_factor"] == 0.5


# ----------------------------------------------------------------------------- modo live (cliente simulado)
class FakeHyperliquid:
    def __init__(self, status=OrderStatus.FILLED, px="88950", exc=None):
        self.status, self.px, self.exc, self.calls = status, D(px), exc, []

    def place_market_order(self, coin, side, size, *, slippage, reference_price, reduce_only, cloid):
        self.calls.append(dict(coin=coin, side=side, size=size, slippage=slippage, ref=reference_price,
                               reduce_only=reduce_only, cloid=cloid))
        if self.exc:
            raise self.exc
        filled = D(size) if self.status is OrderStatus.FILLED else D(0)
        return OrderResult(coin=coin, side=side, kind=OrderKind.MARKET, requested_size=D(size),
                           limit_price=D(reference_price), status=self.status, cloid=cloid, reduce_only=True,
                           oid=4242, filled_size=filled, avg_fill_price=self.px if filled else None,
                           error=None if filled else "Order could not immediately match")


def test_live_mode_uses_exchange_fills(clean_db):
    uid = whale(clean_db)
    fake = FakeHyperliquid(px="88950")
    cfg = LiquidationConfig(mode=ExecutionMode.LIVE_TESTNET)
    out = LiquidationExecutor(clean_db, make_store({"BTC": {"mark": 89_000}}), cfg,
                              client_resolver=lambda acc: fake).liquidate_account(uid)
    assert out.status is LiquidationStatus.LIQUIDATED
    call = fake.calls[0]
    assert call["side"] is Side.SHORT and call["reduce_only"] and call["size"] == D("1")
    assert call["cloid"].startswith("0x") and len(call["cloid"]) == 34
    assert out.fills[0].execution_price == D("88950") and out.fills[0].source == "EXCHANGE"
    assert out.deficit == D("1300")
    ev = fetch(clean_db, "SELECT metadata FROM liquidation_events")[0]
    assert ev["metadata"]["exchange_oid"] == 4242 and ev["metadata"]["cloid"] == call["cloid"]


@pytest.mark.parametrize("fake", [FakeHyperliquid(status=OrderStatus.REJECTED), FakeHyperliquid(exc=TimeoutError("net"))])
def test_live_execution_failure_rolls_back_everything(clean_db, fake):
    seed_insurance_fund(clean_db, 1_000)
    uid = whale(clean_db)
    cfg = LiquidationConfig(mode=ExecutionMode.LIVE_TESTNET)
    out = LiquidationExecutor(clean_db, make_store({"BTC": {"mark": 89_000}}), cfg,
                              client_resolver=lambda acc: fake).liquidate_account(uid)
    assert out.status is LiquidationStatus.EXECUTION_FAILED
    assert account_row(clean_db, uid)["account_status"] == "ACTIVE"
    assert fetch(clean_db, "SELECT status FROM open_positions")[0]["status"] == "OPEN"
    assert fetch(clean_db, "SELECT count(*) AS n FROM liquidation_events")[0]["n"] == 0
    assert fund_balance(clean_db) == D("1000")


def test_live_mode_requires_client_resolver(clean_db):
    with pytest.raises(ValueError):
        LiquidationExecutor(clean_db, make_store({"BTC": {"mark": 1}}), LiquidationConfig(mode=ExecutionMode.LIVE_TESTNET))


# ----------------------------------------------------------------------------- end-to-end con los 3 agentes
def test_cycle_with_three_agent_profiles(clean_db):
    seed_insurance_fund(clean_db, 25_000)
    w = whale(clean_db)
    arb = seed_account(clean_db, "arbitrageur", "ARBITRAGEUR_NEUTRAL", 60_000.0,
                       [("BTC", 1.0, 95_000.0, 3, 0.025), ("ETH", -27.0, 3_500.0, 3, 0.02)])
    manip = seed_account(clean_db, "manipulator", "MARKET_MANIPULATOR", 500_000.0, [("BTC", -5.0, 95_000.0, 2, 0.025)])
    store = make_store({"BTC": {"mark": 88_000, "bids": [(87_950, 10.0)]},
                        "ETH": {"mark": 3_150, "bids": [(3_145, 100.0)]}})
    report = LiquidationExecutor(clean_db, store).run_cycle()
    assert [o.user_id for o in report.liquidated] == [w]
    snap = report.snapshot
    assert not snap.account(arb)["is_liquidatable"] and not snap.account(manip)["is_liquidatable"]
    assert snap.account(arb)["unrealized_pnl"] == pytest.approx(-7_000 + 27 * 350)  # netting cross
    assert fund_balance(clean_db) == D("25000") - report.liquidated[0].deficit


def test_run_forever_background_loop(clean_db):
    uid = whale(clean_db)
    ex = LiquidationExecutor(clean_db, make_store({"BTC": {"mark": 89_000, "bids": [(88_900, 5.0)]}}))
    stop = threading.Event()
    t = threading.Thread(target=ex.run_forever, args=(stop, 0.05), daemon=True)
    t.start()
    for _ in range(100):
        if account_row(clean_db, uid)["account_status"] == "LIQUIDATED":
            break
        threading.Event().wait(0.02)
    stop.set()
    t.join(2)
    assert not t.is_alive() and account_row(clean_db, uid)["account_status"] == "LIQUIDATED"
