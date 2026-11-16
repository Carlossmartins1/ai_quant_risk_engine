"""
Tests del cálculo vectorizado de margen (sin base de datos): casos analíticos, verificación
cruzada contra una implementación de referencia con bucles y rendimiento.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from risk_engine.margin_calculator import compute_margins, marks_from_store
from tests.helpers import make_store


def frames(accounts, positions):
    acc = pd.DataFrame(accounts, columns=["user_id", "deposited_collateral", "account_status"])
    pos = pd.DataFrame(positions, columns=["position_id", "user_id", "symbol", "quantity", "entry_price",
                                           "leverage", "maintenance_margin_rate"])
    return acc, pos


def test_single_long_analytic():
    acc, pos = frames([(1, 4_750.0, "ACTIVE")], [(10, 1, "BTC", 1.0, 95_000.0, 20.0, 0.025)])
    snap = compute_margins(acc, pos, {"BTC": 93_000.0})
    a, p = snap.account(1), snap.positions.iloc[0]
    assert a["unrealized_pnl"] == pytest.approx(-2_000)
    assert a["equity"] == pytest.approx(2_750)
    assert a["maintenance_margin"] == pytest.approx(2_325)
    assert a["margin_buffer"] == pytest.approx(425)
    assert a["health_ratio"] == pytest.approx(2_750 / 2_325)
    assert a["margin_ratio"] == pytest.approx(2_325 / 2_750)
    assert a["initial_margin"] == pytest.approx(93_000 / 20)
    assert not a["is_liquidatable"]
    assert p["liquidation_price"] == pytest.approx((95_000 - 4_750) / 0.975)
    assert p["bankruptcy_price"] == pytest.approx(90_250)
    assert p["liq_distance_pct"] == pytest.approx((93_000 - p["liquidation_price"]) / 93_000)


def test_single_short_analytic():
    acc, pos = frames([(1, 2_000.0, "ACTIVE")], [(1, 1, "ETH", -10.0, 3_500.0, 10.0, 0.02)])
    snap = compute_margins(acc, pos, {"ETH": 3_600.0})
    a, p = snap.account(1), snap.positions.iloc[0]
    assert a["equity"] == pytest.approx(1_000) and a["maintenance_margin"] == pytest.approx(720)
    assert p["liquidation_price"] == pytest.approx(37_000 / 10.2)
    assert p["bankruptcy_price"] == pytest.approx(3_700)
    # en el precio de liquidación, equity == MM
    liq = p["liquidation_price"]
    assert 2_000 - 10 * (liq - 3_500) == pytest.approx(10 * liq * 0.02)


def test_netting_cross_margin_and_boundary():
    acc, pos = frames(
        [(1, 10_000.0, "ACTIVE"), (2, 10.0, "ACTIVE")],
        [(1, 1, "BTC", 1.0, 95_000.0, 3, 0.025), (2, 1, "ETH", -27.0, 3_500.0, 3, 0.02),
         (3, 2, "SOL", 1.0, 1_000.0, 10, 0.0625)],
    )
    # usuario 1: BTC −5 000 y ETH (corto) +2 700 se compensan (netting)
    # usuario 2: frontera exacta en binario: 10 + (P − 1 000) = P/16  =>  P = 1 056, equity = MM = 66
    snap = compute_margins(acc, pos, {"BTC": 90_000.0, "ETH": 3_400.0, "SOL": 1_056.0})
    assert snap.account(1)["unrealized_pnl"] == pytest.approx(-5_000 + 2_700)
    assert snap.account(1)["net_notional"] == pytest.approx(90_000 - 27 * 3_400)
    boundary = snap.account(2)
    assert boundary["equity"] == boundary["maintenance_margin"] == 66.0
    assert boundary["is_liquidatable"]  # Equity <= MM (la igualdad liquida)
    assert not compute_margins(acc, pos, {"BTC": 1.0, "ETH": 1.0, "SOL": 1_056.5}).account(2)["is_liquidatable"]


def test_missing_price_is_fail_safe():
    acc, pos = frames(
        [(1, 100.0, "ACTIVE"), (2, 100.0, "ACTIVE")],
        [(1, 1, "BTC", 1.0, 95_000.0, 20, 0.025), (2, 1, "SOL", 10.0, 200.0, 5, 0.05),
         (3, 2, "BTC", 1.0, 95_000.0, 20, 0.025)],
    )
    snap = compute_margins(acc, pos, {"BTC": 50_000.0, "SOL": 0.0})  # SOL inválido
    assert snap.account(1)["missing_prices"] and np.isnan(snap.account(1)["equity"])
    assert not snap.account(1)["is_liquidatable"]          # NUNCA liquidar con precio desconocido
    assert snap.account(2)["is_liquidatable"]              # el resto no se ve afectado
    assert list(snap.liquidatable().index) == [2]


def test_status_filter_and_empty_accounts():
    acc, pos = frames(
        [(1, 500.0, "ACTIVE"), (2, 500.0, "FROZEN"), (3, 500.0, "ACTIVE")],
        [(1, 1, "BTC", 1.0, 95_000.0, 20, 0.025), (2, 2, "BTC", 5.0, 95_000.0, 20, 0.025)],
    )
    snap = compute_margins(acc, pos, {"BTC": 95_000.0})
    assert list(snap.accounts.index) == [1, 3]
    assert len(snap.positions) == 1
    empty = snap.account(3)
    assert empty["n_positions"] == 0 and empty["maintenance_margin"] == 0
    assert empty["health_ratio"] == np.inf and empty["margin_ratio"] == 0 and not empty["is_liquidatable"]


def _reference(acc: pd.DataFrame, pos: pd.DataFrame, marks: dict) -> dict:
    out = {}
    for _, a in acc.iterrows():
        eq, mm = a.deposited_collateral, 0.0
        for _, p in pos[pos.user_id == a.user_id].iterrows():
            m = marks[p.symbol]
            eq += p.quantity * (m - p.entry_price)
            mm += abs(p.quantity) * m * p.maintenance_margin_rate
        out[a.user_id] = (eq, mm, (pos.user_id == a.user_id).any() and eq <= mm)
    return out


def test_vectorized_matches_reference_loop():
    rng = np.random.default_rng(0)
    n_acc = 1_500
    acc = pd.DataFrame({"user_id": np.arange(1, n_acc + 1), "deposited_collateral": rng.uniform(100, 20_000, n_acc),
                        "account_status": "ACTIVE"})
    rows = []
    pid = 0
    for uid in acc.user_id:
        for sym, base in rng.permutation([("BTC", 95_000.0), ("ETH", 3_500.0), ("SOL", 200.0)])[: rng.integers(0, 4)]:
            pid += 1
            rows.append((pid, uid, sym, rng.normal(0, 2), float(base) * rng.uniform(0.9, 1.1), 10, 0.025))
    pos = pd.DataFrame(rows, columns=["position_id", "user_id", "symbol", "quantity", "entry_price", "leverage",
                                      "maintenance_margin_rate"])
    marks = {"BTC": 91_000.0, "ETH": 3_300.0, "SOL": 185.0}
    snap = compute_margins(acc, pos, marks)
    ref = _reference(acc, pos, marks)
    for uid, (eq, mm, liq) in ref.items():
        row = snap.account(uid)
        assert row["equity"] == pytest.approx(eq, rel=1e-12, abs=1e-6)
        assert row["maintenance_margin"] == pytest.approx(mm, rel=1e-12, abs=1e-9)
        assert bool(row["is_liquidatable"]) == liq

    # propiedad: en P_liq de cualquier posición, equity == MM (resto congelado)
    sample = snap.positions.dropna(subset=["liquidation_price"]).sample(50, random_state=1)
    for _, p in sample.iterrows():
        bumped = dict(marks)
        user_pos = pos[pos.user_id == p.user_id].copy()
        user_pos.loc[user_pos.position_id == p.position_id, "symbol"] = "X"
        bumped["X"] = p.liquidation_price
        s2 = compute_margins(acc[acc.user_id == p.user_id], user_pos, bumped)
        assert s2.account(p.user_id)["margin_buffer"] == pytest.approx(0, abs=1e-5)


def test_performance_vectorized():
    rng = np.random.default_rng(1)
    n_acc, n_pos = 100_000, 300_000
    acc = pd.DataFrame({"user_id": np.arange(n_acc), "deposited_collateral": rng.uniform(100, 1e5, n_acc)})
    pos = pd.DataFrame({
        "position_id": np.arange(n_pos), "user_id": rng.integers(0, n_acc, n_pos),
        "symbol": rng.choice(["BTC", "ETH"], n_pos), "quantity": rng.normal(0, 1, n_pos),
        "entry_price": rng.uniform(3_000, 95_000, n_pos), "leverage": 10.0, "maintenance_margin_rate": 0.025,
    })
    t0 = time.perf_counter()
    snap = compute_margins(acc, pos, {"BTC": 90_000.0, "ETH": 3_400.0})
    elapsed = time.perf_counter() - t0
    assert len(snap.accounts) == n_acc and elapsed < 1.0, f"{elapsed:.3f}s"


def test_marks_from_store_freshness():
    store = make_store({"BTC": {"mark": 95_000}, "ETH": {"mark": 3_500}})
    assert marks_from_store(store, ["BTC", "ETH", "SOL"]) == {"BTC": 95_000.0, "ETH": 3_500.0}
    store.set_connected(False)
    assert marks_from_store(store, ["BTC"]) == {}                       # desconectado => no fiable
    assert marks_from_store(store, ["BTC"], max_age_s=None) == {"BTC": 95_000.0}


def test_no_open_positions_at_all():
    acc, pos = frames([(1, 500.0, "ACTIVE"), (2, 0.0, "ACTIVE")], [])
    snap = compute_margins(acc, pos, {})
    assert snap.accounts["equity"].tolist() == [500.0, 0.0]
    assert snap.accounts["maintenance_margin"].dtype == np.float64 and snap.liquidatable().empty
