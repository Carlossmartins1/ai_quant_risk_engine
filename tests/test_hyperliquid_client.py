"""
Tests del cliente REST con dobles del SDK (sin red). Reproducen los payloads reales de
Hyperliquid para validar redondeo, controles pre-trade, parseo e idempotencia por cloid.
"""

from __future__ import annotations

import threading
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal

import pytest
import requests
from hyperliquid.utils.error import ClientError

from exchange_connector.hyperliquid_client import (
    AccountState,
    ConfigurationError,
    HyperliquidClient,
    HyperliquidConfig,
    OrderStatus,
    OrderValidationError,
    Side,
    TimeInForce,
    UnknownAssetError,
    round_price,
    round_size,
)
from exchange_connector.network import MainnetNotAllowedError, Network

KEY = "0x" + "11" * 32
META = {
    "universe": [
        {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
        {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
        {"name": "SOL", "szDecimals": 2, "maxLeverage": 20},
    ]
}
USER_STATE = {
    "assetPositions": [
        {"type": "oneWay", "position": {
            "coin": "BTC", "szi": "0.2", "entryPx": "95000.0", "positionValue": "18200.0",
            "unrealizedPnl": "-800.0", "returnOnEquity": "-0.84", "liquidationPx": "90500.5",
            "marginUsed": "910.0", "maxLeverage": 40, "leverage": {"type": "cross", "value": 20},
            "cumFunding": {"allTime": "12.5", "sinceOpen": "3.25", "sinceChange": "1.0"}}},
        {"type": "oneWay", "position": {
            "coin": "ETH", "szi": "-2.5", "entryPx": "3500.0", "positionValue": "8625.0",
            "unrealizedPnl": "125.0", "returnOnEquity": "0.04", "liquidationPx": None,
            "marginUsed": "431.25", "maxLeverage": 25, "leverage": {"type": "cross", "value": 20},
            "cumFunding": {"allTime": "0", "sinceOpen": "0", "sinceChange": "0"}}},
    ],
    "crossMaintenanceMarginUsed": "520.0",
    "crossMarginSummary": {"accountValue": "1325.0", "totalMarginUsed": "1341.25", "totalNtlPos": "26825.0", "totalRawUsd": "-1000.0"},
    "marginSummary": {"accountValue": "1325.0", "totalMarginUsed": "1341.25", "totalNtlPos": "26825.0", "totalRawUsd": "-1000.0"},
    "time": 1731750000000,
    "withdrawable": "0.0",
}


def ok(status):
    return {"status": "ok", "response": {"type": "order", "data": {"statuses": [status]}}}


class FakeInfo:
    def __init__(self):
        self.order_status = {"status": "unknownOid"}
        self.order_status_exc = None
        self.mids = {"BTC": "95000.0", "ETH": "3500.0", "SOL": "180.0"}

    def meta(self):
        return META

    def all_mids(self):
        return self.mids

    def user_state(self, address):
        return USER_STATE

    def open_orders(self, address):
        return [{"coin": "BTC", "limitPx": "90000.0", "oid": 11, "side": "B", "sz": "0.001", "timestamp": 1, "origSz": "0.001"},
                {"coin": "ETH", "limitPx": "4000.0", "oid": 12, "side": "A", "sz": "0.01", "timestamp": 2, "origSz": "0.01"}]

    def query_order_by_cloid(self, address, cloid):
        if self.order_status_exc:
            raise self.order_status_exc
        return self.order_status

    def meta_and_asset_ctxs(self):
        ctx = {"markPx": "95010.0", "oraclePx": "95000.0", "midPx": "95005.0", "funding": "0.0000125",
               "premium": "0.0001", "openInterest": "1234.5"}
        return [META, [ctx, {**ctx, "markPx": "3501.0", "oraclePx": "3500.0"}, {**ctx, "markPx": None}]]


class FakeExchange:
    def __init__(self):
        self.calls: list[dict] = []
        self.responses: list = []
        self.call_times_ms: list[int] = []

    def order(self, name, is_buy, sz, limit_px, order_type, reduce_only=False, cloid=None):
        self.call_times_ms.append(time.time_ns() // 1_000_000)
        self.calls.append(dict(name=name, is_buy=is_buy, sz=sz, px=limit_px, ot=order_type, ro=reduce_only, cloid=str(cloid)))
        resp = self.responses.pop(0) if self.responses else ok({"resting": {"oid": len(self.calls)}})
        if isinstance(resp, BaseException):
            raise resp
        return resp

    def cancel(self, name, oid):
        return ok("success")

    def bulk_cancel(self, reqs):
        return {"status": "ok", "response": {"type": "cancel", "data": {"statuses": ["success"] * len(reqs)}}}

    def update_leverage(self, leverage, name, is_cross=True):
        self.calls.append(dict(lev=leverage, name=name, cross=is_cross))
        return {"status": "ok", "response": {"type": "default"}}


@pytest.fixture
def env():
    info, ex = FakeInfo(), FakeExchange()
    cfg = HyperliquidConfig(secret_key=KEY, retry_backoff_s=0.001, allowed_coins=("BTC", "ETH", "SOL"))
    return HyperliquidClient(cfg, info=info, exchange=ex), info, ex


# ----------------------------------------------------------------- precisión
@pytest.mark.parametrize("px,szd,mode,expected", [
    ("95123.46", 5, ROUND_DOWN, "95123"),       # 5 cifras significativas
    ("123456.7", 5, ROUND_DOWN, "123456"),      # entero: siempre válido
    ("3512.345", 4, ROUND_UP, "3512.4"),
    ("1.234567", 2, ROUND_DOWN, "1.2345"),      # 5 cifras significativas
    ("0.0001234567", 0, ROUND_DOWN, "0.000123"),  # máx 6 - szDecimals decimales
])
def test_round_price(px, szd, mode, expected):
    assert round_price(Decimal(px), szd, rounding=mode) == Decimal(expected)


def test_round_size_truncates():
    assert round_size(Decimal("0.123456789"), 5) == Decimal("0.12345")
    assert round_size(Decimal("1.99999"), 2) == Decimal("1.99")


# ------------------------------------------------------------- configuración
def test_config_guards(monkeypatch):
    with pytest.raises(ConfigurationError):
        HyperliquidConfig(secret_key="0xdeadbeef")
    with pytest.raises(ConfigurationError):
        HyperliquidConfig()  # ni clave ni dirección
    with pytest.raises(MainnetNotAllowedError):
        HyperliquidConfig(network="mainnet", secret_key=KEY)
    cfg = HyperliquidConfig(secret_key="11" * 32)
    assert cfg.secret_key.startswith("0x") and cfg.network is Network.TESTNET
    assert KEY not in repr(cfg)
    monkeypatch.setenv("HL_SECRET_KEY", KEY)
    monkeypatch.setenv("HL_ALLOWED_COINS", "btc, eth")
    assert HyperliquidConfig.from_env(load_dotenv=False).allowed_coins == ("BTC", "ETH")


def test_read_only_mode():
    cfg = HyperliquidConfig(account_address="0x" + "ab" * 20)
    client = HyperliquidClient(cfg, info=FakeInfo())
    assert client.get_account_state().equity == Decimal("1325.0")
    with pytest.raises(ConfigurationError):
        client.place_limit_order("BTC", Side.LONG, "0.001", "60000")


# -------------------------------------------------------------------- órdenes
def test_limit_order_passive_rounding_and_resting(env):
    client, _, ex = env
    r = client.place_limit_order("BTC", "LONG", "0.0012345678", "60000.99")
    assert r.status is OrderStatus.RESTING and r.oid == 1 and r.is_success
    call = ex.calls[0]
    assert call["px"] == 60000.0 and call["sz"] == 0.00123 and call["is_buy"] is True
    assert call["ot"] == {"limit": {"tif": "Gtc"}} and call["cloid"].startswith("0x") and len(call["cloid"]) == 34
    r = client.place_limit_order("ETH", Side.SHORT, "0.01", "3512.341", tif=TimeInForce.ALO)
    assert ex.calls[1]["px"] == 3512.4 and ex.calls[1]["ot"] == {"limit": {"tif": "Alo"}}


def test_market_order_filled_with_bounded_slippage(env):
    client, _, ex = env
    ex.responses = [ok({"filled": {"totalSz": "0.01", "avgPx": "95050.0", "oid": 7}})]
    r = client.place_market_order("BTC", Side.LONG, "0.01", slippage=0.01, reference_price="95000")
    assert r.status is OrderStatus.FILLED and r.avg_fill_price == Decimal("95050.0")
    assert r.signed_filled_size == Decimal("0.01")
    assert ex.calls[0]["px"] <= 95000 * 1.01 and ex.calls[0]["ot"] == {"limit": {"tif": "Ioc"}}


def test_market_order_partial_and_rejected(env):
    client, _, ex = env
    ex.responses = [
        ok({"filled": {"totalSz": "0.004", "avgPx": "3499.0", "oid": 8}}),
        ok({"error": "Order could not immediately match against any resting orders."}),
        {"status": "err", "response": "User or API Wallet 0x... does not exist."},
    ]
    r1 = client.place_market_order("ETH", Side.SHORT, "0.01")  # usa allMids
    assert r1.status is OrderStatus.PARTIALLY_FILLED and r1.signed_filled_size == Decimal("-0.004")
    assert ex.calls[0]["px"] >= 3500 * 0.99
    assert client.place_market_order("ETH", Side.SHORT, "0.01").status is OrderStatus.REJECTED
    r3 = client.place_market_order("ETH", Side.SHORT, "0.01")
    assert r3.status is OrderStatus.REJECTED and "does not exist" in r3.error


def test_pre_trade_checks(env):
    client, _, ex = env
    with pytest.raises(OrderValidationError):  # < 10 USD
        client.place_limit_order("BTC", Side.LONG, "0.0001", "60000")
    with pytest.raises(OrderValidationError):  # fat-finger
        client.place_limit_order("BTC", Side.LONG, "10", "95000")
    with pytest.raises(OrderValidationError):  # cero tras lot size
        client.place_limit_order("BTC", Side.LONG, "0.000001", "95000")
    with pytest.raises(UnknownAssetError):
        client.place_limit_order("DOGE", Side.LONG, "100", "0.2")
    with pytest.raises(OrderValidationError):
        client.set_leverage("ETH", 50)
    assert ex.calls == []


# ------------------------------------------------- idempotencia / resiliencia
def test_ambiguous_timeout_reconciled_without_resend(env):
    client, info, ex = env
    ex.responses = [requests.ReadTimeout("read timed out")]
    info.order_status = {"status": "order", "order": {"status": "filled", "statusTimestamp": 1,
                         "order": {"coin": "BTC", "oid": 99, "origSz": "0.01", "sz": "0.0"}}}
    r = client.place_market_order("BTC", Side.LONG, "0.01", reference_price="95000")
    assert r.status is OrderStatus.FILLED and r.oid == 99 and r.filled_size == Decimal("0.01")
    assert len(ex.calls) == 1  # NO se duplicó la orden


def test_ambiguous_timeout_unknown_oid_retries_same_cloid(env):
    client, info, ex = env
    ex.responses = [requests.ConnectionError("reset"), ok({"resting": {"oid": 5}})]
    r = client.place_limit_order("BTC", Side.LONG, "0.001", "60000")
    assert r.status is OrderStatus.RESTING and len(ex.calls) == 2
    assert ex.calls[0]["cloid"] == ex.calls[1]["cloid"]


def test_reconciliation_failure_returns_unknown_never_resends(env):
    client, info, ex = env
    ex.responses = [requests.ReadTimeout("t")]
    info.order_status_exc = requests.ConnectionError("down")
    r = client.place_limit_order("BTC", Side.LONG, "0.001", "60000")
    assert r.status is OrderStatus.UNKNOWN and len(ex.calls) == 1


def test_rate_limit_429_retried(env):
    client, _, ex = env
    ex.responses = [ClientError(429, None, "rate limited", None), ok({"resting": {"oid": 3}})]
    assert client.place_limit_order("BTC", Side.LONG, "0.001", "60000").oid == 3


def test_concurrent_orders_get_strictly_increasing_nonces(env):
    client, _, ex = env
    threads = [threading.Thread(target=client.place_limit_order, args=("BTC", Side.LONG, "0.001", "60000"))
               for _ in range(15)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    times = ex.call_times_ms
    assert len(times) == 15 and all(b > a for a, b in zip(times, times[1:]))


# ------------------------------------------------------------------- lecturas
def test_account_state_parsing(env):
    client, _, _ = env
    st = client.get_account_state()
    assert isinstance(st, AccountState) and len(st.positions) == 2
    btc, eth = st.positions
    assert btc.side is Side.LONG and eth.size == Decimal("-2.5") and eth.liquidation_price is None
    assert btc.mark_price == Decimal("91000") and btc.cum_funding_since_open == Decimal("3.25")
    assert st.equity == Decimal("1325.0") and st.maintenance_margin == Decimal("520.0")
    assert st.maintenance_buffer == Decimal("805.0") and not st.is_liquidatable
    assert st.unrealized_pnl == Decimal("-675.0")


def test_open_orders_cancel_leverage_contexts(env):
    client, _, ex = env
    orders = client.get_open_orders()
    assert [o.side for o in orders] == [Side.LONG, Side.SHORT]
    assert client.cancel_all_orders("BTC") == 1 and client.cancel_all_orders() == 2
    assert client.cancel_order("BTC", 11) is True
    assert client.set_leverage("BTC", 20) and ex.calls[-1] == {"lev": 20, "name": "BTC", "cross": True}
    ctxs = client.get_market_contexts(["BTC", "ETH", "SOL"])
    assert set(ctxs) == {"BTC", "ETH"}  # SOL sin markPx se omite
    assert ctxs["BTC"].funding_rate_annualized == Decimal("0.0000125") * 24 * 365
