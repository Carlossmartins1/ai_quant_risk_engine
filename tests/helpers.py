"""Helpers de tests: mercado sintético (MarketDataStore real) y siembra de cuentas en PostgreSQL."""

from __future__ import annotations

import time
from typing import Iterable

from database.connection import DatabaseManager


def make_store(markets: dict, *, connected: bool = True):
    """
    ``markets = {"BTC": {"mark": 95000, "oracle": 94990, "bids": [(px, sz)...], "asks": [(px, sz)...]}}``
    """
    from exchange_connector.hyperliquid_ws import MarketDataStore, parse_asset_ctx, parse_l2_book

    store = MarketDataStore(markets.keys())
    now = time.time()
    for sym, m in markets.items():
        lv = lambda levels: [{"px": str(p), "sz": str(s), "n": 1} for p, s in levels]  # noqa: E731
        store.update_book(parse_l2_book(
            {"coin": sym, "time": int(now * 1000), "levels": [lv(m.get("bids", [])), lv(m.get("asks", []))]}, now))
        store.update_context(parse_asset_ctx(
            {"coin": sym, "ctx": {"markPx": str(m["mark"]), "oraclePx": str(m.get("oracle", m["mark"])),
                                  "funding": "0.0000125", "openInterest": "1000"}}, now))
    store.set_connected(connected)
    return store


def seed_account(db: DatabaseManager, name: str, profile: str, collateral: float,
                 positions: Iterable[tuple[str, float, float, float, float]] = (), status: str = "ACTIVE",
                 *, max_leverage: float = 20.0, max_gross_notional: float | None = None) -> int:
    """positions: (symbol, quantity, entry_price, leverage, maintenance_margin_rate)"""
    with db.transaction() as cur:
        cur.execute(
            """INSERT INTO users_collateral (agent_name, agent_profile, deposited_collateral, account_status,
                                             max_leverage, max_gross_notional_usd)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING user_id""",
            (name, profile, collateral, status, max_leverage, max_gross_notional),
        )
        user_id = cur.fetchone()["user_id"]
        for sym, qty, entry, lev, mmr in positions:
            cur.execute(
                """INSERT INTO open_positions (user_id, symbol, quantity, entry_price, leverage, maintenance_margin_rate)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (user_id, sym, qty, entry, lev, mmr),
            )
    return user_id


def seed_insurance_fund(db: DatabaseManager, amount: float) -> None:
    with db.transaction() as cur:
        cur.execute("UPDATE insurance_fund SET balance = %s WHERE fund_id = 1", (amount,))
        cur.execute(
            "INSERT INTO insurance_fund_ledger (entry_type, amount, balance_after, note) "
            "VALUES ('CAPITAL_INJECTION', %s, %s, 'seed')",
            (amount, amount),
        )
