"""
Tests del feed WebSocket contra un servidor local que imita el protocolo de Hyperliquid
(subscribe/subscriptionResponse, l2Book, activeAssetCtx, ping/pong) e inyecta fallos:
cortes de conexión, silencio (conexión zombi), mensajes corruptos, libros cruzados y
updates fuera de orden. Incluye un test de estrés de concurrencia lectores/escritor.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import numpy as np
import pytest
from websockets.asyncio.server import serve

from exchange_connector.hyperliquid_ws import (
    HyperliquidWebSocket,
    MarketDataStore,
    parse_asset_ctx,
    parse_l2_book,
)


def book_msg(coin, t, mid, crossed=False):
    bid0, ask0 = (mid + 1, mid - 1) if crossed else (mid - 0.5, mid + 0.5)
    bids = [{"px": str(bid0 - i), "sz": str(1.0 + i), "n": 3} for i in range(5)]
    asks = [{"px": str(ask0 + i), "sz": str(1.0 + i), "n": 2} for i in range(5)]
    return json.dumps({"channel": "l2Book", "data": {"coin": coin, "time": t, "levels": [bids, asks]}})


def ctx_msg(coin, mark):
    return json.dumps({"channel": "activeAssetCtx", "data": {"coin": coin, "ctx": {
        "markPx": str(mark), "oraclePx": str(mark - 2), "midPx": str(mark), "funding": "0.0000125",
        "premium": "0.0002", "openInterest": "1500.5", "dayNtlVlm": "1e9", "prevDayPx": str(mark),
        "impactPxs": [str(mark - 1), str(mark + 1)]}}})


class MockHyperliquid:
    """Servidor WS con escenario por número de conexión."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.connections = 0
        self.subscriptions: list[list[dict]] = []
        self.pings = 0

    async def handler(self, ws):
        self.connections += 1
        conn_no = self.connections
        subs: list[dict] = []
        self.subscriptions.append(subs)

        async def responder():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("method") == "subscribe":
                    subs.append(msg["subscription"])
                    await ws.send(json.dumps({"channel": "subscriptionResponse", "data": msg}))
                elif msg.get("method") == "ping":
                    self.pings += 1
                    if self.scenario(conn_no) != "silent":
                        await ws.send('{"channel":"pong"}')

        task = asyncio.ensure_future(responder())
        try:
            mode = self.scenario(conn_no)
            if mode == "silent":
                await ws.wait_closed()  # no envía nada: conexión "zombi"
                return
            t, mid = 1_000, 95_000.0
            for i in range(10_000):
                if mode == "drop" and i == 20:
                    await ws.close()  # corte abrupto
                    return
                t += 10
                mid += np.sin(i / 7)
                await ws.send(book_msg("BTC", t, mid, crossed=(mode == "anomalies" and i == 3)))
                await ws.send(ctx_msg("BTC", mid))
                await ws.send(book_msg("ETH", t, mid / 27))
                await ws.send(ctx_msg("ETH", mid / 27))
                if mode == "anomalies" and i == 5:
                    await ws.send("{not json")
                    await ws.send(json.dumps({"channel": "l2Book", "data": {"coin": "BTC"}}))
                    await ws.send(book_msg("BTC", 10, mid))  # fuera de orden
                    await ws.send(json.dumps({"channel": "error", "data": "Invalid subscription"}))
                await asyncio.sleep(0.002)
        except Exception:
            pass
        finally:
            task.cancel()


def run_scenario(scenario, duration, **ws_kwargs):
    async def main():
        server_impl = MockHyperliquid(scenario)
        async with serve(server_impl.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            feed = HyperliquidWebSocket(url=f"ws://127.0.0.1:{port}", backoff_base_s=0.05, **ws_kwargs)
            runner = asyncio.ensure_future(feed.run())
            await asyncio.sleep(duration)
            await feed.stop()
            await asyncio.wait_for(runner, 5)
            return feed, server_impl
    return asyncio.run(main())


def test_happy_path_subscriptions_and_store():
    feed, srv = run_scenario(lambda n: "normal", 1.0, ping_interval_s=0.2)
    assert srv.connections == 1
    assert {(s["type"], s["coin"]) for s in srv.subscriptions[0]} == {
        ("l2Book", "BTC"), ("activeAssetCtx", "BTC"), ("l2Book", "ETH"), ("activeAssetCtx", "ETH")}
    assert feed.metrics.book_updates > 50 and feed.metrics.ctx_updates > 50
    assert srv.pings >= 3 and feed.metrics.last_pong_rtt_ms is not None
    snap = feed.store.get("BTC")
    assert snap.is_complete and snap.book.best_bid < snap.book.best_ask
    assert snap.oracle_price == pytest.approx(snap.mark_price - 2)
    assert not feed.store.connected  # tras stop()


def test_reconnects_after_abrupt_drop_and_resubscribes():
    feed, srv = run_scenario(lambda n: "drop" if n <= 2 else "normal", 1.5)
    assert srv.connections >= 3 and feed.metrics.reconnects >= 2
    assert all(len(subs) == 4 for subs in srv.subscriptions)  # re-suscripción completa cada vez
    assert feed.store.get("ETH").is_complete


def test_watchdog_kills_zombie_connection():
    feed, srv = run_scenario(lambda n: "silent" if n == 1 else "normal", 2.0, stale_timeout_s=0.4, ping_interval_s=0.1)
    assert srv.connections >= 2 and feed.metrics.book_updates > 0


def test_anomalies_do_not_kill_feed():
    feed, srv = run_scenario(lambda n: "anomalies", 0.8)
    m = feed.metrics
    assert srv.connections == 1
    assert m.parse_errors == 2 and m.out_of_order == 1 and m.crossed_books == 1
    assert m.book_updates > 20


def test_market_order_simulation_against_book():
    book = parse_l2_book(json.loads(book_msg("BTC", 1, 100.0))["data"], time.time())
    # bids: 99.5x1, 98.5x2, 97.5x3 ... vender 2.5 => 1@99.5 + 1.5@98.5
    vwap, filled = book.simulate_market_order("sell", 2.5)
    assert filled == pytest.approx(2.5) and vwap == pytest.approx((99.5 + 1.5 * 98.5) / 2.5)
    vwap, filled = book.simulate_market_order("buy", 1_000)  # más que el libro visible
    assert filled == pytest.approx(15.0)
    assert book.slippage_bps("sell", 2.5) > 0 and book.spread_bps == pytest.approx(100.0)
    assert book.depth_notional("bid", band_bps=60) == pytest.approx(99.5 * 1)
    with pytest.raises(ValueError):
        book.bids[0, 0] = 1.0  # arrays de solo lectura


def test_parse_rejects_invalid_ctx():
    with pytest.raises(ValueError):
        parse_asset_ctx({"coin": "BTC", "ctx": {"markPx": "0", "oraclePx": "1"}}, 0.0)


def test_store_concurrency_stress():
    """1 escritor a máxima velocidad + 8 lectores: sin excepciones y con instantáneas coherentes."""
    store = MarketDataStore(["BTC", "ETH"])
    stop = threading.Event()
    errors: list[str] = []
    reads = [0] * 8

    def writer():
        t = 0
        while not stop.is_set():
            t += 1
            mid = 95_000 + (t % 100)
            store.update_book(parse_l2_book(json.loads(book_msg("BTC", t, mid))["data"], time.time()))
            store.update_context(parse_asset_ctx(json.loads(ctx_msg("BTC", mid))["data"], time.time()))

    def reader(idx):
        last_version = 0
        while not stop.is_set():
            snap = store.get("BTC")
            if snap.book is not None:
                b = snap.book
                if not (b.bids[0, 0] < b.asks[0, 0] and b.asks[0, 0] - b.bids[0, 0] == pytest.approx(1.0)):
                    errors.append("libro incoherente")
            if snap.version < last_version:
                errors.append("versión retrocede")
            last_version = snap.version
            marks = store.mark_prices(["BTC", "ETH", "XYZ"])
            if marks.shape != (3,) or not np.isnan(marks[1]):
                errors.append("vector de marks inválido")
            reads[idx] += 1

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    time.sleep(1.5)
    stop.set()
    [t.join() for t in threads]
    assert not errors, errors[:3]
    assert store.version > 1_000 and sum(reads) > 10_000


def test_background_thread_api_for_sync_consumers():
    async def serve_forever(ready, holder):
        srv = MockHyperliquid(lambda n: "normal")
        async with serve(srv.handler, "127.0.0.1", 0) as server:
            holder["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await asyncio.sleep(3)

    ready, holder = threading.Event(), {}
    server_thread = threading.Thread(target=lambda: asyncio.run(serve_forever(ready, holder)), daemon=True)
    server_thread.start()
    ready.wait(2)

    feed = HyperliquidWebSocket(url=f"ws://127.0.0.1:{holder['port']}")
    seen = []
    feed.add_listener(lambda ch, coin, snap: seen.append(ch))
    feed.add_listener(lambda ch, coin, snap: 1 / 0)  # listener defectuoso: no debe tumbar el feed
    feed.start_in_background()
    assert feed.store.wait_until_ready(timeout=3)
    v = feed.store.version
    assert feed.store.wait_for_update(v, timeout=2) > v
    assert feed.store.is_fresh("BTC", max_age_s=1.0)
    marks = feed.store.mark_prices(["BTC", "ETH"])
    assert np.all(np.isfinite(marks)) and marks[0] > marks[1]
    t0 = time.monotonic()
    feed.stop_background(timeout=5)
    assert time.monotonic() - t0 < 2 and not feed._thread.is_alive()
    assert "l2Book" in seen and "activeAssetCtx" in seen
