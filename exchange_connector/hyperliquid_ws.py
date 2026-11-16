"""
exchange_connector/hyperliquid_ws.py
====================================

Feed de microestructura en tiempo real de Hyperliquid (asyncio + ``websockets``).

Canales suscritos por activo (``BTC``, ``ETH`` por defecto)
-----------------------------------------------------------
* ``l2Book``          -> libro de órdenes L2 agregado (bids/asks: px, sz, nº de órdenes).
* ``activeAssetCtx``  -> **mark price**, **oracle price**, mid, **funding rate** (horario),
                         premium, open interest e impact prices.

Arquitectura de concurrencia
----------------------------
::

    [event loop asyncio · hilo "hl-ws"]                 [Risk Engine · N hilos síncronos]
      reader ─parse─> MarketDataStore.update_*()  ──>   store.get("BTC").mark_price
      heartbeat (ping app-level)                        store.mark_prices(["BTC","ETH"]) -> np.ndarray
      watchdog (silencio > N s => reconexión)           store.wait_for_update(version, timeout)

``MarketDataStore`` aplica **copy-on-write**: cada actualización construye una
instantánea INMUTABLE nueva (dataclass ``frozen`` + arrays NumPy de solo lectura) y la
publica con un intercambio de referencia bajo un lock de sección crítica mínima. Los
lectores obtienen una instantánea coherente sin copiar datos y sin bloquear al feed:
nunca ven un libro "a medio actualizar".

Resiliencia
-----------
* Reconexión automática con backoff exponencial + full jitter (se resetea tras una sesión estable).
* Heartbeat de aplicación (``{"method":"ping"}``): Hyperliquid cierra conexiones inactivas ~60 s.
* Watchdog: si no llega ningún mensaje en ``stale_timeout_s`` se fuerza la reconexión (conexiones "zombi").
* Re-suscripción completa tras cada reconexión; el store marca los datos como no fiables mientras tanto.
* Mensajes malformados se descartan y contabilizan sin tumbar la conexión; updates fuera de orden se ignoran.

Uso desde código síncrono (el motor de riesgo)::

    feed = HyperliquidWebSocket(coins=("BTC", "ETH"))
    feed.start_in_background()
    feed.store.wait_until_ready(timeout=15)
    snap = feed.store.get("BTC")
    snap.mark_price, snap.oracle_price, snap.funding_rate_hourly, snap.book.best_bid
    feed.stop_background()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import structlog
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, WebSocketException

from exchange_connector.network import Network, resolve_network

__all__ = [
    "L2BookSnapshot",
    "AssetContext",
    "MarketSnapshot",
    "MarketDataStore",
    "FeedMetrics",
    "HyperliquidWebSocket",
    "StaleConnectionError",
]

log = structlog.get_logger(__name__)

Listener = Callable[[str, str, "MarketSnapshot"], None]  # (channel, coin, snapshot)


class StaleConnectionError(ConnectionError):
    """La conexión sigue abierta pero no recibe datos (watchdog)."""


# =============================================================================
# Modelos inmutables
# =============================================================================
def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


_EMPTY_LEVELS = _readonly(np.empty((0, 3), dtype=np.float64))


@dataclass(frozen=True, slots=True)
class L2BookSnapshot:
    """
    Libro L2 inmutable. ``bids``/``asks``: arrays (N, 3) de solo lectura con columnas
    ``[price, size, n_orders]``; bids en orden descendente de precio, asks ascendente.
    """

    coin: str
    exchange_time_ms: int
    received_at: float          # time.time() local
    bids: np.ndarray = field(repr=False)
    asks: np.ndarray = field(repr=False)

    @property
    def best_bid(self) -> float | None:
        return float(self.bids[0, 0]) if len(self.bids) else None

    @property
    def best_ask(self) -> float | None:
        return float(self.asks[0, 0]) if len(self.asks) else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_bps(self) -> float | None:
        mid = self.mid
        return (self.best_ask - self.best_bid) / mid * 1e4 if mid else None  # type: ignore[operator]

    @property
    def is_crossed(self) -> bool:
        return self.best_bid is not None and self.best_ask is not None and self.best_bid >= self.best_ask

    def depth_notional(self, side: str, band_bps: float = 50.0) -> float:
        """Liquidez (USD) disponible en ``side`` ('bid'|'ask') dentro de ±``band_bps`` del mid."""
        mid = self.mid
        if mid is None:
            return 0.0
        levels = self.bids if side == "bid" else self.asks
        bound = mid * (1 - band_bps / 1e4) if side == "bid" else mid * (1 + band_bps / 1e4)
        mask = levels[:, 0] >= bound if side == "bid" else levels[:, 0] <= bound
        return float(np.sum(levels[mask, 0] * levels[mask, 1]))

    def simulate_market_order(self, side: str, size: float) -> tuple[float | None, float]:
        """
        Recorre el libro (vectorizado) para estimar el **precio medio de ejecución** de una
        orden market de ``size`` unidades. ``side='sell'`` consume bids (p.ej. liquidar un largo).

        Devuelve ``(vwap, filled_size)``; ``filled_size < size`` indica liquidez insuficiente
        en el libro visible -> insumo directo para el slippage del motor de liquidación.
        """
        if size <= 0:
            return None, 0.0
        levels = self.bids if side == "sell" else self.asks
        if not len(levels):
            return None, 0.0
        cum = np.cumsum(levels[:, 1])
        prev = np.concatenate(([0.0], cum[:-1]))
        take = np.clip(size - prev, 0.0, levels[:, 1])
        filled = float(take.sum())
        if filled == 0:
            return None, 0.0
        return float(np.dot(take, levels[:, 0]) / filled), filled

    def slippage_bps(self, side: str, size: float) -> float | None:
        vwap, filled = self.simulate_market_order(side, size)
        mid = self.mid
        if vwap is None or mid is None:
            return None
        return abs(vwap - mid) / mid * 1e4


@dataclass(frozen=True, slots=True)
class AssetContext:
    coin: str
    mark_price: float
    oracle_price: float
    mid_price: float | None
    funding_rate_hourly: float       # Hyperliquid liquida funding cada hora
    premium: float | None
    open_interest: float
    day_notional_volume: float
    impact_bid: float | None
    impact_ask: float | None
    received_at: float

    @property
    def funding_rate_annualized(self) -> float:
        return self.funding_rate_hourly * 24 * 365

    @property
    def mark_oracle_basis_bps(self) -> float:
        """Desviación mark vs oracle: alerta temprana de manipulación / dislocación."""
        return (self.mark_price - self.oracle_price) / self.oracle_price * 1e4


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Vista consolidada e inmutable de un activo (libro + contexto)."""

    coin: str
    version: int = 0
    book: L2BookSnapshot | None = None
    ctx: AssetContext | None = None

    @property
    def mark_price(self) -> float | None:
        return self.ctx.mark_price if self.ctx else None

    @property
    def oracle_price(self) -> float | None:
        return self.ctx.oracle_price if self.ctx else None

    @property
    def funding_rate_hourly(self) -> float | None:
        return self.ctx.funding_rate_hourly if self.ctx else None

    @property
    def is_complete(self) -> bool:
        return self.book is not None and self.ctx is not None

    @property
    def last_update(self) -> float:
        return max(self.book.received_at if self.book else 0.0, self.ctx.received_at if self.ctx else 0.0)

    def age_s(self, now: float | None = None) -> float:
        last = self.last_update
        return math.inf if last == 0 else (now or time.time()) - last


# =============================================================================
# Store thread-safe
# =============================================================================
class MarketDataStore:
    """
    Último estado de mercado por activo, compartido entre el hilo del feed y el motor.

    * Escritura (feed): construye una instantánea nueva y la publica bajo ``_cond`` (O(1)).
    * Lectura (motor): devuelve referencias a instantáneas inmutables -> sin copias, sin
      riesgo de lecturas parciales, lock retenido solo durante un lookup de dict.
    """

    def __init__(self, coins: Iterable[str]) -> None:
        self._coins = tuple(c.upper() for c in coins)
        self._cond = threading.Condition(threading.Lock())
        self._snapshots: dict[str, MarketSnapshot] = {c: MarketSnapshot(coin=c) for c in self._coins}
        self._version = 0
        self._connected = False

    # ---------------------------------------------------------- escritura
    def _publish(self, coin: str, **changes: Any) -> MarketSnapshot:
        with self._cond:
            self._version += 1
            current = self._snapshots.get(coin) or MarketSnapshot(coin=coin)
            snap = replace(current, version=self._version, **changes)
            self._snapshots[coin] = snap
            self._cond.notify_all()
            return snap

    def update_book(self, book: L2BookSnapshot) -> MarketSnapshot | None:
        current = self._snapshots.get(book.coin)
        if current and current.book and book.exchange_time_ms < current.book.exchange_time_ms:
            return None  # update fuera de orden: se descarta
        return self._publish(book.coin, book=book)

    def update_context(self, ctx: AssetContext) -> MarketSnapshot:
        return self._publish(ctx.coin, ctx=ctx)

    def set_connected(self, connected: bool) -> None:
        with self._cond:
            self._connected = connected
            self._cond.notify_all()

    # ------------------------------------------------------------ lectura
    @property
    def coins(self) -> tuple[str, ...]:
        return self._coins

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def version(self) -> int:
        return self._version

    def get(self, coin: str) -> MarketSnapshot | None:
        with self._cond:
            return self._snapshots.get(coin.upper())

    def snapshot(self) -> dict[str, MarketSnapshot]:
        """Copia superficial coherente de todas las instantáneas (mismo instante lógico)."""
        with self._cond:
            return dict(self._snapshots)

    def _vector(self, coins: Sequence[str], getter: Callable[[MarketSnapshot], float | None]) -> np.ndarray:
        snaps = self.snapshot()
        values = [getter(snaps[c.upper()]) if c.upper() in snaps else None for c in coins]
        return np.array([np.nan if v is None else v for v in values], dtype=np.float64)

    def mark_prices(self, coins: Sequence[str]) -> np.ndarray:
        """Vector de mark prices alineado con ``coins`` (NaN si falta) para el cálculo vectorizado de uPnL."""
        return self._vector(coins, lambda s: s.mark_price)

    def oracle_prices(self, coins: Sequence[str]) -> np.ndarray:
        return self._vector(coins, lambda s: s.oracle_price)

    def funding_rates(self, coins: Sequence[str]) -> np.ndarray:
        return self._vector(coins, lambda s: s.funding_rate_hourly)

    def is_fresh(self, coin: str, max_age_s: float = 5.0) -> bool:
        """``True`` si hay conexión y los datos del activo tienen menos de ``max_age_s``."""
        snap = self.get(coin)
        return bool(self._connected and snap and snap.is_complete and snap.age_s() <= max_age_s)

    def wait_for_update(self, after_version: int, timeout: float | None = None) -> int:
        """Bloquea hasta que la versión global supere ``after_version``. Devuelve la versión actual."""
        with self._cond:
            self._cond.wait_for(lambda: self._version > after_version, timeout=timeout)
            return self._version

    def wait_until_ready(self, coins: Iterable[str] | None = None, timeout: float | None = None) -> bool:
        """Bloquea hasta tener libro + contexto de todos los activos. ``False`` si expira el timeout."""
        targets = [c.upper() for c in (coins or self._coins)]
        with self._cond:
            return self._cond.wait_for(
                lambda: all(self._snapshots.get(c) and self._snapshots[c].is_complete for c in targets),
                timeout=timeout,
            )


# =============================================================================
# Parsers (payload Hyperliquid -> modelos)
# =============================================================================
def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"valor no finito: {value!r}")
    return result


def _levels(raw_levels: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not raw_levels:
        return _EMPTY_LEVELS
    arr = np.array([(float(l["px"]), float(l["sz"]), float(l.get("n", 0))) for l in raw_levels], dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr[:, 0] <= 0) or np.any(arr[:, 1] < 0):
        raise ValueError("niveles L2 con precio/tamaño inválido")
    return _readonly(arr)


def parse_l2_book(data: Mapping[str, Any], received_at: float) -> L2BookSnapshot:
    """``{"coin":"BTC","time":1708622398623,"levels":[[{px,sz,n}...bids],[...asks]]}``"""
    bids_raw, asks_raw = data["levels"]
    return L2BookSnapshot(
        coin=str(data["coin"]).upper(),
        exchange_time_ms=int(data["time"]),
        received_at=received_at,
        bids=_levels(bids_raw),
        asks=_levels(asks_raw),
    )


def parse_asset_ctx(data: Mapping[str, Any], received_at: float) -> AssetContext:
    """``{"coin":"BTC","ctx":{"markPx","oraclePx","midPx","funding","premium","openInterest",...}}``"""
    ctx = data["ctx"]
    mark, oracle = _f(ctx["markPx"]), _f(ctx["oraclePx"])
    if not mark or not oracle or mark <= 0 or oracle <= 0:
        raise ValueError(f"mark/oracle inválidos: {mark}/{oracle}")
    impact = ctx.get("impactPxs") or (None, None)
    return AssetContext(
        coin=str(data["coin"]).upper(),
        mark_price=mark,
        oracle_price=oracle,
        mid_price=_f(ctx.get("midPx")),
        funding_rate_hourly=_f(ctx.get("funding")) or 0.0,
        premium=_f(ctx.get("premium")),
        open_interest=_f(ctx.get("openInterest")) or 0.0,
        day_notional_volume=_f(ctx.get("dayNtlVlm")) or 0.0,
        impact_bid=_f(impact[0]),
        impact_ask=_f(impact[1]),
        received_at=received_at,
    )


# =============================================================================
# Métricas
# =============================================================================
@dataclass(slots=True)
class FeedMetrics:
    messages: int = 0
    book_updates: int = 0
    ctx_updates: int = 0
    parse_errors: int = 0
    out_of_order: int = 0
    crossed_books: int = 0
    reconnects: int = 0
    last_message_at: float = 0.0
    last_pong_rtt_ms: float | None = None
    connected_since: float | None = None
    confirmed_subscriptions: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages, "book_updates": self.book_updates, "ctx_updates": self.ctx_updates,
            "parse_errors": self.parse_errors, "out_of_order": self.out_of_order,
            "crossed_books": self.crossed_books, "reconnects": self.reconnects,
            "last_pong_rtt_ms": self.last_pong_rtt_ms,
            "subscriptions": sorted(self.confirmed_subscriptions),
        }


# =============================================================================
# Cliente WebSocket
# =============================================================================
class HyperliquidWebSocket:
    """Feed asíncrono con reconexión automática que alimenta un ``MarketDataStore``."""

    def __init__(
        self,
        coins: Sequence[str] = ("BTC", "ETH"),
        *,
        store: MarketDataStore | None = None,
        network: Network | str | None = None,
        url: str | None = None,
        ping_interval_s: float = 20.0,
        stale_timeout_s: float = 30.0,
        open_timeout_s: float = 10.0,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 30.0,
        stable_after_s: float = 60.0,
        l2_n_sig_figs: int | None = None,
    ) -> None:
        if not coins:
            raise ValueError("Se requiere al menos un activo")
        self.coins = tuple(c.upper() for c in coins)
        self.store = store or MarketDataStore(self.coins)
        self.url = url or resolve_network(network).ws_url
        self.ping_interval_s = ping_interval_s
        self.stale_timeout_s = stale_timeout_s
        self.open_timeout_s = open_timeout_s
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.stable_after_s = stable_after_s
        self.l2_n_sig_figs = l2_n_sig_figs
        self.metrics = FeedMetrics()

        self._listeners: list[Listener] = []
        self._stop: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ping_sent_at: float | None = None
        self._log = log.bind(url=self.url, coins=list(self.coins))

    # ------------------------------------------------------------ API pública
    def add_listener(self, listener: Listener) -> None:
        """Callback síncrono ``(channel, coin, snapshot)`` tras cada update. Debe ser NO bloqueante."""
        self._listeners.append(listener)

    def subscriptions(self) -> list[dict[str, Any]]:
        subs: list[dict[str, Any]] = []
        for coin in self.coins:
            book: dict[str, Any] = {"type": "l2Book", "coin": coin}
            if self.l2_n_sig_figs:
                book["nSigFigs"] = self.l2_n_sig_figs
            subs.append(book)
            subs.append({"type": "activeAssetCtx", "coin": coin})
        return subs

    async def run(self) -> None:
        """Bucle principal: conecta, suscribe, consume y reconecta hasta ``stop()``."""
        if self._stop is None:
            self._stop = asyncio.Event()
        attempt = 0
        self._log.info("ws.feed.starting")
        while not self._stop.is_set():
            session_started = time.monotonic()
            try:
                async with connect(
                    self.url,
                    open_timeout=self.open_timeout_s,
                    ping_interval=None,       # heartbeat de aplicación propio (protocolo de HL)
                    close_timeout=5,
                    max_size=2**22,
                    max_queue=1024,
                ) as ws:
                    await self._session(ws)
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, StaleConnectionError) as exc:
                self._log.warning("ws.connection.lost", reason=f"{type(exc).__name__}: {exc}")
            except (OSError, asyncio.TimeoutError, InvalidHandshake, WebSocketException) as exc:
                self._log.warning("ws.connection.failed", reason=f"{type(exc).__name__}: {exc}")
            except Exception:  # noqa: BLE001 - el feed no debe morir por un bug inesperado
                self._log.exception("ws.connection.unexpected_error")
            finally:
                self.store.set_connected(False)
                self.metrics.connected_since = None
                self.metrics.confirmed_subscriptions.clear()

            if self._stop.is_set():
                break
            if time.monotonic() - session_started >= self.stable_after_s:
                attempt = 0  # la sesión fue estable: reinicia el backoff
            attempt += 1
            self.metrics.reconnects += 1
            delay = random.uniform(0, min(self.backoff_max_s, self.backoff_base_s * 2 ** (attempt - 1)))
            self._log.info("ws.reconnect.scheduled", attempt=attempt, delay_s=round(delay, 3))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=delay)  # espera interrumpible
        self._log.info("ws.feed.stopped", **self.metrics.as_dict())
        self._stop = None

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()

    # -------------------------------------------------- ejecución en segundo plano
    def start_in_background(self, *, name: str = "hl-ws") -> threading.Thread:
        """Lanza el feed en un hilo daemon con su propio event loop (para consumidores síncronos)."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("El feed ya está en ejecución")
        ready = threading.Event()

        def _target() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._stop = asyncio.Event()  # creado antes de señalizar: stop_background() nunca pierde la señal
            ready.set()
            try:
                loop.run_until_complete(self.run())
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._thread = threading.Thread(target=_target, name=name, daemon=True)
        self._thread.start()
        ready.wait()
        return self._thread

    def stop_background(self, timeout: float = 10.0) -> None:
        """Detiene el feed de forma ordenada desde cualquier hilo."""
        if self._loop is None or self._thread is None:
            return

        def _signal() -> None:
            if self._stop is not None:
                self._stop.set()

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(_signal)
        self._thread.join(timeout)
        if self._thread.is_alive():
            self._log.error("ws.feed.stop_timeout", timeout_s=timeout)

    # ------------------------------------------------------------- sesión
    async def _session(self, ws: ClientConnection) -> None:
        assert self._stop is not None
        for sub in self.subscriptions():
            await ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
        now = time.time()
        self.metrics.last_message_at = now
        self.metrics.connected_since = now
        self.store.set_connected(True)
        self._log.info("ws.connected", subscriptions=len(self.subscriptions()))

        tasks = {
            asyncio.ensure_future(self._reader(ws)),
            asyncio.ensure_future(self._heartbeat(ws)),
            asyncio.ensure_future(self._watchdog()),
            asyncio.ensure_future(self._stop.wait()),
        }
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in done:  # propaga el motivo de fin (ConnectionClosed, StaleConnectionError...)
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]

    async def _reader(self, ws: ClientConnection) -> None:
        async for raw in ws:
            self._handle_message(raw)
        raise ConnectionClosed(ws.close_rcvd, ws.close_sent)

    async def _heartbeat(self, ws: ClientConnection) -> None:
        while True:
            await asyncio.sleep(self.ping_interval_s)
            self._ping_sent_at = time.perf_counter()
            await ws.send('{"method":"ping"}')

    async def _watchdog(self) -> None:
        check_every = max(0.05, min(1.0, self.stale_timeout_s / 4))
        while True:
            await asyncio.sleep(check_every)
            silence = time.time() - self.metrics.last_message_at
            if silence > self.stale_timeout_s:
                raise StaleConnectionError(f"sin mensajes durante {silence:.1f}s")

    # ----------------------------------------------------------- dispatch
    def _handle_message(self, raw: str | bytes) -> None:
        now = time.time()
        self.metrics.messages += 1
        self.metrics.last_message_at = now
        try:
            msg = json.loads(raw)
            channel = msg.get("channel")
            data = msg.get("data")
            if channel == "l2Book":
                self._on_book(parse_l2_book(data, now))
            elif channel == "activeAssetCtx":
                self._on_ctx(parse_asset_ctx(data, now))
            elif channel == "pong":
                if self._ping_sent_at is not None:
                    self.metrics.last_pong_rtt_ms = round((time.perf_counter() - self._ping_sent_at) * 1e3, 2)
            elif channel == "subscriptionResponse":
                sub = (data or {}).get("subscription", {})
                self.metrics.confirmed_subscriptions.add(f"{sub.get('type')}:{sub.get('coin')}")
                self._log.debug("ws.subscription.confirmed", subscription=sub)
            elif channel == "error":
                self._log.error("ws.server_error", detail=str(data)[:500])
            else:
                self._log.debug("ws.message.ignored", channel=channel)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            self.metrics.parse_errors += 1
            self._log.warning("ws.message.parse_error", error=f"{type(exc).__name__}: {exc}", raw=str(raw)[:300])

    def _on_book(self, book: L2BookSnapshot) -> None:
        if book.coin not in self.store.coins:
            return
        if book.is_crossed:
            self.metrics.crossed_books += 1
            self._log.warning("ws.l2.crossed_book", coin=book.coin, bid=book.best_bid, ask=book.best_ask)
        snap = self.store.update_book(book)
        if snap is None:
            self.metrics.out_of_order += 1
            return
        self.metrics.book_updates += 1
        self._notify("l2Book", snap)

    def _on_ctx(self, ctx: AssetContext) -> None:
        if ctx.coin not in self.store.coins:
            return
        snap = self.store.update_context(ctx)
        self.metrics.ctx_updates += 1
        self._notify("activeAssetCtx", snap)

    def _notify(self, channel: str, snap: MarketSnapshot) -> None:
        for listener in self._listeners:
            try:
                listener(channel, snap.coin, snap)
            except Exception:  # noqa: BLE001 - un listener defectuoso no tumba el feed
                self._log.exception("ws.listener.error", listener=getattr(listener, "__qualname__", repr(listener)))


# =============================================================================
# Demo:  python -m exchange_connector.hyperliquid_ws
# =============================================================================
if __name__ == "__main__":  # pragma: no cover
    from common.logging_config import configure_logging

    configure_logging()
    feed = HyperliquidWebSocket(coins=("BTC", "ETH"))
    feed.start_in_background()
    try:
        if not feed.store.wait_until_ready(timeout=20):
            log.error("demo.not_ready", **feed.metrics.as_dict())
        for _ in range(10):
            time.sleep(2)
            for coin, s in feed.store.snapshot().items():
                if not s.is_complete:
                    continue
                log.info(
                    "market",
                    coin=coin,
                    mark=s.mark_price,
                    oracle=s.oracle_price,
                    basis_bps=round(s.ctx.mark_oracle_basis_bps, 2),
                    funding_1h=s.funding_rate_hourly,
                    bid=s.book.best_bid,
                    ask=s.book.best_ask,
                    spread_bps=round(s.book.spread_bps or 0, 2),
                    slippage_1btc_bps=round(s.book.slippage_bps("sell", 1.0) or 0, 2),
                    age_ms=round(s.age_s() * 1e3, 1),
                )
        log.info("demo.metrics", **feed.metrics.as_dict())
    finally:
        feed.stop_background()
