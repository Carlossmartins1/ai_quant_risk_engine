"""
ai_agents/shadow_market.py
==========================

Mercado sombra: la vista de mercado que consumen el motor de riesgo y los agentes durante una
simulación. Publica en un ``MarketDataStore`` real (misma interfaz que el feed de la Fase 2), de
modo que ``MarginCalculator`` y ``LiquidationExecutor`` funcionan sin cambios.

Capas de precio
---------------
::

    precio base   <- PriceSource: feed en vivo de Hyperliquid (Fase 2) o trayectoria Monte Carlo (Fase 3)
    + impacto     <- flujo agresivo del Market Manipulator: desplaza el libro y decae exponencialmente
    + liquidez    <- factor del Monte Carlo × drenaje por flujo tóxico (se recupera con el tiempo)
    = libro L2 sintético + mark price

El **oracle** NO se ve afectado por el impacto (refleja precios spot externos). El **mark** mezcla
mid del libro y oracle (``mark_oracle_weight``): con peso 0 el mark es manipulable al 100 %; con
peso 1 es inmune. Así se puede estudiar la robustez del mark frente a manipulación.

Garantía de seguridad: el flujo del manipulador SOLO existe aquí. Nunca se enruta a un exchange real.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Protocol

import numpy as np
import structlog

from exchange_connector.hyperliquid_ws import AssetContext, L2BookSnapshot, MarketDataStore

if TYPE_CHECKING:
    from quant_models.monte_carlo import ScenarioConfig, SimulationResult

__all__ = [
    "BaseQuote",
    "PriceSource",
    "MonteCarloPathSource",
    "LiveFeedSource",
    "ShadowMarketConfig",
    "ShadowMarket",
    "FlowResult",
]

log = structlog.get_logger(__name__)


# =============================================================================
# Fuentes de precio base
# =============================================================================
@dataclass(frozen=True, slots=True)
class BaseQuote:
    symbol: str
    price: float                       # precio "justo" del activo
    oracle: float
    liquidity: float = 1.0             # (0, 1]: profundidad relativa del libro
    book: L2BookSnapshot | None = None  # libro real (fuente en vivo); None => sintético
    funding_hourly: float | None = None


class PriceSource(Protocol):
    name: str

    def next_quotes(self) -> dict[str, BaseQuote] | None:
        """Cotizaciones del siguiente tick; ``{}`` = sin datos este tick; ``None`` = fuente agotada."""


class MonteCarloPathSource:
    """Reproduce paso a paso UNA trayectoria de un ``SimulationResult`` (precio + liquidez)."""

    def __init__(self, result: "SimulationResult", path_index: int, *, name: str | None = None) -> None:
        if not 0 <= path_index < result.n_paths:
            raise IndexError("path_index fuera de rango")
        self.result = result
        self.path_index = path_index
        self.name = name or f"mc:{result.config.name}#{path_index}"
        self._t = 0

    @classmethod
    def from_config(
        cls, config: "ScenarioConfig", *, quantile: float = 0.05, symbol: str = "BTC"
    ) -> "MonteCarloPathSource":
        """
        Simula ``config`` y elige la trayectoria cuyo retorno terminal de ``symbol`` está más próximo
        al cuantil pedido (0.05 => un escenario de cola: 1 de cada 20 sería peor).
        """
        from quant_models.monte_carlo import JumpDiffusionSimulator

        result = JumpDiffusionSimulator(config).run()
        returns = result.terminal_log_returns()[:, result.symbols.index(symbol)]
        target = np.quantile(returns, quantile)
        idx = int(np.argmin(np.abs(returns - target)))
        log.info("shadow.mc_path_selected", scenario=config.name, quantile=quantile, path=idx,
                 terminal_return=round(float(np.expm1(returns[idx])), 4), paths=result.n_paths,
                 elapsed_s=round(result.elapsed_s, 3))
        return cls(result, idx)

    @property
    def steps_remaining(self) -> int:
        return self.result.prices.shape[1] - self._t

    def next_quotes(self) -> dict[str, BaseQuote] | None:
        if self._t >= self.result.prices.shape[1]:
            return None
        t, p = self._t, self.path_index
        self._t += 1
        liq = float(self.result.liquidity[p, t])
        return {
            sym: BaseQuote(sym, float(self.result.prices[p, t, i]), float(self.result.prices[p, t, i]), liq)
            for i, sym in enumerate(self.result.symbols)
        }


class LiveFeedSource:
    """Precios en vivo del ``MarketDataStore`` alimentado por ``HyperliquidWebSocket`` (Fase 2)."""

    name = "live:hyperliquid"

    def __init__(self, store: MarketDataStore, *, max_age_s: float = 5.0) -> None:
        self.store = store
        self.max_age_s = max_age_s

    def next_quotes(self) -> dict[str, BaseQuote]:
        quotes: dict[str, BaseQuote] = {}
        for sym, snap in self.store.snapshot().items():
            if snap.ctx is None or not self.store.is_fresh(sym, self.max_age_s):
                continue
            quotes[sym] = BaseQuote(sym, snap.ctx.mark_price, snap.ctx.oracle_price, 1.0, snap.book,
                                    snap.ctx.funding_rate_hourly)
        return quotes


# =============================================================================
# Mercado sombra
# =============================================================================
@dataclass(frozen=True, slots=True)
class ShadowMarketConfig:
    levels: int = 25
    level_spacing_bps: float = 2.0
    base_spread_bps: float = 1.5
    depth_usd_per_level: Mapping[str, float] = field(default_factory=lambda: {"BTC": 300_000.0, "ETH": 150_000.0})
    depth_decay: float = 0.03                 # profundidad decreciente lejos del mid
    impact_half_life_ticks: float = 4.0       # resiliencia del libro tras flujo tóxico (en pasos simulados)
    liquidity_recovery_half_life_ticks: float = 8.0
    mark_oracle_weight: float = 0.5           # mark = (1−w)·mid_libro + w·oracle
    min_liquidity: float = 0.02

    def __post_init__(self) -> None:
        if not 0 <= self.mark_oracle_weight <= 1:
            raise ValueError("mark_oracle_weight en [0, 1]")


@dataclass(frozen=True, slots=True)
class FlowResult:
    symbol: str
    side: str
    requested_notional: float
    filled_quantity: float
    vwap: float | None
    impact_bps: float
    liquidity_after: float


class ShadowMarket:
    """Thread-safe: ``step`` (hilo de mercado) y ``execute_flow`` (hilo de agentes) se serializan."""

    def __init__(self, symbols: tuple[str, ...] = ("BTC", "ETH"), config: ShadowMarketConfig | None = None) -> None:
        self.config = config or ShadowMarketConfig()
        self.symbols = symbols
        self.store = MarketDataStore(symbols)
        self._lock = threading.RLock()
        self._quotes: dict[str, BaseQuote] = {}
        self._impact = {s: 0.0 for s in symbols}         # desplazamiento log del libro
        self._drain = {s: 1.0 for s in symbols}          # multiplicador de liquidez por flujo
        self._seq_ms = 0

    # ---------------------------------------------------------------- publicación
    def step(self, quotes: Mapping[str, BaseQuote]) -> None:
        """
        Avanza un paso simulado: decae impacto y drenaje (semividas en TICKS, coherentes con la escala
        temporal del Monte Carlo y no con el reloj de pared) y publica un nuevo estado de mercado.
        """
        cfg = self.config
        with self._lock:
            k_imp = 0.5 ** (1.0 / cfg.impact_half_life_ticks)
            k_liq = 0.5 ** (1.0 / cfg.liquidity_recovery_half_life_ticks)
            for s in self.symbols:
                self._impact[s] *= k_imp
                self._drain[s] = 1.0 - (1.0 - self._drain[s]) * k_liq
            self._quotes.update({s: q for s, q in quotes.items() if s in self.symbols})
            for s in self._quotes:
                self._publish(s)
            if self._quotes:
                self.store.set_connected(True)

    def disconnect(self) -> None:
        self.store.set_connected(False)

    def _next_ms(self) -> int:
        self._seq_ms = max(int(time.time() * 1000), self._seq_ms + 1)
        return self._seq_ms

    def liquidity(self, symbol: str) -> float:
        q = self._quotes.get(symbol)
        base = q.liquidity if q else 1.0
        return max(self.config.min_liquidity, base * self._drain[symbol])

    def _build_book(self, symbol: str, quote: BaseQuote, liq: float, shift: float) -> L2BookSnapshot:
        cfg = self.config
        now = time.time()
        if quote.book is not None:  # libro real desplazado por el impacto y escalado por la liquidez
            bids = quote.book.bids.copy()
            asks = quote.book.asks.copy()
            bids[:, 0] *= shift
            asks[:, 0] *= shift
            bids[:, 1] *= liq
            asks[:, 1] *= liq
        else:
            mid = quote.price * shift
            half_spread = cfg.base_spread_bps / liq / 2e4
            i = np.arange(cfg.levels)
            offsets = half_spread + i * cfg.level_spacing_bps / 1e4
            depth_usd = cfg.depth_usd_per_level.get(symbol, 100_000.0) * liq * np.exp(-cfg.depth_decay * i)
            bid_px, ask_px = mid * (1 - offsets), mid * (1 + offsets)
            bids = np.column_stack([bid_px, depth_usd / bid_px, np.full(cfg.levels, 3.0)])
            asks = np.column_stack([ask_px, depth_usd / ask_px, np.full(cfg.levels, 3.0)])
        bids.flags.writeable = False
        asks.flags.writeable = False
        return L2BookSnapshot(symbol, self._next_ms(), now, bids, asks)

    def _publish(self, symbol: str) -> None:
        quote = self._quotes[symbol]
        liq = self.liquidity(symbol)
        shift = math.exp(self._impact[symbol])
        book = self._build_book(symbol, quote, liq, shift)
        mid = book.mid or quote.price * shift
        w = self.config.mark_oracle_weight
        mark = (1 - w) * mid + w * quote.oracle
        premium = (mark - quote.oracle) / quote.oracle
        funding = quote.funding_hourly if quote.funding_hourly is not None else float(np.clip(premium / 8, -0.004, 0.004))
        ctx = AssetContext(
            coin=symbol, mark_price=mark, oracle_price=quote.oracle, mid_price=mid,
            funding_rate_hourly=funding, premium=premium, open_interest=0.0, day_notional_volume=0.0,
            impact_bid=book.best_bid, impact_ask=book.best_ask, received_at=time.time(),
        )
        self.store.update_book(book)
        self.store.update_context(ctx)

    # ------------------------------------------------------------ flujo tóxico
    def execute_flow(self, symbol: str, side: str, notional_usd: float) -> FlowResult:
        """
        Ejecuta flujo agresivo contra el libro sombra (``side`` = "SELL" consume bids).

        Efectos: el mid salta al último nivel consumido (impacto que decae con ``impact_half_life_ticks``)
        y la liquidez cae ``exp(−notional / profundidad_nominal)``.
        El nuevo estado se publica inmediatamente: el motor de riesgo lo ve en su siguiente ciclo.
        """
        with self._lock:
            snap = self.store.get(symbol)
            if snap is None or snap.book is None or symbol not in self._quotes:
                return FlowResult(symbol, side, notional_usd, 0.0, None, 0.0, self.liquidity(symbol))
            book = snap.book
            mid = book.mid or self._quotes[symbol].price
            book_side = "sell" if side == "SELL" else "buy"
            levels = book.bids if book_side == "sell" else book.asks
            vwap, filled = book.simulate_market_order(book_side, notional_usd / mid)
            if vwap is None or filled <= 0:
                return FlowResult(symbol, side, notional_usd, 0.0, None, 0.0, self.liquidity(symbol))
            consumed = np.searchsorted(np.cumsum(levels[:, 1]), filled - 1e-12)
            last_px = float(levels[min(consumed, len(levels) - 1), 0])
            self._impact[symbol] += math.log(last_px / mid)
            # drenaje relativo a la profundidad NOMINAL (no a la ya drenada: evita colapso compuesto)
            self._drain[symbol] *= math.exp(-(filled * vwap) / self._nominal_depth_usd(symbol))
            self._publish(symbol)
            impact_bps = (last_px - mid) / mid * 1e4
            result = FlowResult(symbol, side, notional_usd, filled, vwap, impact_bps, self.liquidity(symbol))
        log.info("shadow.toxic_flow", symbol=symbol, side=side, notional=round(notional_usd), filled=round(filled, 6),
                 vwap=round(vwap, 2), impact_bps=round(impact_bps, 2), liquidity=round(result.liquidity_after, 4))
        return result

    def _nominal_depth_usd(self, symbol: str) -> float:
        cfg = self.config
        per_level = cfg.depth_usd_per_level.get(symbol, 100_000.0)
        return 2 * per_level * float(np.exp(-cfg.depth_decay * np.arange(cfg.levels)).sum())

    def state(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for s in self.symbols:
            snap = self.store.get(s)
            if snap is None or snap.ctx is None:
                continue
            out[s] = {
                "mark": snap.ctx.mark_price, "oracle": snap.ctx.oracle_price,
                "basis_bps": snap.ctx.mark_oracle_basis_bps, "liquidity": self.liquidity(s),
                "spread_bps": snap.book.spread_bps if snap.book else float("nan"),
            }
        return out
