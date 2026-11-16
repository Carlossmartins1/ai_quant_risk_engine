"""
quant_models/monte_carlo.py
===========================

Simulador Monte Carlo multi-activo de **difusión con saltos** (Merton) con dinámica de crisis,
diseñado para el **Modo Estrés / Cisne Negro** del Risk Engine.

Dinámica por activo i (log-precio, medida física, tiempo en años)
-----------------------------------------------------------------
::

    d ln S_i = (μ_i − ½ σ_i² v_t²) dt  +  σ_i v_t dW_i          difusión correlacionada (Cholesky)
             + β_i · Σ_k Y_k dN_t                                saltos SISTÉMICOS (contagio BTC↔ETH)
             + Σ_k Y_ik dN_it                                    saltos idiosincráticos
             + ln(1 − c_i)/M · 1[t_crash ≤ t < t_crash + M]      shock forzado repartido en M pasos (cisne negro)

    Y_k ~ N(μ_J, σ_J²)       tamaño de salto (log)
    λ_t = λ_0 + H_t,  H_t = H_{t−1} e^{−κ dt} + η · eventos_{t−1}    intensidad Hawkes (autoexcitada)
    v_t = 1 + X_t,    X_t = X_{t−1} e^{−κ_v dt} + s_v · eventos_t    multiplicador de vol post-shock
    L_t = exp(−D_t),  D_t = D_{t−1} e^{−κ_L dt} + s_L · eventos_t    factor de liquidez del libro ∈ (0, 1]

Interpretación de mercado
-------------------------
* **Cascada de liquidaciones**: cada salto (o el shock forzado) eleva la probabilidad de nuevos saltos
  (proceso de Hawkes con ratio de ramificación η/κ < 1): las liquidaciones generan ventas que
  generan más liquidaciones.
* **Colapso de liquidez**: ``L_t`` cae con cada evento y se recupera exponencialmente; el coste de
  ejecución de una liquidación escala como ``spread_base / L_t`` (los market makers retiran órdenes).
* **Ruptura de correlaciones**: en crisis las correlaciones tienden a 1 (preset ``black_swan``: 0.95).

Rendimiento
-----------
La única recursión con realimentación real (Hawkes: los saltos dependen de saltos pasados) se itera
sobre el tiempo vectorizando sobre trayectorias. Todo lo demás (volatilidad, liquidez, difusión,
precios) se resuelve en bloque: ``scipy.signal.lfilter`` para recursiones lineales, ``cumsum`` para
integrar y BLAS para correlacionar. 10 000 trayectorias × 288 pasos × 2 activos ≈ 1 s.

Ejemplo
-------
>>> cfg = ScenarioConfig.black_swan(n_paths=10_000, seed=42)
>>> res = JumpDiffusionSimulator(cfg).run()
>>> whale = MarginAccount("whale", collateral=5_000, quantities={"BTC": 1.0}, entry_prices={"BTC": 95_000})
>>> res.margin_stress(whale).as_dict()   # prob. liquidación, tiempo a liquidación, bad debt esperado...
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import signal

from quant_models.var_es import RiskEstimate, VaRMethod, tail_var_es

__all__ = [
    "AssetParams",
    "JumpParams",
    "CrisisDynamics",
    "ScenarioConfig",
    "JumpDiffusionSimulator",
    "SimulationResult",
    "MarginAccount",
    "MarginStressReport",
    "per_day",
]

FloatArray = NDArray[np.float64]
DAYS_PER_YEAR = 365.0  # mercado 24/7


def per_day(rate: float) -> float:
    """Convierte una tasa por día a tasa anual (unidad interna del simulador)."""
    return rate * DAYS_PER_YEAR


# =============================================================================
# Configuración
# =============================================================================
@dataclass(frozen=True, slots=True)
class AssetParams:
    symbol: str
    s0: float                         # precio inicial (mark)
    sigma: float                      # vol anual de difusión
    mu: float = 0.0                   # drift anual (medida física)
    jump_beta: float = 1.0            # sensibilidad al salto sistémico (ETH > 1: cae más que BTC)
    idio_jump_intensity: float = 0.0  # saltos propios por año
    idio_jump_mean: float = 0.0
    idio_jump_std: float = 0.0
    crash_pct: float = 0.0            # caída del shock forzado (0.18 = −18 %)

    def __post_init__(self) -> None:
        if self.s0 <= 0 or self.sigma < 0:
            raise ValueError(f"{self.symbol}: s0 > 0 y sigma >= 0")
        if not 0 <= self.crash_pct < 1:
            raise ValueError(f"{self.symbol}: crash_pct en [0, 1)")


@dataclass(frozen=True, slots=True)
class JumpParams:
    """Saltos sistémicos (afectan a todos los activos vía ``jump_beta``). Tasas anuales."""

    intensity: float = 0.0            # λ_0
    mean: float = 0.0                 # μ_J (log)
    std: float = 0.0                  # σ_J
    hawkes_excitation: float = 0.0    # η: incremento de intensidad por evento
    hawkes_decay: float = 0.0         # κ: velocidad de disipación de la excitación

    @property
    def branching_ratio(self) -> float:
        """η/κ: nº medio de saltos "hijos" por salto. Debe ser < 1 (estacionariedad)."""
        return self.hawkes_excitation / self.hawkes_decay if self.hawkes_decay > 0 else 0.0

    @property
    def is_hawkes(self) -> bool:
        return self.hawkes_excitation > 0 and self.hawkes_decay > 0

    def __post_init__(self) -> None:
        if self.intensity < 0 or self.std < 0:
            raise ValueError("intensity y std deben ser >= 0")
        if self.hawkes_excitation > 0 and self.hawkes_decay <= 0:
            raise ValueError("hawkes_excitation > 0 requiere hawkes_decay > 0")
        if self.branching_ratio >= 1:
            raise ValueError(f"Hawkes explosivo: ratio de ramificación {self.branching_ratio:.2f} >= 1")


@dataclass(frozen=True, slots=True)
class CrisisDynamics:
    """Efectos de segundo orden de cada evento de salto. Tasas anuales."""

    vol_shock_per_event: float = 0.0       # +X a v_t por evento (0.5 => vol ×1.5 tras un salto)
    vol_decay: float = per_day(6.0)        # κ_v
    liquidity_drain_per_event: float = 0.0  # s_L: L cae un factor e^{−s_L} por evento
    liquidity_recovery: float = per_day(6.0)  # κ_L
    base_spread_bps: float = 2.0           # coste de ejecución (½ spread + impacto) con liquidez plena
    max_vol_multiplier: float = 4.0        # techo de v_t (evita varianzas explosivas en clusters extremos)

    def __post_init__(self) -> None:
        if self.max_vol_multiplier < 1:
            raise ValueError("max_vol_multiplier debe ser >= 1")


@dataclass(frozen=True)
class ScenarioConfig:
    assets: tuple[AssetParams, ...]
    correlation: FloatArray
    jumps: JumpParams = field(default_factory=JumpParams)
    dynamics: CrisisDynamics = field(default_factory=CrisisDynamics)
    horizon_days: float = 1.0
    n_steps: int = 288                     # 5 minutos sobre 24 h
    n_paths: int = 10_000
    crash_time_days: float | None = None   # instante del shock forzado (None = sin shock)
    crash_steps: int = 1                   # pasos en los que se reparte el shock (1 = gap instantáneo)
    compensate_jumps: bool = False         # True => E[S_T] neutral al salto (martingala); False => sesgo bajista
    antithetic: bool = True                # variables antitéticas en la difusión (reduce varianza)
    seed: int | None = None
    name: str = "custom"

    def __post_init__(self) -> None:
        corr = np.asarray(self.correlation, dtype=np.float64)
        n = len(self.assets)
        if corr.shape != (n, n) or not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation debe ser simétrica (N×N) con diagonal 1")
        if len({a.symbol for a in self.assets}) != n:
            raise ValueError("Símbolos duplicados")
        if self.n_steps < 1 or self.n_paths < 2 or self.horizon_days <= 0:
            raise ValueError("n_steps >= 1, n_paths >= 2, horizon_days > 0")
        if self.crash_time_days is not None and not 0 <= self.crash_time_days < self.horizon_days:
            raise ValueError("crash_time_days debe estar dentro del horizonte")
        if self.crash_steps < 1:
            raise ValueError("crash_steps >= 1")
        object.__setattr__(self, "correlation", corr)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(a.symbol for a in self.assets)

    @property
    def dt(self) -> float:
        return self.horizon_days / DAYS_PER_YEAR / self.n_steps

    # ----------------------------------------------------------- presets
    @classmethod
    def normal(cls, *, btc_s0: float = 95_000.0, eth_s0: float = 3_500.0, **overrides: Any) -> "ScenarioConfig":
        """Régimen normal: vol histórica típica, saltos raros y pequeños, sin cascadas."""
        base = cls(
            name="normal",
            assets=(
                AssetParams("BTC", btc_s0, sigma=0.55),
                AssetParams("ETH", eth_s0, sigma=0.70, jump_beta=1.2),
            ),
            correlation=np.array([[1.0, 0.80], [0.80, 1.0]]),
            jumps=JumpParams(intensity=12.0, mean=-0.005, std=0.02),
            dynamics=CrisisDynamics(
                vol_shock_per_event=0.2, vol_decay=per_day(6), liquidity_drain_per_event=0.1,
                liquidity_recovery=per_day(12), base_spread_bps=2.0,
            ),
        )
        return replace(base, **overrides)

    @classmethod
    def black_swan(cls, *, btc_s0: float = 95_000.0, eth_s0: float = 3_500.0, **overrides: Any) -> "ScenarioConfig":
        """
        Cisne negro (parámetros ilustrativos, del orden de magnitud de episodios como marzo 2020
        o mayo 2021): shock forzado a la hora 1 (BTC −18 %, ETH −24 %), cascada Hawkes de saltos
        bajistas (ramificación 0.75, memoria ~1 h), vol de difusión ×1.6 (hasta ×3 tras clusters de
        saltos), correlación 0.95 y drenaje severo de liquidez. ``crash_steps`` > 1 reparte el shock
        en caída geométrica uniforme (p.ej. 18 pasos => BTC ≈ −1.1 % por paso).
        """
        base = cls(
            name="black_swan",
            assets=(
                AssetParams("BTC", btc_s0, sigma=0.90, crash_pct=0.18),
                AssetParams("ETH", eth_s0, sigma=1.15, jump_beta=1.25, crash_pct=0.24),
            ),
            correlation=np.array([[1.0, 0.95], [0.95, 1.0]]),
            jumps=JumpParams(
                intensity=per_day(2.0), mean=-0.02, std=0.02,
                hawkes_excitation=per_day(18.0), hawkes_decay=per_day(24.0),
            ),
            dynamics=CrisisDynamics(
                vol_shock_per_event=0.4, vol_decay=per_day(6), liquidity_drain_per_event=0.5,
                liquidity_recovery=per_day(3), base_spread_bps=3.0, max_vol_multiplier=3.0,
            ),
            crash_time_days=1 / 24,
        )
        return replace(base, **overrides)

    def with_volatility(self, annual_vols: Mapping[str, float], *, multiplier: float = 1.0) -> "ScenarioConfig":
        """
        Recalibra σ con volatilidades condicionales (p.ej. ``VolatilityFit.current_vol_annualized``).
        ``multiplier`` > 1 aplica un estrés adicional sobre la vol GARCH actual.
        """
        assets = tuple(
            replace(a, sigma=float(annual_vols[a.symbol]) * multiplier) if a.symbol in annual_vols else a
            for a in self.assets
        )
        return replace(self, assets=assets)

    def with_prices(self, prices: Mapping[str, float]) -> "ScenarioConfig":
        """Arranca la simulación desde los mark prices actuales (p.ej. ``MarketDataStore``)."""
        return replace(self, assets=tuple(replace(a, s0=float(prices.get(a.symbol, a.s0))) for a in self.assets))


# =============================================================================
# Cuentas de margen para el test de estrés
# =============================================================================
@dataclass(frozen=True)
class MarginAccount:
    """Cuenta cross-margin (misma semántica que ``users_collateral`` + ``open_positions``)."""

    name: str
    collateral: float
    quantities: Mapping[str, float]                 # con signo: +long / −short
    entry_prices: Mapping[str, float]
    maintenance_margin_rate: Mapping[str, float] | float = 0.025
    liquidation_fee_rate: float = 0.005

    def vectors(self, symbols: Sequence[str]) -> tuple[FloatArray, FloatArray, FloatArray]:
        q = np.array([self.quantities.get(s, 0.0) for s in symbols], dtype=np.float64)
        entry = np.array([self.entry_prices.get(s, 0.0) for s in symbols], dtype=np.float64)
        if np.any((q != 0) & (entry <= 0)):
            raise ValueError(f"{self.name}: falta entry_price para alguna posición")
        mmr_src = self.maintenance_margin_rate
        mmr = np.array(
            [mmr_src if isinstance(mmr_src, (int, float)) else mmr_src.get(s, 0.025) for s in symbols],
            dtype=np.float64,
        )
        return q, entry, mmr


@dataclass(frozen=True)
class MarginStressReport:
    account: str
    scenario: str
    n_paths: int
    initial_equity: float
    initial_margin_ratio: float           # equity / MM (≤ 1 => ya liquidable)
    liquidation_probability: float
    median_time_to_liquidation_min: float | None
    p05_time_to_liquidation_min: float | None
    bad_debt_probability: float           # P(equity tras liquidar < 0) -> golpea al Insurance Fund
    expected_deficit: float               # E[déficit] sobre todas las trayectorias
    deficit_es_99: float                  # media del 1 % peor de déficits
    expected_fee_to_fund: float
    worst_equity_p01: float               # percentil 1 del mínimo de equity intra-horizonte

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


# =============================================================================
# Resultado
# =============================================================================
@dataclass(frozen=True)
class SimulationResult:
    config: ScenarioConfig
    times_days: FloatArray                          # (T+1,)
    prices: FloatArray = field(repr=False)          # (P, T+1, N)
    liquidity: FloatArray = field(repr=False)       # (P, T+1)  factor ∈ (0, 1]
    vol_multiplier: FloatArray = field(repr=False)  # (P, T+1)
    jump_events: NDArray[np.int32] = field(repr=False)  # (P, T) eventos sistémicos (incluye shock forzado)
    elapsed_s: float = 0.0

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.config.symbols

    @property
    def n_paths(self) -> int:
        return self.prices.shape[0]

    def _idx(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def paths(self, symbol: str) -> FloatArray:
        """Trayectorias de precio (P, T+1) de un activo."""
        return self.prices[:, :, self._idx(symbol)]

    def terminal_log_returns(self) -> FloatArray:
        """(P, N) log-retornos a horizonte: escenarios directos para ``var_es``."""
        return np.log(self.prices[:, -1, :] / self.prices[:, 0, :])

    def max_drawdown(self) -> FloatArray:
        """(P, N) máximo drawdown intra-horizonte por trayectoria y activo."""
        running_max = np.maximum.accumulate(self.prices, axis=1)
        return 1.0 - np.min(self.prices / running_max, axis=1)

    def pnl_paths(self, quantities: ArrayLike) -> FloatArray:
        """P&L mark-to-market ``Σ q_i (S_t − S_0)``: (P, T+1) o (U, P, T+1) si ``quantities`` es (U, N)."""
        q = np.asarray(quantities, dtype=np.float64)
        dS = self.prices - self.prices[:, :1, :]
        return np.einsum("ptn,n->pt", dS, q) if q.ndim == 1 else np.einsum("ptn,un->upt", dS, q)

    def var_es(self, quantities: ArrayLike, *, alpha: float = 0.99) -> RiskEstimate:
        """VaR/ES Monte Carlo del P&L a horizonte (revaluación completa)."""
        pnl = self.pnl_paths(quantities)[..., -1]
        pnl = pnl if pnl.ndim == 1 else pnl.T  # (P,) o (P, U)
        var, es = tail_var_es(pnl, alpha)
        horizon_steps = self.config.n_steps
        return RiskEstimate(var, es, alpha, horizon_steps, VaRMethod.MONTE_CARLO, n_scenarios=self.n_paths)

    def asset_summary(self, drawdown_threshold: float = 0.20) -> pd.DataFrame:
        """Estadísticos por activo: distribución del retorno terminal, drawdowns y saltos."""
        r = np.expm1(self.terminal_log_returns())
        dd = self.max_drawdown()
        rows = []
        for i, s in enumerate(self.symbols):
            rows.append({
                "asset": s,
                "mean_return": r[:, i].mean(),
                "p01_return": np.quantile(r[:, i], 0.01),
                "p50_return": np.median(r[:, i]),
                "mean_max_drawdown": dd[:, i].mean(),
                f"P(drawdown>{drawdown_threshold:.0%})": (dd[:, i] > drawdown_threshold).mean(),
            })
        df = pd.DataFrame(rows).set_index("asset")
        df.attrs.update(
            mean_jump_events=float(self.jump_events.sum(axis=1).mean()),
            min_liquidity_p50=float(np.median(self.liquidity.min(axis=1))),
            peak_vol_multiplier_p50=float(np.median(self.vol_multiplier.max(axis=1))),
        )
        return df

    # ------------------------------------------------------- estrés de margen
    def margin_stress(self, account: MarginAccount) -> MarginStressReport:
        """
        Recorre cada trayectoria aplicando la regla del motor (Fase 1):

            Equity_t = Collateral + Σ q_i (S_it − Entry_i)
            MM_t     = Σ |q_i| S_it · MMR_i
            liquidación en el primer t con Equity_t ≤ MM_t

        En ese instante la cuenta se cierra pagando un coste de ejecución ``spread_base / L_t``
        (liquidez degradada) y una comisión de liquidación. Si la equity resultante es negativa
        (salto que atraviesa el precio de bancarrota, *gap risk*), el déficit lo absorbe el
        Insurance Fund: ``bad_debt_probability`` y ``deficit_es_99`` cuantifican esa exposición.
        """
        q, entry, mmr = account.vectors(self.symbols)
        S = self.prices
        upnl = np.einsum("ptn,n->pt", S, q) - float(entry @ q)          # (P, T+1)
        equity = account.collateral + upnl
        mm = np.einsum("ptn,n->pt", S, np.abs(q) * mmr)
        breach = equity <= mm
        liquidated = breach.any(axis=1)
        first = np.argmax(breach, axis=1)
        P = self.n_paths
        rows = np.arange(P)

        eq_at = equity[rows, first]
        notional = np.einsum("pn,n->p", S[rows, first, :], np.abs(q))
        slip = notional * self.config.dynamics.base_spread_bps * 1e-4 / self.liquidity[rows, first]
        eq_after = eq_at - slip
        deficit = np.where(liquidated, np.maximum(-eq_after, 0.0), 0.0)
        fee = np.where(liquidated, np.minimum(notional * account.liquidation_fee_rate, np.maximum(eq_after, 0.0)), 0.0)

        minutes = self.times_days[first] * 24 * 60
        t_liq = minutes[liquidated]
        k = max(1, math.ceil(0.01 * P))
        worst_deficits = np.partition(deficit, P - k)[P - k:]
        mm0 = float(mm[0, 0])
        return MarginStressReport(
            account=account.name,
            scenario=self.config.name,
            n_paths=P,
            initial_equity=float(equity[0, 0]),
            initial_margin_ratio=float(equity[0, 0] / mm0) if mm0 > 0 else math.inf,
            liquidation_probability=float(liquidated.mean()),
            median_time_to_liquidation_min=float(np.median(t_liq)) if t_liq.size else None,
            p05_time_to_liquidation_min=float(np.quantile(t_liq, 0.05)) if t_liq.size else None,
            bad_debt_probability=float((deficit > 0).mean()),
            expected_deficit=float(deficit.mean()),
            deficit_es_99=float(worst_deficits.mean()),
            expected_fee_to_fund=float(fee.mean()),
            worst_equity_p01=float(np.quantile(equity.min(axis=1), 0.01)),
        )

    def margin_stress_table(self, accounts: Sequence[MarginAccount]) -> pd.DataFrame:
        return pd.DataFrame([self.margin_stress(a).as_dict() for a in accounts]).set_index("account")


# =============================================================================
# Simulador
# =============================================================================
def _cholesky_psd(corr: FloatArray) -> FloatArray:
    """Cholesky robusto: si la matriz no es definida positiva, proyecta a la PSD más cercana."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(corr)
        fixed = V @ np.diag(np.clip(w, 1e-10, None)) @ V.T
        d = np.sqrt(np.diag(fixed))
        warnings.warn("Correlación no definida positiva: proyectada a la PSD más cercana", RuntimeWarning, stacklevel=3)
        return np.linalg.cholesky(fixed / np.outer(d, d))


def _decay_filter(impulses: FloatArray, decay_rate: float, dt: float) -> FloatArray:
    """x_t = x_{t−1}·e^{−κ dt} + impulse_t a lo largo del eje temporal (axis=1), en C."""
    return signal.lfilter([1.0], [1.0, -math.exp(-decay_rate * dt)], impulses, axis=1)


class JumpDiffusionSimulator:
    """Generador de escenarios. Sin estado entre ejecuciones salvo el RNG (reproducible con ``seed``)."""

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self._chol = _cholesky_psd(config.correlation)
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------- componentes
    def _systemic_events(self, P: int, T: int, crash_step: int | None) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Conteos de saltos sistémicos (P, T) y eventos totales (incluye el shock forzado)."""
        jp, dt, rng = self.config.jumps, self.config.dt, self._rng
        forced = np.zeros(T, dtype=np.int32)
        if crash_step is not None:
            forced[crash_step] = 1
        if not jp.is_hawkes:
            counts = rng.poisson(jp.intensity * dt, size=(P, T)).astype(np.int32)
            return counts, counts + forced[None, :]

        counts = np.empty((P, T), dtype=np.int32)
        excitation = np.zeros(P)
        decay = math.exp(-jp.hawkes_decay * dt)
        for t in range(T):  # realimentación genuina: vectorizado sobre trayectorias
            counts[:, t] = rng.poisson((jp.intensity + excitation) * dt)
            excitation = excitation * decay + jp.hawkes_excitation * (counts[:, t] + forced[t])
        return counts, counts + forced[None, :]

    @staticmethod
    def _compound_normal(rng: np.random.Generator, counts: NDArray[np.int32], mean: float, std: float) -> FloatArray:
        """Suma de n saltos N(μ, σ²) en un paso = N(nμ, nσ²): una sola normal por celda."""
        sizes = mean * counts
        if std > 0:
            nz = counts > 0
            sizes = sizes.astype(np.float64)
            sizes[nz] += std * np.sqrt(counts[nz]) * rng.standard_normal(int(nz.sum()))
        return sizes

    def _diffusion_shocks(self, P: int, T: int, N: int) -> FloatArray:
        """Normales correlacionadas (P, T, N), con variables antitéticas si procede."""
        rng = self._rng
        if self.config.antithetic:
            half = rng.standard_normal(((P + 1) // 2, T, N))
            Z = np.concatenate([half, -half], axis=0)[:P]
        else:
            Z = rng.standard_normal((P, T, N))
        return Z @ self._chol.T

    # -------------------------------------------------------------------- run
    def run(self) -> SimulationResult:
        cfg = self.config
        started = time.perf_counter()
        P, T, N, dt = cfg.n_paths, cfg.n_steps, len(cfg.assets), cfg.dt
        rng = self._rng
        crash_step = (
            min(T - 1, int(round(cfg.crash_time_days / cfg.horizon_days * T)))
            if cfg.crash_time_days is not None and any(a.crash_pct > 0 for a in cfg.assets)
            else None
        )

        # 1) Saltos sistémicos (Hawkes) y su tamaño común
        counts, events = self._systemic_events(P, T, crash_step)
        common_jump = self._compound_normal(rng, counts, cfg.jumps.mean, cfg.jumps.std)  # (P, T)

        # 2) Dinámica de crisis: multiplicador de vol (predecible: afecta desde t+1) y liquidez
        dyn = cfg.dynamics
        ev = events.astype(np.float64)
        vol_excess = _decay_filter(dyn.vol_shock_per_event * ev, dyn.vol_decay, dt)
        vol_mult = np.ones((P, T + 1))
        vol_mult[:, 1:] += vol_excess
        np.minimum(vol_mult, dyn.max_vol_multiplier, out=vol_mult)
        drain = _decay_filter(dyn.liquidity_drain_per_event * ev, dyn.liquidity_recovery, dt)
        liquidity = np.ones((P, T + 1))
        liquidity[:, 1:] = np.exp(-drain)

        # 3) Difusión correlacionada con vol estocástica por régimen
        sigma = np.array([a.sigma for a in cfg.assets])
        mu = np.array([a.mu for a in cfg.assets])
        beta = np.array([a.jump_beta for a in cfg.assets])
        v = vol_mult[:, :-1, None]                                             # σ efectivo en [t, t+1)
        Z = self._diffusion_shocks(P, T, N)
        increments = (mu - 0.5 * (sigma * v) ** 2) * dt + sigma * v * math.sqrt(dt) * Z

        # 4) Saltos: sistémicos (β_i · Y) + idiosincráticos + shock forzado
        increments += common_jump[:, :, None] * beta
        for i, a in enumerate(cfg.assets):
            if a.idio_jump_intensity > 0:
                idio = rng.poisson(a.idio_jump_intensity * dt, size=(P, T)).astype(np.int32)
                increments[:, :, i] += self._compound_normal(rng, idio, a.idio_jump_mean, a.idio_jump_std)
            if crash_step is not None and a.crash_pct > 0:
                end = min(T, crash_step + cfg.crash_steps)  # truncado al horizonte: la caída total se conserva
                increments[:, crash_step:end, i] += math.log1p(-a.crash_pct) / (end - crash_step)
        if cfg.compensate_jumps and cfg.jumps.intensity > 0:
            # E[e^{βY}] − 1 por salto; compensa la intensidad base (aprox. de primer orden con Hawkes)
            k = np.exp(beta * cfg.jumps.mean + 0.5 * (beta * cfg.jumps.std) ** 2) - 1
            mean_intensity = cfg.jumps.intensity / (1 - cfg.jumps.branching_ratio)
            increments -= mean_intensity * k * dt

        # 5) Integración: log S_t = log S_0 + Σ incrementos
        log_s0 = np.log([a.s0 for a in cfg.assets])
        log_paths = np.empty((P, T + 1, N))
        log_paths[:, 0, :] = log_s0
        np.cumsum(increments, axis=1, out=log_paths[:, 1:, :])
        log_paths[:, 1:, :] += log_s0
        prices = np.exp(log_paths, out=log_paths)

        return SimulationResult(
            config=cfg,
            times_days=np.linspace(0.0, cfg.horizon_days, T + 1),
            prices=prices,
            liquidity=liquidity,
            vol_multiplier=vol_mult,
            jump_events=events,
            elapsed_s=time.perf_counter() - started,
        )


# =============================================================================
# Demo:  python -m quant_models.monte_carlo
# =============================================================================
if __name__ == "__main__":  # pragma: no cover
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    agents = [
        MarginAccount("whale_20x", collateral=4_750, quantities={"BTC": 1.0}, entry_prices={"BTC": 95_000}),
        MarginAccount(
            "arbitrageur_3x", collateral=60_000,
            quantities={"BTC": 1.0, "ETH": -27.0}, entry_prices={"BTC": 95_000, "ETH": 3_500},
        ),
    ]
    for cfg in (ScenarioConfig.normal(seed=1), ScenarioConfig.black_swan(seed=1)):
        res = JumpDiffusionSimulator(cfg).run()
        print(f"\n=== {cfg.name}  ({res.n_paths:,} trayectorias en {res.elapsed_s:.2f}s) ===")
        print(res.asset_summary().to_string())
        print(res.margin_stress_table(agents).to_string())
