"""
quant_models/var_es.py
======================

Value at Risk y Expected Shortfall (CVaR) para carteras de perpetuos cripto en margen cruzado.

Modelo de P&L
-------------
Bajo margen cruzado todas las posiciones de un usuario comparten colateral, así que el riesgo
relevante es el de la **cartera neta** del usuario. Con cantidades con signo ``q`` (+long/−short,
misma convención que ``open_positions``) y precios ``P``, la exposición nocional es ``e = q·P`` y
el P&L de un escenario de log-retornos ``r`` es exacto (no aproximación delta):

    PnL = Σ_i e_i · (exp(r_i) − 1)

Todo es matricial: con ``R`` (S escenarios × N activos) y ``E`` (U cuentas × N activos)

    PnL = expm1(R) @ Eᵀ        -> (S × U)  — VaR/ES de TODAS las cuentas en una sola operación BLAS.

Métodos
-------
* ``historical_var_es``     Simulación histórica (con opción de horizontes solapados h-periodos).
* ``filtered_historical_scenarios``  FHS: residuos estandarizados GARCH reescalados a la vol actual.
* ``parametric_var_es``     Varianza-covarianza (delta-normal) con Normal o Student-t.
* ``cornish_fisher_var``    VaR modificado por asimetría y curtosis (colas cripto).
* ``component_var``         Descomposición de Euler: contribución de cada activo al VaR.
* ``kupiec_pof_test``       Backtesting (proportion of failures) para validar el modelo.

Convención de signos: **VaR y ES se reportan como pérdidas positivas** (USD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "VaRMethod",
    "RiskEstimate",
    "exposures",
    "portfolio_pnl",
    "tail_var_es",
    "historical_var_es",
    "filtered_historical_scenarios",
    "parametric_var_es",
    "cornish_fisher_var",
    "component_var",
    "ewma_covariance",
    "kupiec_pof_test",
    "KupiecResult",
]

FloatArray = NDArray[np.float64]


class VaRMethod(str, Enum):
    HISTORICAL = "historical"
    FILTERED_HISTORICAL = "filtered_historical"
    PARAMETRIC_NORMAL = "parametric_normal"
    PARAMETRIC_T = "parametric_t"
    CORNISH_FISHER = "cornish_fisher"
    MONTE_CARLO = "monte_carlo"


@dataclass(frozen=True, slots=True)
class RiskEstimate:
    """
    VaR/ES de una o varias cuentas. ``var`` y ``es`` son escalares (una cuenta) o arrays (U,).
    """

    var: FloatArray | float
    es: FloatArray | float
    alpha: float
    horizon: int
    method: VaRMethod
    n_scenarios: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "var": np.round(self.var, 2).tolist() if isinstance(self.var, np.ndarray) else round(self.var, 2),
            "es": np.round(self.es, 2).tolist() if isinstance(self.es, np.ndarray) else round(self.es, 2),
            "alpha": self.alpha,
            "horizon": self.horizon,
            "method": self.method.value,
            "n_scenarios": self.n_scenarios,
        }


# =============================================================================
# Validación y construcción de exposiciones
# =============================================================================
def _check_alpha(alpha: float) -> None:
    if not 0.5 < alpha < 1:
        raise ValueError(f"alpha debe estar en (0.5, 1): {alpha}")


def _as_2d_exposures(e: ArrayLike) -> tuple[FloatArray, bool]:
    """Normaliza exposiciones a (U, N). Devuelve también si la entrada era una sola cuenta."""
    arr = np.asarray(e, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError("exposures debe ser (N,) o (U, N)")


def _squeeze(x: FloatArray, single: bool) -> FloatArray | float:
    return float(x[0]) if single else x


def exposures(quantities: ArrayLike, prices: ArrayLike) -> FloatArray:
    """Nocional con signo ``q · P``. Admite ``q`` (N,) o (U, N) con ``P`` (N,) por broadcasting."""
    q = np.asarray(quantities, dtype=np.float64)
    p = np.asarray(prices, dtype=np.float64)
    if np.any(p <= 0) or not np.all(np.isfinite(p)):
        raise ValueError("Precios inválidos")
    return q * p


def portfolio_pnl(log_return_scenarios: ArrayLike, exposure: ArrayLike) -> FloatArray:
    """
    P&L exacto por escenario y cuenta: ``expm1(R) @ Eᵀ``.

    Args:
        log_return_scenarios: (S, N) log-retornos.
        exposure: (N,) o (U, N) nocionales con signo.

    Returns:
        (S,) si una cuenta, (S, U) si varias.
    """
    R = np.asarray(log_return_scenarios, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("Los escenarios deben ser una matriz (S, N)")
    E, single = _as_2d_exposures(exposure)
    if E.shape[1] != R.shape[1]:
        raise ValueError(f"N activos inconsistente: escenarios {R.shape[1]} vs exposiciones {E.shape[1]}")
    pnl = np.expm1(R) @ E.T
    return pnl[:, 0] if single else pnl


# =============================================================================
# Núcleo: VaR / ES empíricos vectorizados
# =============================================================================
def tail_var_es(pnl: ArrayLike, alpha: float = 0.99) -> tuple[FloatArray | float, FloatArray | float]:
    """
    VaR y ES empíricos de una distribución de P&L (por columnas).

    Con ``k = ceil(S·(1−α))`` escenarios en la cola:
        VaR = k-ésima mayor pérdida ; ES = media de las k mayores pérdidas  (=> ES ≥ VaR siempre).

    Usa ``np.partition`` (selección O(S), no ordenación O(S log S)) a lo largo del eje de escenarios,
    de forma que miles de cuentas se evalúan en una sola llamada.
    """
    _check_alpha(alpha)
    losses = -np.asarray(pnl, dtype=np.float64)
    single = losses.ndim == 1
    L = losses[:, None] if single else losses
    S = L.shape[0]
    k = max(1, math.ceil(S * (1 - alpha) - 1e-9))
    if k > S:
        raise ValueError("Escenarios insuficientes para el nivel de confianza")
    worst = -np.partition(-L, k - 1, axis=0)[:k]   # las k mayores pérdidas (la k-ésima en la fila k-1)
    var = worst[k - 1]
    es = worst.mean(axis=0)
    return _squeeze(var, single), _squeeze(es, single)


def _aggregate_horizon(R: FloatArray, horizon: int) -> FloatArray:
    """Log-retornos h-periodo solapados vía suma acumulada (O(T)): r_{t:t+h} = C_{t+h} − C_t."""
    if horizon == 1:
        return R
    if horizon >= len(R):
        raise ValueError("horizon mayor que la muestra")
    C = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R, axis=0)])
    return C[horizon:] - C[:-horizon]


def historical_var_es(
    log_returns: ArrayLike,
    exposure: ArrayLike,
    *,
    alpha: float = 0.99,
    horizon: int = 1,
    scaling: str = "overlapping",
) -> RiskEstimate:
    """
    VaR/ES por simulación histórica (revaluación completa sobre escenarios históricos).

    Args:
        log_returns: (T, N) log-retornos históricos alineados por activo.
        exposure: (N,) o (U, N) nocionales con signo.
        horizon: periodos de tenencia (en liquidación cripto, típicamente 1 día o menos).
        scaling: ``"overlapping"`` (retornos h-periodo solapados, captura autocorrelación/colas)
                 o ``"sqrt"`` (VaR 1-periodo × √h, regla de Basilea; asume i.i.d.).
    """
    R = np.asarray(log_returns, dtype=np.float64)
    if R.ndim == 1:
        R = R[:, None]
    if not np.all(np.isfinite(R)):
        raise ValueError("Escenarios con NaN/inf: limpia la serie antes")
    if scaling == "overlapping":
        scen = _aggregate_horizon(R, horizon)
        var, es = tail_var_es(portfolio_pnl(scen, exposure), alpha)
    elif scaling == "sqrt":
        scen = R
        var, es = tail_var_es(portfolio_pnl(R, exposure), alpha)
        var, es = var * math.sqrt(horizon), es * math.sqrt(horizon)
    else:
        raise ValueError("scaling debe ser 'overlapping' o 'sqrt'")
    return RiskEstimate(var, es, alpha, horizon, VaRMethod.HISTORICAL, n_scenarios=len(scen))


def filtered_historical_scenarios(std_residuals: ArrayLike, current_vol: ArrayLike) -> FloatArray:
    """
    Filtered Historical Simulation (Barone-Adesi): ``r* = z_t · σ_{T+1}``.

    Combina la forma empírica de las colas (residuos estandarizados z del GARCH, (T, N)) con el
    régimen de volatilidad ACTUAL (σ por activo, (N,)). Corrige el principal defecto de la
    simulación histórica pura: ignorar que hoy la vol puede ser 3× la media muestral.
    Conserva la dependencia cruzada si los z están alineados en el tiempo.
    """
    Z = np.asarray(std_residuals, dtype=np.float64)
    Z = Z[:, None] if Z.ndim == 1 else Z
    sigma = np.asarray(current_vol, dtype=np.float64).reshape(1, -1)
    if sigma.shape[1] != Z.shape[1] or np.any(sigma <= 0):
        raise ValueError("current_vol debe ser positiva y alinearse con las columnas de residuos")
    return Z * sigma


# =============================================================================
# Paramétrico
# =============================================================================
def ewma_covariance(log_returns: ArrayLike, lam: float = 0.94) -> FloatArray:
    """
    Covarianza EWMA (RiskMetrics) al último instante, vectorizada: Σ = Σ_t w_t r_t r_tᵀ con
    pesos geométricos normalizados (sin bucle sobre t).
    """
    R = np.asarray(log_returns, dtype=np.float64)
    T = len(R)
    w = (1 - lam) * lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    return (R * w[:, None]).T @ R


def _portfolio_sigma(E: FloatArray, cov: FloatArray) -> FloatArray:
    """σ_p = sqrt(e Σ eᵀ) para cada fila de E, en una sola contracción einsum."""
    var = np.einsum("un,nm,um->u", E, cov, E, optimize=True)
    return np.sqrt(np.clip(var, 0.0, None))


def parametric_var_es(
    exposure: ArrayLike,
    cov: ArrayLike,
    *,
    alpha: float = 0.99,
    horizon: int = 1,
    mean: ArrayLike | None = None,
    dist: str = "normal",
    nu: float = 4.0,
) -> RiskEstimate:
    """
    VaR/ES varianza-covarianza (delta-normal) para U cuentas a la vez.

    Normal:     VaR = z_α σ√h − μh ;  ES = σ√h · φ(z_α)/(1−α) − μh
    Student-t (σ = desviación típica, se reescala por √((ν−2)/ν)):
                VaR = s·t_α ;  ES = s · f_ν(t_α)/(1−α) · (ν + t_α²)/(ν − 1)

    Args:
        cov: (N, N) covarianza de log-retornos por periodo (muestral, EWMA o GARCH-DCC).
        mean: (N,) retorno medio por periodo (por defecto 0: estándar en horizontes cortos).
        dist: ``"normal"`` o ``"t"``.
    """
    _check_alpha(alpha)
    E, single = _as_2d_exposures(exposure)
    C = np.asarray(cov, dtype=np.float64)
    if C.shape != (E.shape[1], E.shape[1]):
        raise ValueError("cov debe ser (N, N) consistente con las exposiciones")
    if not np.allclose(C, C.T, atol=1e-12):
        raise ValueError("cov no es simétrica")

    sigma = _portfolio_sigma(E, C) * math.sqrt(horizon)
    mu = (E @ np.asarray(mean, dtype=np.float64)) * horizon if mean is not None else np.zeros(len(E))

    if dist == "normal":
        z = stats.norm.ppf(alpha)
        var = z * sigma - mu
        es = sigma * stats.norm.pdf(z) / (1 - alpha) - mu
        method = VaRMethod.PARAMETRIC_NORMAL
    elif dist == "t":
        if nu <= 2:
            raise ValueError("nu debe ser > 2 para varianza finita")
        t_a = stats.t.ppf(alpha, nu)
        s = sigma * math.sqrt((nu - 2) / nu)
        var = s * t_a - mu
        es = s * stats.t.pdf(t_a, nu) / (1 - alpha) * (nu + t_a**2) / (nu - 1) - mu
        method = VaRMethod.PARAMETRIC_T
    else:
        raise ValueError("dist debe ser 'normal' o 't'")
    return RiskEstimate(_squeeze(var, single), _squeeze(es, single), alpha, horizon, method)


def cornish_fisher_var(pnl: ArrayLike, alpha: float = 0.99) -> FloatArray | float:
    """
    VaR modificado (Cornish-Fisher) a partir de los momentos de la distribución de P&L:

        z_cf = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36   (K = curtosis en exceso)

    Recoge asimetría negativa y colas gruesas sin asumir una distribución concreta.
    """
    _check_alpha(alpha)
    X = np.asarray(pnl, dtype=np.float64)
    single = X.ndim == 1
    X = X[:, None] if single else X
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    S = stats.skew(X, axis=0, bias=False)
    K = stats.kurtosis(X, axis=0, fisher=True, bias=False)
    z = stats.norm.ppf(1 - alpha)  # cuantil de la cola izquierda del P&L
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24 - (2 * z**3 - 5 * z) * S**2 / 36
    return _squeeze(-(mu + z_cf * sd), single)


def component_var(exposure: ArrayLike, cov: ArrayLike, *, alpha: float = 0.99, horizon: int = 1) -> FloatArray:
    """
    Descomposición de Euler del VaR paramétrico normal: ``CVaR_i = e_i (Σe)_i / σ_p · z_α √h``.

    Las contribuciones suman exactamente el VaR total; contribuciones negativas identifican
    posiciones que actúan como cobertura (p.ej. el corto ETH del arbitrajista).
    Admite (N,) -> (N,) o (U, N) -> (U, N).
    """
    E, single = _as_2d_exposures(exposure)
    C = np.asarray(cov, dtype=np.float64)
    sigma = _portfolio_sigma(E, C)
    with np.errstate(invalid="ignore", divide="ignore"):
        contrib = E * (E @ C) / sigma[:, None] * stats.norm.ppf(alpha) * math.sqrt(horizon)
    contrib = np.nan_to_num(contrib)
    return contrib[0] if single else contrib


# =============================================================================
# Backtesting
# =============================================================================
@dataclass(frozen=True, slots=True)
class KupiecResult:
    exceptions: int
    observations: int
    expected_rate: float
    observed_rate: float
    lr_statistic: float
    p_value: float

    @property
    def reject(self) -> bool:
        """Rechaza H0 (modelo bien calibrado) al 5 %."""
        return self.p_value < 0.05


def kupiec_pof_test(realized_pnl: ArrayLike, var_forecasts: ArrayLike, alpha: float = 0.99) -> KupiecResult:
    """
    Test de Kupiec (POF): ¿la frecuencia de excepciones (pérdida > VaR previsto) es coherente con 1−α?

        LR = −2 ln[(1−p)^{T−x} p^x] + 2 ln[(1−x/T)^{T−x} (x/T)^x]  ~  χ²(1)
    """
    _check_alpha(alpha)
    pnl = np.asarray(realized_pnl, dtype=np.float64)
    var = np.broadcast_to(np.asarray(var_forecasts, dtype=np.float64), pnl.shape)
    T = pnl.size
    x = int(np.sum(-pnl > var))
    p = 1 - alpha
    phat = x / T

    def _ll(prob: float) -> float:
        # x·ln(p) + (T−x)·ln(1−p) con convención 0·ln0 = 0
        a = x * math.log(prob) if x > 0 else 0.0
        b = (T - x) * math.log(1 - prob) if T - x > 0 else 0.0
        return a + b

    lr = -2 * (_ll(p) - (_ll(phat) if 0 < phat < 1 else 0.0))
    lr = max(lr, 0.0)
    return KupiecResult(x, T, p, phat, lr, float(stats.chi2.sf(lr, df=1)))
