"""
quant_models/volatility.py
==========================

Modelos de volatilidad condicional para criptoactivos (BTC, ETH).

Contenido
---------
* ``fit_volatility``      GARCH(1,1), GJR-GARCH(1,1,1) y EGARCH(1,1,1) vía ``arch`` (MLE), con
                          innovaciones Normal / Student-t / Skew-t y diagnóstico ``statsmodels``
                          (Ljung-Box sobre z², ARCH-LM sobre z).
* ``select_best_model``   Selección por BIC entre especificaciones candidatas.
* ``fit_garch11_mle``     GARCH(1,1) implementado desde cero con SciPy (validación cruzada del
                          estimador de ``arch`` y fallback sin dependencias).
* ``ewma_volatility``     RiskMetrics (λ = 0.94), baseline de la industria.

Convenciones
------------
* Entrada: precios o log-retornos. Salida: volatilidades **en unidades decimales por periodo**
  (0.03 = 3 % diario si la serie es diaria). ``periods_per_year`` anualiza: los mercados cripto
  operan 24/7 -> 365 para datos diarios, 8760 para horarios.
* ``arch`` optimiza mejor con retornos en porcentaje: se reescala internamente (``scale=100``)
  y todos los resultados se devuelven ya desescalados.

Todas las recursiones de varianza se resuelven con ``scipy.signal.lfilter`` (filtro IIR en C):
sin bucles Python, O(T).

Ejemplo
-------
>>> fit = fit_volatility(btc_prices, model=VolModel.GJR_GARCH, dist=Distribution.STUDENT_T, asset="BTC")
>>> fit.current_vol_annualized, fit.persistence, fit.half_life_periods
>>> fit.horizon_vol(10)            # volatilidad agregada a 10 periodos para VaR multi-step
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from arch import arch_model
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, signal
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

__all__ = [
    "VolModel",
    "Distribution",
    "VolatilityFit",
    "Garch11Params",
    "log_returns",
    "fit_volatility",
    "fit_assets",
    "select_best_model",
    "fit_garch11_mle",
    "garch11_variance",
    "ewma_volatility",
    "InsufficientDataError",
]

FloatArray = NDArray[np.float64]
MIN_OBSERVATIONS = 250
_SCALE = 100.0


class InsufficientDataError(ValueError):
    """Serie demasiado corta o con valores no finitos para una estimación fiable."""


class VolModel(str, Enum):
    GARCH = "GARCH"          # σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
    GJR_GARCH = "GJR-GARCH"  # + γ ε²_{t-1}·1[ε<0]  (efecto apalancamiento: las caídas suben más la vol)
    EGARCH = "EGARCH"        # ln σ²_t = ω + α(|z|-E|z|) + γ z + β ln σ²_{t-1}  (sin restricciones de signo)

    @property
    def arch_spec(self) -> dict[str, Any]:
        return {
            VolModel.GARCH: {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
            VolModel.GJR_GARCH: {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
            VolModel.EGARCH: {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
        }[self]


class Distribution(str, Enum):
    NORMAL = "normal"
    STUDENT_T = "t"        # colas gruesas: el default razonable para cripto
    SKEW_T = "skewt"       # colas gruesas + asimetría (Hansen)


# =============================================================================
# Utilidades
# =============================================================================
def log_returns(prices: ArrayLike | pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame | FloatArray:
    """Log-retornos ``ln(P_t / P_{t-1})``. Conserva el índice temporal si la entrada es pandas."""
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        if (prices <= 0).to_numpy().any():
            raise ValueError("Precios no positivos en la serie")
        return np.log(prices).diff().iloc[1:]
    arr = np.asarray(prices, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError("Precios no positivos en la serie")
    return np.diff(np.log(arr), axis=0)


def _clean_returns(returns: ArrayLike | pd.Series, *, min_obs: int = MIN_OBSERVATIONS) -> pd.Series:
    series = returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns, dtype=np.float64))
    series = series.astype(np.float64).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < min_obs:
        raise InsufficientDataError(f"Se requieren >= {min_obs} observaciones finitas (hay {len(series)})")
    if float(series.std()) == 0.0:
        raise InsufficientDataError("Serie de retornos con varianza nula")
    return series


# =============================================================================
# Resultado de ajuste
# =============================================================================
@dataclass(frozen=True)
class VolatilityFit:
    """Resultado inmutable de un ajuste de volatilidad condicional (unidades decimales por periodo)."""

    asset: str | None
    model: VolModel
    dist: Distribution
    params: Mapping[str, float]            # parámetros en la escala del modelo (retornos ×100)
    conditional_vol: pd.Series             # σ_t in-sample (decimal)
    std_residuals: pd.Series               # z_t = ε_t / σ_t  (insumo para Filtered Historical Simulation)
    persistence: float
    unconditional_vol: float | None        # σ de largo plazo (decimal por periodo)
    loglik: float
    aic: float
    bic: float
    nobs: int
    converged: bool
    ljung_box_pvalue: float                # H0: sin autocorrelación en z² (p alto = modelo captura el clustering)
    arch_lm_pvalue: float                  # H0: sin efectos ARCH remanentes
    periods_per_year: int = 365
    _result: Any = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ vistas
    @property
    def current_vol(self) -> float:
        """Previsión a 1 paso σ_{T+1|T} (decimal por periodo)."""
        return float(self.forecast_variance(1)[0] ** 0.5)

    @property
    def current_vol_annualized(self) -> float:
        return self.current_vol * math.sqrt(self.periods_per_year)

    @property
    def unconditional_vol_annualized(self) -> float | None:
        return None if self.unconditional_vol is None else self.unconditional_vol * math.sqrt(self.periods_per_year)

    @property
    def half_life_periods(self) -> float:
        """Periodos para que un shock de varianza se disipe a la mitad: ln(0.5)/ln(persistencia)."""
        if not 0 < self.persistence < 1:
            return math.inf
        return math.log(0.5) / math.log(self.persistence)

    @property
    def nu(self) -> float | None:
        """Grados de libertad de la Student-t / Skew-t (``None`` si Normal)."""
        return self.params.get("nu", self.params.get("eta"))

    # --------------------------------------------------------------- previsión
    def forecast_variance(self, horizon: int, *, simulations: int = 2_000, seed: int = 7) -> FloatArray:
        """
        Trayectoria esperada de varianzas ``σ²_{T+1..T+h}`` (decimal²). Analítica para
        GARCH/GJR; por simulación para EGARCH con h > 1 (no existe forma cerrada).
        """
        if horizon < 1:
            raise ValueError("horizon debe ser >= 1")
        if self._result is None:
            raise RuntimeError("Ajuste sin objeto de resultados de arch")
        kwargs: dict[str, Any] = {"horizon": horizon, "reindex": False}
        if self.model is VolModel.EGARCH and horizon > 1:
            rs = np.random.RandomState(seed)
            kwargs.update(method="simulation", simulations=simulations, random_state=rs)
        variance = self._result.forecast(**kwargs).variance.to_numpy()[-1]
        return np.asarray(variance, dtype=np.float64) / _SCALE**2

    def horizon_vol(self, horizon: int) -> float:
        """Volatilidad agregada a ``horizon`` periodos: sqrt(Σ σ²_{T+k}). Respeta la reversión a la media."""
        return float(np.sqrt(self.forecast_variance(horizon).sum()))

    def summary(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "model": self.model.value,
            "dist": self.dist.value,
            "current_vol_ann": round(self.current_vol_annualized, 4),
            "long_run_vol_ann": None if self.unconditional_vol is None else round(self.unconditional_vol_annualized, 4),
            "persistence": round(self.persistence, 5),
            "half_life": round(self.half_life_periods, 2),
            "nu": None if self.nu is None else round(self.nu, 2),
            "bic": round(self.bic, 2),
            "ljung_box_p": round(self.ljung_box_pvalue, 4),
            "arch_lm_p": round(self.arch_lm_pvalue, 4),
            "converged": self.converged,
        }


def _persistence_and_unconditional(model: VolModel, p: Mapping[str, float]) -> tuple[float, float | None]:
    """Persistencia y varianza incondicional (escala ×100²) por especificación."""
    omega, alpha, beta = p["omega"], p.get("alpha[1]", 0.0), p.get("beta[1]", 0.0)
    if model is VolModel.EGARCH:
        # En EGARCH la persistencia es β; exp(ω/(1-β)) aproxima la varianza de largo plazo
        # (ignora el término de Jensen E[exp(·)] ≥ exp(E[·]); suficiente como referencia).
        return beta, (math.exp(omega / (1 - beta)) if beta < 1 else None)
    gamma = p.get("gamma[1]", 0.0)
    persistence = alpha + beta + 0.5 * gamma  # E[1(ε<0)] = 1/2 con innovaciones simétricas
    return persistence, (omega / (1 - persistence) if persistence < 1 else None)


# =============================================================================
# Ajuste con arch
# =============================================================================
def fit_volatility(
    data: ArrayLike | pd.Series,
    *,
    model: VolModel | str = VolModel.GARCH,
    dist: Distribution | str = Distribution.STUDENT_T,
    is_prices: bool | None = None,
    demean: bool = True,
    asset: str | None = None,
    periods_per_year: int = 365,
    lb_lags: int = 10,
) -> VolatilityFit:
    """
    Ajusta un modelo de volatilidad condicional por máxima verosimilitud.

    Args:
        data: precios o log-retornos (``is_prices=None`` lo detecta: todos > 0 y media > 1 => precios).
        model: ``GARCH`` | ``GJR-GARCH`` | ``EGARCH``.
        dist: distribución de las innovaciones.
        demean: media constante (True) o cero. Para retornos intradía cripto, ``False`` suele bastar.
        periods_per_year: 365 (diario), 8760 (horario), 105120 (5 min).
        lb_lags: retardos para Ljung-Box / ARCH-LM.

    Raises:
        InsufficientDataError: menos de ``MIN_OBSERVATIONS`` observaciones válidas.
    """
    model, dist = VolModel(model), Distribution(dist)
    raw = data if isinstance(data, pd.Series) else pd.Series(np.asarray(data, dtype=np.float64))
    if is_prices is None:
        values = raw.dropna().to_numpy()
        is_prices = bool(len(values) and np.all(values > 0) and values.mean() > 1.0)
    returns = _clean_returns(log_returns(raw) if is_prices else raw)

    spec = model.arch_spec
    am = arch_model(returns * _SCALE, mean="Constant" if demean else "Zero", dist=dist.value, rescale=False, **spec)
    res = am.fit(disp="off", options={"maxiter": 1_000})
    params = {k: float(v) for k, v in res.params.items()}

    std_resid = res.std_resid.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lb_p = float(acorr_ljungbox(std_resid**2, lags=[lb_lags], return_df=True)["lb_pvalue"].iloc[0])
        arch_p = float(het_arch(std_resid, nlags=lb_lags)[1])

    persistence, uncond_var = _persistence_and_unconditional(model, params)
    fit = VolatilityFit(
        asset=asset,
        model=model,
        dist=dist,
        params=params,
        conditional_vol=res.conditional_volatility / _SCALE,
        std_residuals=std_resid,
        persistence=float(persistence),
        unconditional_vol=None if uncond_var is None else math.sqrt(uncond_var) / _SCALE,
        loglik=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        nobs=int(res.nobs),
        converged=res.convergence_flag == 0,
        ljung_box_pvalue=lb_p,
        arch_lm_pvalue=arch_p,
        periods_per_year=periods_per_year,
        _result=res,
    )
    if not fit.converged:
        warnings.warn(f"{asset or ''} {model.value}: el optimizador no convergió", RuntimeWarning, stacklevel=2)
    if persistence >= 1:
        warnings.warn(
            f"{asset or ''} {model.value}: persistencia {persistence:.4f} >= 1 (proceso IGARCH, sin varianza "
            "incondicional finita)",
            RuntimeWarning,
            stacklevel=2,
        )
    return fit


def select_best_model(
    data: ArrayLike | pd.Series,
    candidates: Iterable[tuple[VolModel, Distribution]] = (
        (VolModel.GARCH, Distribution.NORMAL),
        (VolModel.GARCH, Distribution.STUDENT_T),
        (VolModel.GJR_GARCH, Distribution.STUDENT_T),
        (VolModel.EGARCH, Distribution.STUDENT_T),
    ),
    **kwargs: Any,
) -> tuple[VolatilityFit, pd.DataFrame]:
    """Ajusta todas las especificaciones candidatas y devuelve la de menor BIC + tabla comparativa."""
    fits: list[VolatilityFit] = []
    for model, dist in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fits.append(fit_volatility(data, model=model, dist=dist, **kwargs))
        except (np.linalg.LinAlgError, ValueError) as exc:  # una especificación fallida no aborta la selección
            warnings.warn(f"{model.value}/{dist.value} descartado: {exc}", RuntimeWarning, stacklevel=2)
    if not fits:
        raise RuntimeError("Ninguna especificación pudo ajustarse")
    table = pd.DataFrame([f.summary() for f in fits]).sort_values("bic").reset_index(drop=True)
    return min(fits, key=lambda f: f.bic), table


def fit_assets(
    prices: pd.DataFrame,
    *,
    model: VolModel | str = VolModel.GJR_GARCH,
    dist: Distribution | str = Distribution.STUDENT_T,
    **kwargs: Any,
) -> dict[str, VolatilityFit]:
    """Ajusta el mismo modelo a cada columna de un DataFrame de precios (p.ej. columnas BTC, ETH)."""
    return {
        str(col): fit_volatility(prices[col], model=model, dist=dist, is_prices=True, asset=str(col), **kwargs)
        for col in prices.columns
    }


# =============================================================================
# GARCH(1,1) desde cero (SciPy) — vectorizado con lfilter
# =============================================================================
@dataclass(frozen=True, slots=True)
class Garch11Params:
    omega: float
    alpha: float
    beta: float
    mu: float
    loglik: float
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence) if self.persistence < 1 else math.inf


def garch11_variance(eps: FloatArray, omega: float, alpha: float, beta: float, sigma2_0: float) -> FloatArray:
    """
    Recursión σ²_t = ω + α ε²_{t-1} + β σ²_{t-1} resuelta como filtro IIR de primer orden:

        σ²_t − β σ²_{t-1} = ω + α ε²_{t-1}   =>   lfilter(b=[1], a=[1, −β], x)

    con x_0 = σ²_0 (arranque) y x_t = ω + α ε²_{t-1}. Coste O(T) en C, sin bucle Python.
    """
    x = np.empty_like(eps)
    x[0] = sigma2_0
    x[1:] = omega + alpha * eps[:-1] ** 2
    return signal.lfilter([1.0], [1.0, -beta], x)


def fit_garch11_mle(returns: ArrayLike, *, scale: float = _SCALE) -> Garch11Params:
    """
    GARCH(1,1)-Normal por MLE con SciPy (SLSQP, restricción de estacionariedad α + β < 1).

    Implementación de referencia: valida el estimador de ``arch`` y sirve como fallback.
    Devuelve ω en la escala ``scale²`` (igual que ``arch``) y α, β adimensionales.
    """
    r = np.asarray(_clean_returns(returns), dtype=np.float64) * scale
    var0 = float(np.var(r))

    def neg_loglik(theta: FloatArray) -> float:
        mu, omega, alpha, beta = theta
        eps = r - mu
        s2 = garch11_variance(eps, omega, alpha, beta, var0)
        if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
            return 1e12
        return 0.5 * float(np.sum(np.log(2 * np.pi) + np.log(s2) + eps**2 / s2))

    x0 = np.array([r.mean(), var0 * 0.05, 0.08, 0.90])
    res = optimize.minimize(
        neg_loglik,
        x0,
        method="SLSQP",
        bounds=[(None, None), (1e-8, 10 * var0), (1e-6, 0.999), (1e-6, 0.999)],
        constraints=[{"type": "ineq", "fun": lambda t: 0.99999 - t[2] - t[3]}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    mu, omega, alpha, beta = (float(v) for v in res.x)
    return Garch11Params(omega=omega, alpha=alpha, beta=beta, mu=mu, loglik=-float(res.fun), converged=bool(res.success))


# =============================================================================
# EWMA (RiskMetrics)
# =============================================================================
def ewma_volatility(returns: ArrayLike | pd.Series | pd.DataFrame, lam: float = 0.94, *, warmup: int = 30) -> Any:
    """
    Volatilidad EWMA: σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}  (previsión one-step-ahead, sin look-ahead).

    Vectorizado por columnas con ``lfilter`` (acepta matriz T×N). Semilla: varianza de las
    primeras ``warmup`` observaciones. Devuelve el mismo tipo de contenedor que la entrada.
    """
    if not 0 < lam < 1:
        raise ValueError("lam debe estar en (0, 1)")
    is_pandas = isinstance(returns, (pd.Series, pd.DataFrame))
    arr = np.asarray(returns, dtype=np.float64)
    squeeze = arr.ndim == 1
    r = arr.reshape(-1, 1) if squeeze else arr
    if len(r) < 2:
        raise InsufficientDataError("EWMA requiere al menos 2 observaciones")

    seed = np.var(r[: max(2, min(warmup, len(r)))], axis=0)
    x = np.empty_like(r)
    x[0] = seed
    x[1:] = (1 - lam) * r[:-1] ** 2
    # condición inicial: arrancamos el filtro en el estado "seed" ya ponderado
    sigma2 = signal.lfilter([1.0], [1.0, -lam], x, axis=0)
    vol = np.sqrt(sigma2)
    vol = vol[:, 0] if squeeze else vol
    if is_pandas:
        return (
            pd.Series(vol, index=returns.index, name=getattr(returns, "name", None))
            if squeeze
            else pd.DataFrame(vol, index=returns.index, columns=returns.columns)
        )
    return vol


# =============================================================================
# Demo:  python -m quant_models.volatility
# =============================================================================
if __name__ == "__main__":  # pragma: no cover
    from arch.univariate import StudentsT

    generator = arch_model(None, mean="Zero", vol="GARCH", p=1, o=1, q=1)
    generator.distribution = StudentsT(seed=np.random.default_rng(7))
    sim = generator.simulate([0.05, 0.05, 0.08, 0.88, 5.0], nobs=2_000, burn=500)
    btc = pd.Series(60_000 * np.exp(np.cumsum(sim["data"].to_numpy() / _SCALE)), name="BTC")
    best, table = select_best_model(btc, is_prices=True, asset="BTC")
    pd.set_option("display.width", 200)
    print(table.to_string())
    print(f"\nMejor: {best.model.value}/{best.dist.value}  vol actual anual = {best.current_vol_annualized:.2%}")
    print(f"Vol agregada 10 días = {best.horizon_vol(10):.2%}")
