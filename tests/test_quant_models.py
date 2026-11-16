"""
Tests de quant_models: recuperación de parámetros GARCH conocidos, equivalencia de las
implementaciones vectorizadas con bucles de referencia, propiedades teóricas de VaR/ES
y casos analíticos del simulador de difusión con saltos.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from arch.univariate import Normal, StudentsT
from scipy import stats

from quant_models.monte_carlo import (
    AssetParams,
    CrisisDynamics,
    JumpDiffusionSimulator,
    JumpParams,
    MarginAccount,
    ScenarioConfig,
    per_day,
)
from quant_models.var_es import (
    component_var,
    cornish_fisher_var,
    ewma_covariance,
    exposures,
    filtered_historical_scenarios,
    historical_var_es,
    kupiec_pof_test,
    parametric_var_es,
    portfolio_pnl,
    tail_var_es,
)
from quant_models.volatility import (
    Distribution,
    InsufficientDataError,
    VolModel,
    ewma_volatility,
    fit_assets,
    fit_garch11_mle,
    fit_volatility,
    garch11_variance,
    log_returns,
    select_best_model,
)


def simulate(params, *, o=0, dist="t", nobs=3_000, seed=0, vol="GARCH"):
    """Serie GARCH simulada (en %, como arch) con semilla reproducible."""
    am = arch_model(None, mean="Zero", vol=vol, p=1, o=o, q=1)
    rng = np.random.default_rng(seed)
    am.distribution = StudentsT(seed=rng) if dist == "t" else Normal(seed=rng)
    return am.simulate(params, nobs=nobs, burn=500)["data"].to_numpy() / 100.0


# =============================================================================
# Volatilidad
# =============================================================================
def test_garch_t_recovers_true_parameters():
    r = simulate([0.05, 0.08, 0.90, 6.0], seed=1)
    fit = fit_volatility(r, model=VolModel.GARCH, dist=Distribution.STUDENT_T, is_prices=False, demean=False)
    assert fit.converged
    assert fit.params["alpha[1]"] == pytest.approx(0.08, abs=0.03)
    assert fit.params["beta[1]"] == pytest.approx(0.90, abs=0.04)
    assert fit.nu == pytest.approx(6.0, abs=2.0)
    assert fit.persistence == pytest.approx(0.98, abs=0.02)
    assert fit.ljung_box_pvalue > 0.01 and fit.arch_lm_pvalue > 0.01
    # Varianza incondicional verdadera: ω/(1−α−β) = 2.5 (%²) -> σ = 1.58 %
    assert fit.unconditional_vol == pytest.approx(math.sqrt(2.5) / 100, rel=0.35)
    assert len(fit.conditional_vol) == len(r) and (fit.conditional_vol > 0).all()


def test_scipy_mle_matches_arch_estimator():
    r = simulate([0.05, 0.10, 0.85], dist="normal", seed=3)
    own = fit_garch11_mle(r)
    ref = fit_volatility(r, model=VolModel.GARCH, dist=Distribution.NORMAL, is_prices=False)
    assert own.converged
    assert own.alpha == pytest.approx(ref.params["alpha[1]"], abs=0.01)
    assert own.beta == pytest.approx(ref.params["beta[1]"], abs=0.015)
    assert own.loglik == pytest.approx(ref.loglik, rel=1e-3)


def test_lfilter_recursions_match_naive_loops():
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(500)
    fast = garch11_variance(eps, 0.05, 0.08, 0.9, 1.3)
    slow = np.empty(500)
    slow[0] = 1.3
    for t in range(1, 500):
        slow[t] = 0.05 + 0.08 * eps[t - 1] ** 2 + 0.9 * slow[t - 1]
    np.testing.assert_allclose(fast, slow, rtol=1e-12)

    R = rng.standard_normal((400, 2)) * [0.03, 0.04]
    vol = ewma_volatility(R, lam=0.94, warmup=30)
    s2 = np.var(R[:30], axis=0)
    for t in range(1, 400):
        s2 = 0.94 * s2 + 0.06 * R[t - 1] ** 2
    np.testing.assert_allclose(vol[-1], np.sqrt(s2), rtol=1e-12)
    df = pd.DataFrame(R, columns=["BTC", "ETH"])
    assert list(ewma_volatility(df).columns) == ["BTC", "ETH"]


def test_model_selection_detects_leverage_effect():
    r = simulate([0.05, 0.02, 0.18, 0.85, 7.0], o=1, seed=5, nobs=4_000)
    best, table = select_best_model(r, is_prices=False)
    assert best.model in (VolModel.GJR_GARCH, VolModel.EGARCH)
    assert table["bic"].is_monotonic_increasing and len(table) == 4


def test_forecasts_mean_revert_and_egarch_horizon():
    prices = pd.Series(60_000 * np.exp(np.cumsum(simulate([0.05, 0.08, 0.9, 6.0], seed=8))))
    fit = fit_volatility(prices, model=VolModel.GJR_GARCH, asset="BTC")
    path = fit.forecast_variance(500)
    target = fit.unconditional_vol**2
    assert abs(path[-1] - target) < abs(path[0] - target) + 1e-12  # converge a la varianza de largo plazo
    assert fit.horizon_vol(10) == pytest.approx(math.sqrt(path[:10].sum()))
    assert fit.current_vol_annualized == pytest.approx(fit.current_vol * math.sqrt(365))
    eg = fit_volatility(prices, model=VolModel.EGARCH)
    assert eg.forecast_variance(5).shape == (5,) and eg.half_life_periods > 0

    multi = fit_assets(pd.DataFrame({"BTC": prices, "ETH": prices * 0.04}))
    assert set(multi) == {"BTC", "ETH"}


def test_input_validation():
    with pytest.raises(InsufficientDataError):
        fit_volatility(np.random.default_rng(0).normal(size=100) * 0.01, is_prices=False)
    with pytest.raises(ValueError):
        log_returns([100.0, -1.0, 101.0])


# =============================================================================
# VaR / ES
# =============================================================================
def test_tail_var_es_exact_and_vectorized():
    losses = np.arange(1, 1001, dtype=float)
    var, es = tail_var_es(-losses, 0.99)
    assert var == 991 and es == pytest.approx(995.5)
    rng = np.random.default_rng(1)
    pnl = rng.standard_normal((5_000, 3)) * [100, 200, 50]
    var_v, es_v = tail_var_es(pnl, 0.99)
    for j in range(3):
        v, e = tail_var_es(pnl[:, j], 0.99)
        assert var_v[j] == v and es_v[j] == pytest.approx(e, rel=1e-12)
    assert np.all(es_v >= var_v)


def test_historical_converges_to_parametric_normal():
    rng = np.random.default_rng(2)
    # vols intradía pequeñas: la convexidad de expm1 (revaluación exacta) es despreciable frente
    # al modelo lineal delta-normal, de modo que ambos métodos deben coincidir
    cov = np.array([[0.005**2, 0.8 * 0.005 * 0.007], [0.8 * 0.005 * 0.007, 0.007**2]])
    R = rng.multivariate_normal([0, 0], cov, size=400_000)
    e = exposures([1.0, -10.0], [95_000, 3_500])
    hist = historical_var_es(R, e, alpha=0.99)
    para = parametric_var_es(e, cov, alpha=0.99)
    assert hist.var == pytest.approx(para.var, rel=0.015)
    assert hist.es == pytest.approx(para.es, rel=0.015)


def test_parametric_t_es_matches_simulation():
    nu, sigma, alpha = 4.0, 1_000.0, 0.99
    est = parametric_var_es([1.0], [[sigma**2]], alpha=alpha, dist="t", nu=nu)
    x = stats.t.rvs(nu, size=2_000_000, random_state=3) * sigma * math.sqrt((nu - 2) / nu)
    var_mc, es_mc = tail_var_es(x, alpha)
    assert est.var == pytest.approx(var_mc, rel=0.02)
    assert est.es == pytest.approx(es_mc, rel=0.03)
    assert est.es > parametric_var_es([1.0], [[sigma**2]], alpha=alpha).es  # colas gruesas


def test_multi_account_parametric_and_component_var():
    cov = np.array([[0.0009, 0.00096], [0.00096, 0.0016]])
    E = np.array([[95_000.0, 0.0], [95_000.0, -94_500.0], [0.0, 70_000.0]])
    batch = parametric_var_es(E, cov)
    singles = [parametric_var_es(row, cov).var for row in E]
    np.testing.assert_allclose(batch.var, singles)
    assert batch.var[1] < batch.var[0]  # la cobertura reduce el riesgo
    contrib = component_var(E, cov)
    np.testing.assert_allclose(contrib.sum(axis=1), batch.var, rtol=1e-10)


def test_overlapping_vs_sqrt_scaling_iid():
    R = np.random.default_rng(4).normal(0, 0.02, size=(100_000, 1))
    a = historical_var_es(R, [1e5], horizon=10, scaling="overlapping")
    b = historical_var_es(R, [1e5], horizon=10, scaling="sqrt")
    assert a.var == pytest.approx(b.var, rel=0.08)


def test_fhs_rescales_to_current_vol():
    Z = np.random.default_rng(5).standard_normal((50_000, 2))
    scen = filtered_historical_scenarios(Z, [0.05, 0.08])
    np.testing.assert_allclose(scen.std(axis=0), [0.05, 0.08], rtol=0.02)
    with pytest.raises(ValueError):
        filtered_historical_scenarios(Z, [0.05])


def test_portfolio_pnl_is_exact_revaluation():
    R = np.log([[0.5, 1.1]])
    pnl = portfolio_pnl(R, exposures([2.0, -10.0], [100.0, 10.0]))
    assert pnl[0] == pytest.approx(2 * 100 * -0.5 + -10 * 10 * 0.1)


def test_cornish_fisher_and_ewma_cov():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(500_000)
    assert cornish_fisher_var(x) == pytest.approx(stats.norm.ppf(0.99), rel=0.02)
    skewed = -stats.lognorm.rvs(0.6, size=500_000, random_state=6)
    skewed = (skewed - skewed.mean()) / skewed.std()
    assert cornish_fisher_var(skewed) > cornish_fisher_var(x)

    R = rng.standard_normal((300, 2)) * 0.03
    w = 0.06 * 0.94 ** np.arange(299, -1, -1)
    w /= w.sum()
    ref = sum(w[t] * np.outer(R[t], R[t]) for t in range(300))
    np.testing.assert_allclose(ewma_covariance(R), ref, rtol=1e-12)


def test_kupiec_backtest():
    rng = np.random.default_rng(7)
    pnl = rng.standard_normal(2_000)
    good = kupiec_pof_test(pnl, stats.norm.ppf(0.99), 0.99)
    bad = kupiec_pof_test(pnl, stats.norm.ppf(0.95), 0.99)  # VaR infraestimado
    assert not good.reject and bad.reject and bad.exceptions > good.exceptions


# =============================================================================
# Monte Carlo
# =============================================================================
def _flat(**over):
    base = ScenarioConfig(
        assets=(AssetParams("BTC", 95_000, sigma=0.0), AssetParams("ETH", 3_500, sigma=0.0)),
        correlation=np.eye(2), n_paths=200, n_steps=48, seed=0,
    )
    return ScenarioConfig(**{**base.__dict__, **over})


def test_deterministic_drift():
    cfg = _flat(assets=(AssetParams("BTC", 100.0, sigma=0.0, mu=0.365), AssetParams("ETH", 10.0, sigma=0.0)))
    res = JumpDiffusionSimulator(cfg).run()
    np.testing.assert_allclose(res.paths("BTC")[:, -1], 100 * math.exp(0.365 / 365))
    np.testing.assert_allclose(res.paths("ETH"), 10.0)


def test_martingale_with_compensated_jumps():
    cfg = ScenarioConfig.normal(n_paths=40_000, n_steps=96, compensate_jumps=True, seed=11,
                                jumps=JumpParams(intensity=per_day(1.0), mean=-0.03, std=0.03))
    res = JumpDiffusionSimulator(cfg).run()
    st = res.prices[:, -1, :] / res.prices[:, 0, :]
    se = st.std(axis=0) / math.sqrt(len(st))
    assert np.all(np.abs(st.mean(axis=0) - 1) < 4 * se)


def test_diffusion_correlation_and_antithetic():
    cfg = ScenarioConfig.normal(n_paths=20_000, n_steps=24, seed=2,
                                jumps=JumpParams(), correlation=np.array([[1, 0.7], [0.7, 1]]))
    res = JumpDiffusionSimulator(cfg).run()
    inc = np.diff(np.log(res.prices), axis=1).reshape(-1, 2)
    assert np.corrcoef(inc.T)[0, 1] == pytest.approx(0.7, abs=0.01)
    assert abs(inc.mean(axis=0) / inc.std(axis=0)).max() < 0.01


def test_forced_crash_liquidity_and_vol_cap():
    dyn = CrisisDynamics(vol_shock_per_event=5.0, liquidity_drain_per_event=1.0, max_vol_multiplier=3.0)
    cfg = _flat(
        assets=(AssetParams("BTC", 95_000, sigma=0.0, crash_pct=0.18), AssetParams("ETH", 3_500, sigma=0.0, crash_pct=0.24)),
        crash_time_days=0.25, dynamics=dyn,
    )
    res = JumpDiffusionSimulator(cfg).run()
    np.testing.assert_allclose(res.prices[:, -1, :], np.broadcast_to([95_000 * 0.82, 3_500 * 0.76], (cfg.n_paths, 2)))
    step = 12
    assert np.all(res.liquidity[:, step + 1] == pytest.approx(math.exp(-1.0)))
    assert np.all(res.liquidity[:, step] == 1.0)
    assert res.vol_multiplier.max() == 3.0 and np.all(res.vol_multiplier[:, : step + 1] == 1.0)
    dd = res.max_drawdown()
    np.testing.assert_allclose(dd, np.broadcast_to([0.18, 0.24], (cfg.n_paths, 2)))


def test_hawkes_cascade_amplifies_jumps():
    base = dict(n_paths=20_000, n_steps=288, seed=3)
    lam = per_day(2.0)
    poisson = JumpDiffusionSimulator(ScenarioConfig.normal(jumps=JumpParams(intensity=lam), **base)).run()
    hawkes = JumpDiffusionSimulator(ScenarioConfig.normal(
        jumps=JumpParams(intensity=lam, hawkes_excitation=per_day(12), hawkes_decay=per_day(24)), **base)).run()
    m_p = poisson.jump_events.sum(axis=1).mean()
    m_h = hawkes.jump_events.sum(axis=1).mean()
    assert m_p == pytest.approx(2.0, rel=0.05)
    assert m_h == pytest.approx(2.0 / (1 - 0.5), rel=0.12)  # λ0/(1−n), n = 0.5 (con corrección de borde)
    assert hawkes.jump_events.sum(axis=1).var() > poisson.jump_events.sum(axis=1).var()  # clustering
    with pytest.raises(ValueError):
        JumpParams(intensity=1, hawkes_excitation=2, hawkes_decay=1)


def test_reproducibility_and_recalibration():
    a = JumpDiffusionSimulator(ScenarioConfig.black_swan(n_paths=500, seed=9)).run()
    b = JumpDiffusionSimulator(ScenarioConfig.black_swan(n_paths=500, seed=9)).run()
    np.testing.assert_array_equal(a.prices, b.prices)
    cfg = ScenarioConfig.normal().with_volatility({"BTC": 0.4}, multiplier=2).with_prices({"ETH": 4_000})
    assert cfg.assets[0].sigma == 0.8 and cfg.assets[1].s0 == 4_000


def test_margin_stress_gap_risk_analytic():
    """Shock determinista −18 %: el déficit del Insurance Fund tiene forma cerrada."""
    dyn = CrisisDynamics(liquidity_drain_per_event=0.5, base_spread_bps=3.0)
    cfg = _flat(assets=(AssetParams("BTC", 95_000, sigma=0.0, crash_pct=0.18), AssetParams("ETH", 3_500, sigma=0.0)),
                crash_time_days=0.5, dynamics=dyn, name="crash")
    res = JumpDiffusionSimulator(cfg).run()
    whale = MarginAccount("whale", 4_750, {"BTC": 1.0}, {"BTC": 95_000})
    rep = res.margin_stress(whale)
    p_crash = 95_000 * 0.82
    equity = 4_750 + (p_crash - 95_000)
    slip = p_crash * 3e-4 / math.exp(-0.5)
    assert rep.liquidation_probability == 1.0 and rep.bad_debt_probability == 1.0
    assert rep.expected_deficit == pytest.approx(-(equity - slip))
    # el shock se aplica en el incremento 24->25: visible en el instante 25 (25 × 30 min)
    assert rep.median_time_to_liquidation_min == pytest.approx(25 * 30.0)
    assert rep.initial_margin_ratio == pytest.approx(4_750 / (95_000 * 0.025))

    safe = MarginAccount("safe", 1e6, {"BTC": 1.0}, {"BTC": 95_000})
    assert res.margin_stress(safe).liquidation_probability == 0.0


def test_gradual_crash_liquidates_before_bankruptcy():
    """Mismo −18 % repartido en 18 pasos: total idéntico, pero la liquidación llega antes de la quiebra."""
    dyn = CrisisDynamics(liquidity_drain_per_event=0.5, base_spread_bps=3.0)
    base = dict(assets=(AssetParams("BTC", 95_000, sigma=0.0, crash_pct=0.18), AssetParams("ETH", 3_500, sigma=0.0)),
                crash_time_days=0.25, dynamics=dyn)
    res = JumpDiffusionSimulator(_flat(crash_steps=18, **base)).run()
    btc = res.paths("BTC")[0]
    steps = btc[1:] / btc[:-1] - 1
    np.testing.assert_allclose(steps[12:30], 0.82 ** (1 / 18) - 1)
    assert np.all(steps[:12] == 0) and np.all(steps[30:] == 0)
    assert btc[-1] == pytest.approx(95_000 * 0.82)

    whale = MarginAccount("whale", 4_750, {"BTC": 1.0}, {"BTC": 95_000})
    assert res.margin_stress(whale).bad_debt_probability == 0.0
    assert JumpDiffusionSimulator(_flat(crash_steps=1, **base)).run().margin_stress(whale).bad_debt_probability == 1.0
    with pytest.raises(ValueError):
        _flat(crash_steps=0, **base)


def test_black_swan_separates_agent_profiles_and_mc_var():
    res = JumpDiffusionSimulator(ScenarioConfig.black_swan(n_paths=5_000, seed=21)).run()
    agents = [
        MarginAccount("whale_20x", 4_750, {"BTC": 1.0}, {"BTC": 95_000}),
        MarginAccount("arb_3x", 60_000, {"BTC": 1.0, "ETH": -27.0}, {"BTC": 95_000, "ETH": 3_500}),
    ]
    table = res.margin_stress_table(agents)
    assert table.loc["whale_20x", "liquidation_probability"] > 0.99
    assert table.loc["arb_3x", "liquidation_probability"] < 0.05
    est = res.var_es(np.array([[1.0, 0.0], [1.0, -27.0]]), alpha=0.99)
    assert est.var.shape == (2,) and est.var[0] > est.var[1] and np.all(est.es >= est.var)


def test_simulation_performance():
    cfg = ScenarioConfig.black_swan(n_paths=10_000, n_steps=288, seed=1)
    t0 = time.perf_counter()
    res = JumpDiffusionSimulator(cfg).run()
    assert time.perf_counter() - t0 < 5.0 and res.prices.shape == (10_000, 289, 2)
    assert np.all(np.isfinite(res.prices)) and np.all(res.prices > 0)
