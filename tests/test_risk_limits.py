"""Límites de riesgo por cuenta: cálculo puro, guardarraíles del agente, gateway (server-side) y CLI."""

from __future__ import annotations

import math

import pytest

from ai_agents.order_gateway import OrderGateway
from ai_agents.orchestrator import apply_limit_overrides, parse_overrides
from ai_agents.trader_agents import (
    AccountView,
    AgentPersona,
    MarketMode,
    MarketObservation,
    OrderIntent,
    TargetExposure,
    TradeDecision,
    TradingAgent,
    default_agents,
)
from risk_engine.risk_limits import RiskLimits, stress_coverage
from tests.helpers import make_store, seed_account


# ============================================================================= núcleo puro
def test_gross_cap_is_min_of_leverage_and_notional():
    lim = RiskLimits(3.0, 10e6)
    assert lim.gross_cap(1e6) == 3e6                       # manda el apalancamiento
    assert lim.gross_cap(5e6) == 10e6                      # manda el tope absoluto
    assert lim.gross_cap(-1.0) == 0.0                      # equity ≤ 0: nada que abrir
    assert lim.headroom(8e6, 5e6) == pytest.approx(2e6)
    assert RiskLimits(20).gross_cap(10_000) == 200_000     # sin tope absoluto


def test_breach_reasons():
    lim = RiskLimits(5.0, 1e6)
    assert lim.breach(900_000, 200_000) is None
    b = lim.breach(1.2e6, 1e6)
    assert b.limit == "MAX_GROSS_NOTIONAL" and "nocional" in str(b)
    b = lim.breach(600_000, 100_000)
    assert b.limit == "MAX_LEVERAGE" and b.value == pytest.approx(6.0) and "6.00x" in str(b)
    assert lim.breach(1.0, 0.0).limit == "MAX_LEVERAGE"   # equity nula
    with pytest.raises(ValueError):
        RiskLimits(0)


def test_stress_coverage_flags_the_incident_calibration():
    """Calibración del incidente (manipulador 5x sin tope) vs calibración actual, salto adverso del 30 %."""
    old = stress_coverage([("whale_trader", 10_000, RiskLimits(20)),
                           ("arbitrageur", 100_000, RiskLimits(3)),
                           ("market_manipulator", 5e6, RiskLimits(5))], insurance_fund=100_000, jump=0.30)
    assert old.by_account["market_manipulator"] == pytest.approx(2.5e6)       # 25M · 30 % − 5M
    assert old.ratio == pytest.approx(100_000 / (50_000 + 2.5e6)) and not old.adequate  # ≈ 0.04x
    new = stress_coverage([(a.persona.name, a.persona.initial_collateral, a.persona.limits)
                           for a in default_agents(use_llm=False)], insurance_fund=100_000, jump=0.30)
    assert new.by_account["market_manipulator"] == 0.0 and new.adequate and new.ratio >= 1.5
    assert stress_coverage([], 1.0).ratio == math.inf


# ============================================================================= capa cliente (agente)
def _obs():
    marks = {"BTC": 100_000.0, "ETH": 4_000.0}
    return MarketObservation(MarketMode.NORMAL, marks, marks, {s: 0.0 for s in marks}, {s: 1.0 for s in marks},
                             {s: 1.0 for s in marks})


def test_agent_guardrail_clips_to_notional_cap():
    persona = AgentPersona.arbitrageur(max_gross_notional_usd=50_000.0, max_order_notional_usd=1e9)
    view = AccountView(1, persona.name, 100_000, 100_000, 0, math.inf, 100_000, {})
    agent = TradingAgent(persona)
    notes: list[str] = []
    decision = TradeDecision(rationale="t", confidence=0.5, targets=[
        TargetExposure(symbol="BTC", target_notional_usd=150_000), TargetExposure(symbol="ETH", target_notional_usd=-150_000)])
    intents = agent._guard_targets(decision, _obs(), view, notes)
    assert sum(i.notional for i in intents) == pytest.approx(50_000, rel=1e-3)
    assert any("recortada" in n for n in notes)


# ============================================================================= capa servidor (gateway)
def _store():
    asks = [(100_000 + 10 * i, 50.0) for i in range(1, 50)]
    bids = [(100_000 - 10 * i, 50.0) for i in range(1, 50)]
    return make_store({"BTC": {"mark": 100_000.0, "bids": bids, "asks": asks}})


@pytest.mark.integration
def test_gateway_enforces_notional_cap_but_never_blocks_reductions(clean_db):
    gw = OrderGateway(clean_db, _store())
    uid = seed_account(clean_db, "m", "MARKET_MANIPULATOR", 1e6, max_leverage=5, max_gross_notional=250_000)
    px = {"BTC": 100_000.0}
    assert gw.headroom_usd(uid) == pytest.approx(250_000)
    (f,) = gw.execute(uid, [OrderIntent("BTC", -2.0, 1e5, False)], leverage=5, price_overrides=px)
    assert f.status == "FILLED" and gw.headroom_usd(uid) == pytest.approx(50_000, rel=1e-3)
    (g,) = gw.execute(uid, [OrderIntent("BTC", -1.0, 1e5, False)], leverage=5, price_overrides=px)  # 300k > 250k
    assert g.status == "REJECTED" and "nocional bruto" in g.reason and g.position_after == pytest.approx(-2.0)
    (h,) = gw.execute(uid, [OrderIntent("BTC", 1.5, 1e5, True)], leverage=5, price_overrides=px)     # reducción
    assert h.status == "FILLED" and h.position_after == pytest.approx(-0.5)


@pytest.mark.integration
def test_gateway_enforces_account_leverage_independently_of_agent(clean_db):
    """El agente pide 20x (su persona lo permitiría) pero la cuenta tiene 2x: manda la cuenta."""
    gw = OrderGateway(clean_db, _store())
    uid = seed_account(clean_db, "w", "WHALE_AGGRESSIVE", 10_000, max_leverage=2)
    px = {"BTC": 100_000.0}
    (f,) = gw.execute(uid, [OrderIntent("BTC", 0.5, 1e5, False)], leverage=20, price_overrides=px)   # 5x
    assert f.status == "REJECTED" and "apalancamiento" in f.reason
    (g,) = gw.execute(uid, [OrderIntent("BTC", 0.15, 1e5, False)], leverage=20, price_overrides=px)  # 1.5x
    assert g.status == "FILLED"
    frozen = seed_account(clean_db, "f", "WHALE_AGGRESSIVE", 1e6, status="FROZEN")
    assert gw.headroom_usd(frozen) == 0.0


@pytest.mark.integration
def test_bootstrap_syncs_limits_and_reports_coverage(clean_db):
    from ai_agents.orchestrator import Orchestrator, OrchestratorConfig
    from tests.helpers import seed_insurance_fund

    seed_insurance_fund(clean_db, 100_000)
    seed_account(clean_db, "market_manipulator", "MARKET_MANIPULATOR", 5e6, max_leverage=5)   # cuenta "vieja"
    agents = default_agents(use_llm=False)
    orch = Orchestrator(clean_db, config=OrchestratorConfig(control_interval_s=None), agents=agents)
    ids = orch.bootstrap()
    with clean_db.transaction(readonly=True) as cur:
        cur.execute("SELECT max_leverage, max_gross_notional_usd FROM users_collateral WHERE user_id = %s",
                    (ids["market_manipulator"],))
        row = cur.fetchone()
    assert float(row["max_leverage"]) == 3.0 and float(row["max_gross_notional_usd"]) == 10e6
    assert orch.coverage is not None and orch.coverage.adequate


# ============================================================================= CLI
def test_cli_limit_overrides():
    assert parse_overrides(["whale_trader=10", "arbitrageur=none"], allow_none=True) == {
        "whale_trader": 10.0, "arbitrageur": None}
    for bad in (["whale_trader"], ["whale_trader=0"], ["=3"]):
        with pytest.raises(ValueError):
            parse_overrides(bad)
    agents = apply_limit_overrides(default_agents(use_llm=False), max_leverage={"whale_trader": 5},
                                   max_notional={"market_manipulator": 2e6, "arbitrageur": None})
    by = {a.persona.name: a.persona for a in agents}
    assert by["whale_trader"].max_leverage == 5 and by["whale_trader"].target_leverage == 5
    assert by["market_manipulator"].max_gross_notional_usd == 2e6 and by["arbitrageur"].max_gross_notional_usd is None
    with pytest.raises(ValueError, match="desconocidos"):
        apply_limit_overrides(default_agents(use_llm=False), max_leverage={"nobody": 2})
