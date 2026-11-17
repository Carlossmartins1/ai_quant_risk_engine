"""
Tests de la Fase 5: guardarraíles de los agentes, motor LLM (LangChain con modelo simulado),
mercado sombra, pasarela de órdenes (netting sobre PostgreSQL) y orquestador end-to-end.
"""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest
from langchain_core.runnables import RunnableLambda

from ai_agents.order_gateway import OrderGateway
from ai_agents.orchestrator import Orchestrator, OrchestratorConfig
from ai_agents.shadow_market import BaseQuote, MonteCarloPathSource, ReconvergingSource, ShadowMarket
from ai_agents.trader_agents import (
    AccountView,
    AgentAction,
    AgentPersona,
    CircuitBreaker,
    LLMDecisionEngine,
    ManipulationPlan,
    MarketMode,
    MarketObservation,
    OrderIntent,
    StressBurst,
    TargetExposure,
    TradeDecision,
    TradingAgent,
)
from quant_models.monte_carlo import ScenarioConfig
from tests.helpers import make_store, seed_account, seed_insurance_fund

MARKS = {"BTC": 95_000.0, "ETH": 3_500.0}


def obs(mode=MarketMode.NORMAL, marks=MARKS, rets=None):
    return MarketObservation(mode, marks, marks, {s: 0.0 for s in marks}, {s: 1.0 for s in marks},
                             {s: 1.0 for s in marks}, rets or {"BTC": 0.001})


def account(equity=10_000.0, positions=None, health=math.inf, liquidatable=False):
    return AccountView(1, "x", equity, equity, 0.0, health, equity, positions or {}, liquidatable)


class FixedEngine:
    name = "LLM"

    def __init__(self, decision):
        self.decision = decision

    def decide(self, *a):
        return self.decision


# ============================================================================= guardarraíles
def test_whale_leverage_clamped_and_symbols_filtered():
    persona = AgentPersona.whale(symbols=("BTC",), max_order_notional_usd=1e9)
    decision = TradeDecision(rationale="all in", confidence=1.0, targets=[
        TargetExposure(symbol="BTC", target_notional_usd=1_000_000), TargetExposure(symbol="ETH", target_notional_usd=-5_000)])
    action = TradingAgent(persona, engine=FixedEngine(decision)).act(obs(), account(10_000))
    (intent,) = action.intents
    assert intent.symbol == "BTC" and intent.notional == pytest.approx(200_000, rel=1e-4)  # 20x
    assert any("recortada a 200,000 USD" in n for n in action.guardrail_notes)
    assert any("ETH no permitido" in n for n in action.guardrail_notes)
    assert intent.delta_quantity == round(intent.delta_quantity, 5)  # lot size BTC


def test_arbitrageur_neutrality_enforced():
    persona = AgentPersona.arbitrageur()
    decision = TradeDecision(rationale="sesgado", confidence=0.5, targets=[
        TargetExposure(symbol="BTC", target_notional_usd=120_000), TargetExposure(symbol="ETH", target_notional_usd=-40_000)])
    action = TradingAgent(persona, engine=FixedEngine(decision)).act(obs(), account(100_000))
    notional = {i.symbol: i.delta_quantity * MARKS[i.symbol] for i in action.intents}
    assert notional["BTC"] == pytest.approx(40_000, rel=1e-3) and notional["ETH"] == pytest.approx(-40_000, rel=1e-3)


def test_order_splitting_band_and_reduce_only():
    persona = AgentPersona.whale(max_order_notional_usd=50_000)
    big = TradeDecision(rationale="", confidence=1, targets=[TargetExposure(symbol="BTC", target_notional_usd=150_000)])
    (i,) = TradingAgent(persona, engine=FixedEngine(big)).act(obs(), account(10_000)).intents
    assert i.notional == pytest.approx(50_000, rel=1e-3)                       # fraccionada
    tiny = TradeDecision(rationale="", confidence=1, targets=[TargetExposure(symbol="BTC", target_notional_usd=95_500)])
    pos = {"BTC": (1.0, 95_000.0, 0.0)}
    assert TradingAgent(persona, engine=FixedEngine(tiny)).act(obs(), account(10_000, pos)).intents == ()  # banda
    flat = TradeDecision(rationale="", confidence=1, targets=[TargetExposure(symbol="BTC", target_notional_usd=0)])
    (r,) = TradingAgent(AgentPersona.whale(), engine=FixedEngine(flat)).act(obs(), account(10_000, pos)).intents
    assert r.delta_quantity == pytest.approx(-1.0) and r.reduce_only


def test_insolvent_or_liquidatable_accounts_do_not_trade():
    agent = TradingAgent(AgentPersona.whale())
    assert agent.act(obs(), account(-50.0)).source == "SKIPPED"
    assert agent.act(obs(), account(500.0, liquidatable=True)).source == "SKIPPED"


def test_manipulator_only_in_black_swan_and_capped_by_room():
    m = TradingAgent(AgentPersona.manipulator(burst_every_n_decisions=1))
    assert m.act(obs(MarketMode.NORMAL), account(5e6)).source == "SKIPPED"
    action = m.act(obs(MarketMode.BLACK_SWAN), account(1e6))     # room = 5x · 1M = 5M
    assert [b.symbol for b in action.bursts] == ["BTC", "ETH"]
    assert sum(b.notional_usd for b in action.bursts) <= 5e6
    plan = ManipulationPlan(rationale="x", bursts=[StressBurst(symbol="BTC", side="SELL", notional_usd=9e9, slices=2)])
    capped = TradingAgent(AgentPersona.manipulator(), engine=FixedEngine(plan)).act(obs(MarketMode.BLACK_SWAN), account(1e6))
    assert capped.bursts[0].notional_usd == 1_000_000  # max_order_notional 500k × 2 slices


def test_rule_policies_shape():
    whale = TradingAgent(AgentPersona.whale(directional_bias="SHORT")).act(obs(), account(10_000))
    assert whale.intents[0].delta_quantity < 0 and whale.intents[0].notional == pytest.approx(190_000, rel=1e-3)
    arb = TradingAgent(AgentPersona.arbitrageur()).act(obs(), account(100_000))
    legs = {i.symbol: i.delta_quantity * MARKS[i.symbol] for i in arb.intents}
    assert legs["BTC"] == pytest.approx(-legs["ETH"], rel=1e-3) and legs["BTC"] == pytest.approx(125_000, rel=1e-3)
    derisk = TradingAgent(AgentPersona.arbitrageur()).act(obs(), account(100_000, health=2.0))
    assert sum(abs(i.notional) for i in derisk.intents) == pytest.approx(125_000, rel=1e-3)


# ============================================================================= motor LLM (LangChain)
def test_llm_engine_structured_output_and_memory():
    seen = []

    def fake_llm(schema):
        def respond(prompt_value):
            msgs = prompt_value.to_messages()
            seen.append((msgs[0].content, msgs[1].content))
            return schema.model_validate({"rationale": "LLM dice largo", "confidence": 0.7,
                                          "targets": [{"symbol": "BTC", "target_notional_usd": 50_000}]})
        return RunnableLambda(respond)

    agent = TradingAgent(AgentPersona.whale(), engine=LLMDecisionEngine(chain_factory=fake_llm))
    a1 = agent.act(obs(), account(10_000))
    assert a1.source == "LLM" and a1.rationale == "LLM dice largo" and a1.intents[0].symbol == "BTC"
    assert "Whale" in seen[0][0] and '"max_leverage": 20.0' in seen[0][1]
    agent.record_outcome(a1, account(9_950))
    agent.act(obs(), account(9_950))
    assert '"equity_after": 9950' in seen[1][1]  # la memoria episódica viaja en el prompt


def test_llm_failure_falls_back_to_rules():
    def broken(schema):
        return RunnableLambda(lambda _: (_ for _ in ()).throw(TimeoutError("rate limited")))

    action = TradingAgent(AgentPersona.arbitrageur(), engine=LLMDecisionEngine(chain_factory=broken)).act(obs(), account(100_000))
    assert action.source == "RULES_FALLBACK" and len(action.intents) == 2


# ============================================================================= mercado sombra
def _market(weight=0.5):
    from ai_agents.shadow_market import ShadowMarketConfig

    m = ShadowMarket(("BTC", "ETH"), ShadowMarketConfig(mark_oracle_weight=weight))
    m.step({s: BaseQuote(s, p, p) for s, p in MARKS.items()})
    return m


def test_shadow_market_publishes_consistent_book():
    m = _market()
    snap = m.store.get("BTC")
    assert snap.is_complete and m.store.connected
    assert snap.book.best_bid < snap.book.mid < snap.book.best_ask
    assert snap.ctx.mark_price == pytest.approx(95_000, rel=1e-6) and np.all(np.diff(snap.book.bids[:, 0]) < 0)


def test_toxic_flow_impact_liquidity_and_recovery():
    m = _market(weight=0.5)
    flow = m.execute_flow("BTC", "SELL", 3_000_000)
    snap = m.store.get("BTC")
    assert flow.impact_bps < -5 and flow.vwap < 95_000 and flow.liquidity_after < 1
    assert snap.ctx.oracle_price == 95_000                       # el oracle no se manipula
    assert snap.ctx.mark_price < 95_000 and snap.ctx.mark_oracle_basis_bps < 0
    spread_hit = snap.book.spread_bps
    for _ in range(40):
        m.step({s: BaseQuote(s, p, p) for s, p in MARKS.items()})
    healed = m.store.get("BTC")
    assert abs(healed.ctx.mark_oracle_basis_bps) < 0.5 and healed.book.spread_bps < spread_hit
    immune = _market(weight=1.0)
    immune.execute_flow("BTC", "SELL", 3_000_000)
    assert immune.store.get("BTC").ctx.mark_price == pytest.approx(95_000)  # mark 100 % oracle


def test_monte_carlo_path_source_tail_selection():
    cfg = ScenarioConfig.black_swan(n_paths=400, n_steps=48, seed=3)
    src = MonteCarloPathSource.from_config(cfg, quantile=0.05)
    rets = src.result.terminal_log_returns()[:, 0]
    assert abs((rets < rets[src.path_index]).mean() - 0.05) < 0.01
    quotes = [src.next_quotes() for _ in range(49)]
    assert all(q is not None for q in quotes) and src.next_quotes() is None
    assert quotes[0]["BTC"].price == pytest.approx(95_000) and 0 < quotes[-1]["ETH"].liquidity <= 1


# ============================================================================= pasarela (PostgreSQL)
def _positions(db, uid):
    with db.transaction(readonly=True) as cur:
        cur.execute("SELECT symbol, quantity, entry_price, status FROM open_positions WHERE user_id = %s ORDER BY position_id", (uid,))
        rows = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT deposited_collateral FROM users_collateral WHERE user_id = %s", (uid,))
        return rows, float(cur.fetchone()["deposited_collateral"])


@pytest.mark.integration
def test_gateway_netting_lifecycle(clean_db):
    market = _market()
    gw = OrderGateway(clean_db, market.store)
    uid = seed_account(clean_db, "w", "WHALE_AGGRESSIVE", 10_000.0)
    fee = lambda q, px: abs(q) * px * 4.5e-4  # noqa: E731

    def trade(q, px):
        (f,) = gw.execute(uid, [OrderIntent("BTC", q, px, False)], leverage=20, price_overrides={"BTC": px})
        return f

    assert trade(0.5, 94_000).status == "FILLED"                       # apertura
    assert trade(0.5, 96_000).position_after == pytest.approx(1.0)     # aumento
    rows, _ = _positions(clean_db, uid)
    assert len(rows) == 1 and rows[0]["entry_price"] == Decimal("95000")
    f = trade(-0.4, 97_000)                                            # reducción: +800 realizado
    assert f.realized_pnl == pytest.approx(800)
    f = trade(-1.6, 93_000)                                            # giro: cierra 0.6 (−1 200) y abre −1.0
    assert f.realized_pnl == pytest.approx(-1_200) and f.position_after == pytest.approx(-1.0)
    rows, _ = _positions(clean_db, uid)
    assert [r["status"] for r in rows] == ["CLOSED", "OPEN"] and rows[1]["entry_price"] == Decimal("93000")
    trade(1.0, 92_000)                                                 # cierre: +1 000
    rows, collateral = _positions(clean_db, uid)
    assert all(r["status"] == "CLOSED" for r in rows)
    fees = fee(0.5, 94_000) + fee(0.5, 96_000) + fee(0.4, 97_000) + fee(1.6, 93_000) + fee(1.0, 92_000)
    assert collateral == pytest.approx(10_000 + 800 - 1_200 + 1_000 - fees, abs=1e-6)


@pytest.mark.integration
def test_gateway_pre_trade_controls(clean_db):
    market = _market()
    gw = OrderGateway(clean_db, market.store)
    uid = seed_account(clean_db, "a", "ARBITRAGEUR_NEUTRAL", 1_000.0)
    (f,) = gw.execute(uid, [OrderIntent("BTC", 1.0, 95_000, False)], leverage=20)    # IM 4 750 > 1 000
    assert f.status == "REJECTED" and "margen inicial" in f.reason
    assert _positions(clean_db, uid)[0] == []
    (g,) = gw.execute(uid, [OrderIntent("BTC", 0.1, 95_000, False)], leverage=20)    # IM 475: OK, precio del L2
    assert g.status == "FILLED" and g.price > market.store.get("BTC").book.mid
    frozen = seed_account(clean_db, "f", "WHALE_AGGRESSIVE", 1e6, status="FROZEN")
    assert gw.execute(frozen, [OrderIntent("BTC", 0.1, 95_000, False)], leverage=20)[0].status == "REJECTED"


@pytest.mark.integration
def test_live_routing_blocked_for_manipulator(clean_db):
    market = _market()
    gw = OrderGateway(clean_db, market.store, client_resolver=lambda acc: pytest.fail("no debe enrutarse"))
    uid = seed_account(clean_db, "m", "MARKET_MANIPULATOR", 1e6)
    (f,) = gw.execute(uid, [OrderIntent("BTC", -1.0, 95_000, False)], leverage=5)
    assert f.status == "REJECTED" and "prohibido" in f.reason


# ============================================================================= orquestador end-to-end
@pytest.mark.integration
def test_orchestrator_black_swan_cascade_hits_insurance_fund(clean_db):
    seed_insurance_fund(clean_db, 100_000)
    agents = [TradingAgent(AgentPersona.whale(directional_bias="LONG")), TradingAgent(AgentPersona.arbitrageur()),
              TradingAgent(AgentPersona.manipulator())]
    cfg = OrchestratorConfig(tick_interval_s=0.05, risk_interval_s=0.05, decision_interval_s=0.4,
                             black_swan_after_s=1.5, stress_paths=300, seed=11, stress_crash_steps=1)  # gap: déficit
    report = Orchestrator(clean_db, config=cfg, agents=agents).run(6.0)
    s = report.summary
    assert [m for _, m in report.mode_changes][:2] == ["NORMAL", "BLACK_SWAN"]
    assert s["liquidations_by_agent"].get("whale_trader") == 1
    assert s["deficit_absorbed_usd"] > 0
    assert s["insurance_fund_end"] == pytest.approx(100_000 - s["deficit_absorbed_usd"] + s["fees_collected_usd"], abs=0.01)
    assert s["loop_errors"] == 0 and not report.timeline.empty
    liq = report.liquidations.iloc[0]
    assert liq["mode"] == "BLACK_SWAN" and liq["equity_before"] < liq["mm_before"]
    bursts = report.actions[(report.actions.agent == "market_manipulator") & (report.actions.bursts > 0)]
    assert not bursts.empty and (bursts["mode"] == "BLACK_SWAN").all()
    with clean_db.transaction(readonly=True) as cur:
        cur.execute("SELECT count(*) AS n FROM liquidation_events WHERE market_regime = 'STRESS_MONTE_CARLO'")
        assert cur.fetchone()["n"] >= 1


# ============================================================================= reconvergencia post-estrés
class _ConstantSource:
    name = "live:test"

    def __init__(self, prices):
        self.prices = prices

    def next_quotes(self):
        return {s: BaseQuote(s, p, p) for s, p in self.prices.items()}


def test_reconverging_source_removes_the_gap():
    """Tras un crash simulado (BTC 74 000) el feed real está en 95 000: sin gap, convergencia geométrica."""
    src = ReconvergingSource(_ConstantSource({"BTC": 95_000.0, "ETH": 3_500.0}), {"BTC": 74_000.0, "ETH": 2_700.0},
                             half_life_ticks=12)
    path = [src.next_quotes()["BTC"].price for _ in range(200)]
    assert path[0] == pytest.approx(74_000)                          # primer tick: continuidad exacta
    assert path[12] == pytest.approx(95_000 * (74_000 / 95_000) ** 0.5, rel=1e-9)  # semivida en log-precio
    steps = np.diff(np.log(path))
    assert np.all(steps >= 0) and np.all(steps[:100] > 0)           # monótono hasta converger
    assert steps.max() < 0.025                                      # sin saltos (> 2.5 %/tick)
    assert src.converged and path[-1] == 95_000.0                    # transparente al converger
    eth = src.next_quotes()["ETH"]
    assert eth.oracle == eth.price                                   # oracle desplazado igual que el precio


def test_reconverging_source_scales_live_book():
    store = make_store({"BTC": {"mark": 100.0, "bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)]}})
    from ai_agents.shadow_market import LiveFeedSource

    q = ReconvergingSource(LiveFeedSource(store), {"BTC": 80.0}).next_quotes()["BTC"]
    assert q.price == pytest.approx(80.0) and q.book.best_bid == pytest.approx(99.0 * 0.8)
    assert not q.book.bids.flags.writeable


def test_orchestrator_returns_to_live_without_gap():
    """Regresión del incidente −$334k: fin de Cisne Negro con --live ya no salta al precio real."""
    store = make_store({"BTC": {"mark": 95_000.0}, "ETH": {"mark": 3_500.0}})
    orch = Orchestrator(None, config=OrchestratorConfig(control_interval_s=None), agents=[], live_store=store)
    orch._history["BTC"].append(74_000.0)
    orch._history["ETH"].append(2_700.0)
    orch._mode = MarketMode.BLACK_SWAN
    orch._switch_mode(MarketMode.NORMAL)
    assert isinstance(orch._source, ReconvergingSource)
    assert orch._source.next_quotes()["BTC"].price == pytest.approx(74_000)
    fresh = Orchestrator(None, config=OrchestratorConfig(control_interval_s=None), agents=[], live_store=store)
    fresh._switch_mode(MarketMode.NORMAL)                            # arranque: sin estrés previo => live directo
    assert fresh._source.name == "live:hyperliquid"


# ============================================================================= circuit breaker LLM
class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_circuit_breaker_state_machine():
    clock = _Clock()
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30, clock=clock)
    assert cb.allow() and not cb.record_failure() and cb.state == "CLOSED"
    assert cb.record_failure() and cb.state == "OPEN" and not cb.allow()      # 2º fallo consecutivo: abre
    clock.t = 31
    assert cb.state == "HALF_OPEN" and cb.allow() and not cb.allow()          # una única sonda
    cb.record_failure()
    assert cb.state == "OPEN"                                                 # la sonda falla: reabre
    clock.t = 62
    assert cb.allow()
    cb.record_success()
    assert cb.state == "CLOSED" and cb.allow()


def test_open_circuit_skips_the_llm_entirely():
    calls = {"n": 0}

    def failing_chain(schema):
        def boom(_):
            calls["n"] += 1
            raise TimeoutError("openai down")
        return RunnableLambda(boom)

    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=60)
    agent = TradingAgent(AgentPersona.whale(), engine=LLMDecisionEngine(chain_factory=failing_chain), breaker=breaker)
    obs = MarketObservation(MarketMode.NORMAL, {"BTC": 95_000.0, "ETH": 3_500.0}, {"BTC": 95_000.0, "ETH": 3_500.0},
                            {"BTC": 0.0, "ETH": 0.0}, {"BTC": 0.5, "ETH": 0.6}, {"BTC": 1.0, "ETH": 1.0},
                            returns_short={"BTC": 0.001, "ETH": 0.0})
    account = AccountView(1, "whale_trader", 10_000.0, 10_000.0, 0.0, math.inf, 10_000.0, {})
    sources = [agent.act(obs, account).source for _ in range(5)]
    assert sources == ["RULES_FALLBACK", "RULES_FALLBACK"] + ["RULES_CIRCUIT_OPEN"] * 3
    assert calls["n"] == 2                                                    # sin más timeouts tras abrir


def test_burst_stops_pushing_price_after_gateway_reject():
    """Una fracción rechazada (margen/límite) aborta la ráfaga: sin flujo fantasma contra el libro."""
    from types import SimpleNamespace

    from ai_agents.order_gateway import Fill

    flows = []
    orch = Orchestrator(None, config=OrchestratorConfig(control_interval_s=None), agents=[])
    orch.market = SimpleNamespace(execute_flow=lambda sym, side, usd: flows.append(usd) or SimpleNamespace(vwap=100.0, filled_quantity=usd / 100))
    statuses = iter(["FILLED", "FILLED", "REJECTED", "FILLED"])
    orch.gateway = SimpleNamespace(headroom_usd=lambda uid: 1e12, execute=lambda uid, intents, **kw: (
        Fill(intents[0].symbol, intents[0].delta_quantity, 0.0, 100.0, 0.0, 0.0, 0.0, next(statuses), "límite"),))
    agent = TradingAgent(AgentPersona.manipulator())
    burst = StressBurst(symbol="BTC", side="SELL", notional_usd=1_000_000, slices=10)
    orch._execute(agent, 1, AgentAction(agent.name, agent.persona.profile, "RULES", "t", bursts=(burst,)))
    assert len(flows) == 3                                                    # 2 OK + la rechazada; no 10
    assert orch._actions[-1]["rejected"] == 1
