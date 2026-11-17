"""
ai_agents/trader_agents.py
==========================

Agentes autónomos de trading con razonamiento LLM (LangChain + OpenAI) y guardarraíles deterministas.

Arquitectura de decisión (patrón "LLM propone, la política de riesgo dispone")
-----------------------------------------------------------------------------
::

    observación (mercado sombra + cuenta + memoria)
          │
          ▼
    DecisionEngine ──► LLMDecisionEngine   prompt | ChatOpenAI.with_structured_output(Pydantic)
          │              (si falla / sin API key)
          └──────────► RuleBasedEngine     política cuantitativa determinista por perfil
          │
          ▼
    Guardarraíles (código, no prompt): símbolos permitidos, apalancamiento máximo, neutralidad,
    tamaño máximo de orden, lot size, umbral anti-churning, cuentas insolventes
          │
          ▼
    OrderIntent / StressBurst  ──►  OrderGateway / ShadowMarket   (orquestador)

El LLM nunca ejecuta nada directamente: solo produce un objeto Pydantic validado que la capa de
riesgo puede recortar o descartar. Un fallo del LLM (timeout, rate limit, JSON inválido) degrada a
la política determinista sin detener la simulación.

Perfiles
--------
* ``WHALE_AGGRESSIVE``     direccional, 20x, colateral ajustado. Objetivo principal del motor de riesgo.
* ``ARBITRAGEUR_NEUTRAL``  largo BTC / corto ETH dollar-neutral, 2-3x en margen cruzado.
* ``MARKET_MANIPULATOR``   generador de estrés: ráfagas de órdenes agresivas contra el libro SOMBRA
                           (solo en modo Cisne Negro; nunca se enruta a un exchange real).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol

import numpy as np
import structlog
from pydantic import BaseModel, Field

from risk_engine.risk_limits import RiskLimits

if TYPE_CHECKING:
    from risk_engine.margin_calculator import MarginSnapshot

__all__ = [
    "AgentProfile",
    "MarketMode",
    "TargetExposure",
    "TradeDecision",
    "StressBurst",
    "ManipulationPlan",
    "MarketObservation",
    "AccountView",
    "AgentPersona",
    "OrderIntent",
    "AgentAction",
    "AgentMemory",
    "DecisionEngine",
    "RuleBasedEngine",
    "LLMDecisionEngine",
    "CircuitBreaker",
    "TradingAgent",
    "default_agents",
    "LOT_DECIMALS",
]

log = structlog.get_logger(__name__)

Symbol = Literal["BTC", "ETH"]
LOT_DECIMALS: dict[str, int] = {"BTC": 5, "ETH": 4}  # szDecimals de Hyperliquid


class AgentProfile(str, Enum):  # = ENUM agent_profile del esquema
    WHALE_AGGRESSIVE = "WHALE_AGGRESSIVE"
    ARBITRAGEUR_NEUTRAL = "ARBITRAGEUR_NEUTRAL"
    MARKET_MANIPULATOR = "MARKET_MANIPULATOR"


class MarketMode(str, Enum):
    NORMAL = "NORMAL"
    BLACK_SWAN = "BLACK_SWAN"


# =============================================================================
# Esquemas de salida estructurada del LLM (validados por Pydantic)
# =============================================================================
class TargetExposure(BaseModel):
    symbol: Symbol
    target_notional_usd: float = Field(description="Exposición objetivo con signo en USD: + largo, − corto, 0 plano.")


class TradeDecision(BaseModel):
    rationale: str = Field(max_length=800, description="Justificación breve (máx. 2-3 frases).")
    targets: list[TargetExposure] = Field(default_factory=list, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)


class StressBurst(BaseModel):
    symbol: Symbol
    side: Literal["SELL", "BUY"]
    notional_usd: float = Field(gt=0)
    slices: int = Field(ge=1, le=50, description="Nº de órdenes en que se fragmenta la ráfaga.")


class ManipulationPlan(BaseModel):
    rationale: str = Field(max_length=800)
    bursts: list[StressBurst] = Field(default_factory=list, max_length=4)


Decision = TradeDecision | ManipulationPlan


# =============================================================================
# Observaciones
# =============================================================================
@dataclass(frozen=True)
class MarketObservation:
    mode: MarketMode
    marks: Mapping[str, float]
    oracles: Mapping[str, float]
    funding_hourly: Mapping[str, float]
    spread_bps: Mapping[str, float]
    liquidity: Mapping[str, float]
    returns_short: Mapping[str, float] = field(default_factory=dict)   # log-retorno ventana corta
    realized_vol: Mapping[str, float] = field(default_factory=dict)    # desviación por tick
    timestamp: float = field(default_factory=time.time)

    def prompt_dict(self) -> dict[str, Any]:
        r = lambda d, n=6: {k: round(float(v), n) for k, v in d.items()}  # noqa: E731
        return {
            "mode": self.mode.value, "marks": r(self.marks, 2), "oracles": r(self.oracles, 2),
            "funding_hourly": r(self.funding_hourly, 8), "spread_bps": r(self.spread_bps, 2),
            "liquidity_factor": r(self.liquidity, 3), "return_short_window": r(self.returns_short),
            "realized_vol_per_tick": r(self.realized_vol),
        }


@dataclass(frozen=True)
class AccountView:
    user_id: int
    agent_name: str
    collateral: float
    equity: float
    maintenance_margin: float
    health_ratio: float
    margin_buffer: float
    positions: Mapping[str, tuple[float, float, float]]   # symbol -> (quantity, entry, upnl)
    is_liquidatable: bool = False

    @classmethod
    def from_snapshot(cls, snap: "MarginSnapshot", user_id: int) -> "AccountView":
        a = snap.account(user_id)
        pos = snap.positions[snap.positions["user_id"] == user_id]
        return cls(
            user_id=user_id,
            agent_name=str(a.get("agent_name", user_id)),
            collateral=float(a["collateral"]),
            equity=float(a["equity"]),
            maintenance_margin=float(a["maintenance_margin"]),
            health_ratio=float(a["health_ratio"]),
            margin_buffer=float(a["margin_buffer"]),
            positions={r.symbol: (float(r.quantity), float(r.entry_price), float(r.unrealized_pnl))
                       for r in pos.itertuples()},
            is_liquidatable=bool(a["is_liquidatable"]),
        )

    def quantity(self, symbol: str) -> float:
        return self.positions.get(symbol, (0.0, 0.0, 0.0))[0]

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "equity_usd": round(self.equity, 2), "collateral_usd": round(self.collateral, 2),
            "maintenance_margin_usd": round(self.maintenance_margin, 2),
            "health_ratio": None if not math.isfinite(self.health_ratio) else round(self.health_ratio, 4),
            "positions": {s: {"quantity": q, "entry": round(e, 2), "unrealized_pnl": round(u, 2)}
                          for s, (q, e, u) in self.positions.items()},
        }


# =============================================================================
# Personas
# =============================================================================
_COMMON_RULES = (
    "Opera solo perpetuos de BTC y ETH en margen cruzado. Expresa tu decisión como exposición OBJETIVO "
    "con signo en USD por activo (no órdenes). La capa de riesgo recortará cualquier exceso, así que "
    "propón objetivos coherentes con tus límites. Responde con el esquema estructurado solicitado."
)


@dataclass(frozen=True)
class AgentPersona:
    name: str
    profile: AgentProfile
    initial_collateral: float
    max_leverage: float                 # límite duro (guardarraíl)
    target_leverage: float              # apalancamiento operativo deseado
    symbols: tuple[str, ...]
    max_order_notional_usd: float       # tamaño máximo por orden (las metas se alcanzan en varias rondas)
    max_gross_notional_usd: float | None = None  # tope absoluto de nocional bruto (también en users_collateral)
    rebalance_band: float = 0.05        # no operar si |Δ| < banda × (target_leverage × equity)
    neutrality_tolerance: float = 1.0   # |neto| / bruto permitido (1.0 = sin restricción)
    directional_bias: Literal["MOMENTUM", "LONG", "SHORT"] = "MOMENTUM"
    burst_notional_usd: float = 0.0     # solo manipulador
    burst_slices: int = 10
    burst_every_n_decisions: int = 3    # cadencia: una ráfaga cada N decisiones (solo manipulador)
    stress_only: bool = False           # solo actúa en modo Cisne Negro
    system_prompt: str = ""

    @property
    def limits(self) -> RiskLimits:
        return RiskLimits(self.max_leverage, self.max_gross_notional_usd)

    @classmethod
    def whale(cls, **kw: Any) -> "AgentPersona":
        base = cls(
            name="whale_trader", profile=AgentProfile.WHALE_AGGRESSIVE, initial_collateral=10_000.0,
            max_leverage=20.0, target_leverage=19.0, symbols=("BTC", "ETH"), max_order_notional_usd=250_000.0,
            max_gross_notional_usd=250_000.0,
            system_prompt=(
                "Eres 'Whale', un trader direccional agresivo de perpetuos cripto. Buscas rentabilidad máxima "
                "con apalancamiento cercano a 20x y colateral ajustado; aceptas un riesgo de liquidación elevado. "
                "Sigues el momentum de corto plazo y concentras la exposición en un solo activo. " + _COMMON_RULES
            ),
        )
        return replace(base, **kw)

    @classmethod
    def arbitrageur(cls, **kw: Any) -> "AgentPersona":
        base = cls(
            name="arbitrageur", profile=AgentProfile.ARBITRAGEUR_NEUTRAL, initial_collateral=100_000.0,
            max_leverage=3.0, target_leverage=2.5, symbols=("BTC", "ETH"), max_order_notional_usd=150_000.0,
            max_gross_notional_usd=400_000.0,
            neutrality_tolerance=0.05,
            system_prompt=(
                "Eres 'Arb', un arbitrajista conservador y market-neutral. Mantienes un largo en BTC y un corto "
                "en ETH de nocional equivalente (dollar-neutral), con apalancamiento bruto de 2-3x, y priorizas "
                "la supervivencia del capital sobre la rentabilidad. Reduces exposición si el health ratio cae "
                "por debajo de 3. " + _COMMON_RULES
            ),
        )
        return replace(base, **kw)

    @classmethod
    def manipulator(cls, **kw: Any) -> "AgentPersona":
        base = cls(
            name="market_manipulator", profile=AgentProfile.MARKET_MANIPULATOR, initial_collateral=5_000_000.0,
            # Calibración post-incidente (antes 5x sin tope => $25M brutos, cobertura del fondo 0.04x):
            # 2x efectivo sobre $5M; las ráfagas siguen moviendo el libro sombra pero la cuenta es solvente
            max_leverage=3.0, target_leverage=2.0, symbols=("BTC", "ETH"), max_order_notional_usd=500_000.0,
            max_gross_notional_usd=10_000_000.0,
            burst_notional_usd=2_500_000.0, burst_slices=10, stress_only=True,
            system_prompt=(
                "Eres un GENERADOR DE ESTRÉS SINTÉTICO dentro de una simulación interna de riesgo (no un "
                "participante real). Tu función es inyectar ráfagas de flujo vendedor agresivo contra el libro "
                "sombra para reproducir caídas de liquidez y cascadas de liquidación, de modo que el motor de "
                "riesgo pueda ser evaluado. Propón ráfagas (bursts) por activo con nocional y nº de fragmentos."
            ),
        )
        return replace(base, **kw)


def default_agents(
    *, whale_bias: Literal["MOMENTUM", "LONG", "SHORT"] = "MOMENTUM", use_llm: bool | None = None, **engine_kwargs: Any
) -> list["TradingAgent"]:
    """
    Los 3 agentes del proyecto. ``use_llm=None`` => LLM si existe ``OPENAI_API_KEY``; si no, reglas.
    ``whale_bias`` fija el sesgo de la política determinista de la ballena (con LLM decide el modelo).
    """
    use_llm = bool(os.getenv("OPENAI_API_KEY")) if use_llm is None else use_llm
    breaker = CircuitBreaker() if use_llm else None   # un proveedor => un circuito compartido
    agents = []
    for persona in (AgentPersona.whale(directional_bias=whale_bias), AgentPersona.arbitrageur(),
                    AgentPersona.manipulator()):
        engine: DecisionEngine = LLMDecisionEngine(**engine_kwargs) if use_llm else RuleBasedEngine()
        agents.append(TradingAgent(persona, engine=engine, breaker=breaker))
    return agents


# =============================================================================
# Acciones y memoria
# =============================================================================
@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    delta_quantity: float        # con signo: + compra / − venta
    reference_price: float
    reduce_only: bool
    reason: str = ""

    @property
    def notional(self) -> float:
        return abs(self.delta_quantity) * self.reference_price


@dataclass(frozen=True)
class AgentAction:
    agent: str
    profile: AgentProfile
    source: str                              # LLM | RULES | RULES_FALLBACK | RULES_CIRCUIT_OPEN | SKIPPED
    rationale: str
    intents: tuple[OrderIntent, ...] = ()
    bursts: tuple[StressBurst, ...] = ()
    guardrail_notes: tuple[str, ...] = ()
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class AgentMemory:
    """Memoria episódica acotada: decisiones recientes y su resultado, inyectada en el prompt."""

    def __init__(self, maxlen: int = 8) -> None:
        self._items: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def remember(self, action: AgentAction, account: AccountView) -> None:
        self._items.append({
            "t": round(action.timestamp, 1), "source": action.source, "rationale": action.rationale[:160],
            "orders": [(i.symbol, round(i.delta_quantity, 6)) for i in action.intents],
            "bursts": [(b.symbol, b.side, round(b.notional_usd)) for b in action.bursts],
            "equity_after": round(account.equity, 2), "health_ratio": None if not math.isfinite(account.health_ratio)
            else round(account.health_ratio, 3),
        })

    def render(self) -> list[dict[str, Any]]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


# =============================================================================
# Motores de decisión
# =============================================================================
class DecisionEngine(Protocol):
    name: str

    def decide(self, persona: AgentPersona, obs: MarketObservation, account: AccountView,
               memory: AgentMemory) -> Decision: ...


class RuleBasedEngine:
    """Política cuantitativa determinista por perfil (fallback del LLM y modo sin API key)."""

    name = "RULES"

    def __init__(self) -> None:
        self._decisions: dict[str, int] = {}

    def decide(self, persona: AgentPersona, obs: MarketObservation, account: AccountView,
               memory: AgentMemory) -> Decision:
        equity = max(account.equity, 0.0)
        if persona.profile is AgentProfile.MARKET_MANIPULATOR:
            if obs.mode is not MarketMode.BLACK_SWAN:
                return ManipulationPlan(rationale="Modo normal: sin inyección de estrés.")
            n = self._decisions.get(persona.name, 0)
            self._decisions[persona.name] = n + 1
            if n % max(1, persona.burst_every_n_decisions):
                return ManipulationPlan(rationale="Cadencia: dejo que la cascada actúe antes de la siguiente ráfaga.")
            return ManipulationPlan(
                rationale="Cisne negro: ráfagas vendedoras para drenar bids y forzar cascadas.",
                bursts=[
                    StressBurst(symbol="BTC", side="SELL", notional_usd=persona.burst_notional_usd,
                                slices=persona.burst_slices),
                    StressBurst(symbol="ETH", side="SELL", notional_usd=persona.burst_notional_usd / 2,
                                slices=persona.burst_slices),
                ],
            )

        gross = persona.target_leverage * equity
        if persona.profile is AgentProfile.ARBITRAGEUR_NEUTRAL:
            de_risk = 0.5 if math.isfinite(account.health_ratio) and account.health_ratio < 3 else 1.0
            leg = gross * de_risk / 2
            return TradeDecision(
                rationale=f"Dollar-neutral: +{leg:,.0f} BTC / −{leg:,.0f} ETH (factor de-risk {de_risk}).",
                targets=[TargetExposure(symbol="BTC", target_notional_usd=leg),
                         TargetExposure(symbol="ETH", target_notional_usd=-leg)],
                confidence=0.8,
            )

        # WHALE: direccional concentrado en BTC
        if persona.directional_bias == "LONG":
            sign = 1.0
        elif persona.directional_bias == "SHORT":
            sign = -1.0
        else:
            sign = 1.0 if obs.returns_short.get("BTC", 0.0) >= 0 else -1.0
        return TradeDecision(
            rationale=f"Momentum BTC {obs.returns_short.get('BTC', 0.0):+.4%}: exposición {sign * gross:,.0f} USD.",
            targets=[TargetExposure(symbol="BTC", target_notional_usd=sign * gross),
                     TargetExposure(symbol="ETH", target_notional_usd=0.0)],
            confidence=0.6,
        )


class LLMDecisionEngine:
    """
    Decisión mediante LLM con salida estructurada (Pydantic) vía LangChain.

    ``chain_factory`` permite inyectar cualquier ``Runnable`` (tests, otros proveedores): recibe el
    esquema de salida y devuelve un runnable que, invocado con ``{"system", "context"}``, produce una
    instancia de ese esquema.
    """

    name = "LLM"

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        timeout_s: float = 20.0,
        max_retries: int = 1,
        chain_factory: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._chain_factory = chain_factory
        self._chains: dict[type, Any] = {}

    def _chain(self, schema: type[BaseModel]) -> Any:
        if schema not in self._chains:
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system}"),
                ("human", "Estado actual (JSON):\n{context}\n\nDevuelve tu decisión para esta ronda."),
            ])
            if self._chain_factory is not None:
                llm = self._chain_factory(schema)
            else:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(model=self.model, temperature=self.temperature, timeout=self.timeout_s,
                                 max_retries=self.max_retries).with_structured_output(schema)
            self._chains[schema] = prompt | llm
        return self._chains[schema]

    def decide(self, persona: AgentPersona, obs: MarketObservation, account: AccountView,
               memory: AgentMemory) -> Decision:
        schema = ManipulationPlan if persona.profile is AgentProfile.MARKET_MANIPULATOR else TradeDecision
        limits = {
            "max_leverage": persona.max_leverage, "target_leverage": persona.target_leverage,
            "max_order_notional_usd": persona.max_order_notional_usd,
            "max_gross_notional_usd": persona.max_gross_notional_usd, "symbols": list(persona.symbols),
            "neutrality_tolerance": persona.neutrality_tolerance,
        }
        context = json.dumps({
            "agent": persona.name, "limits": limits, "market": obs.prompt_dict(),
            "account": account.prompt_dict(), "memory": memory.render(),
        }, ensure_ascii=False)
        result = self._chain(schema).invoke({"system": persona.system_prompt, "context": context})
        if not isinstance(result, schema):
            result = schema.model_validate(result)
        return result


class CircuitBreaker:
    """
    Corta llamadas a un proveedor externo que está fallando (OpenAI caído, cuota agotada, 401...).

    ``CLOSED`` → tras ``failure_threshold`` fallos consecutivos pasa a ``OPEN`` durante ``cooldown_s``
    (las decisiones van directas a la política de reglas, sin esperar ``timeout × reintentos`` por agente
    y ronda) → ``HALF_OPEN``: deja pasar UNA llamada de prueba; éxito => ``CLOSED``, fallo => ``OPEN``.
    Compartido entre agentes (un único proveedor) y thread-safe: el orquestador decide en paralelo.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 60.0, *, clock: Any = time.monotonic) -> None:
        if failure_threshold < 1 or cooldown_s <= 0:
            raise ValueError("failure_threshold >= 1 y cooldown_s > 0")
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state()

    def _state(self) -> str:
        if self._opened_at is None:
            return "CLOSED"
        return "OPEN" if self._clock() - self._opened_at < self.cooldown_s else "HALF_OPEN"

    def allow(self) -> bool:
        with self._lock:
            state = self._state()
            if state == "CLOSED":
                return True
            if state == "HALF_OPEN" and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures, self._opened_at, self._probe_in_flight = 0, None, False

    def record_failure(self) -> bool:
        """Registra un fallo. Devuelve ``True`` si el circuito acaba de abrirse."""
        with self._lock:
            self._failures += 1
            was_open = self._opened_at is not None
            if self._probe_in_flight or self._failures >= self.failure_threshold:
                self._opened_at, self._probe_in_flight = self._clock(), False
                return not was_open
            return False


# =============================================================================
# Agente
# =============================================================================
class TradingAgent:
    """Agente autónomo: decide (LLM o reglas), aplica guardarraíles y recuerda el resultado."""

    def __init__(self, persona: AgentPersona, *, engine: DecisionEngine | None = None,
                 fallback: DecisionEngine | None = None, memory_size: int = 8,
                 breaker: CircuitBreaker | None = None) -> None:
        self.persona = persona
        self.engine = engine or RuleBasedEngine()
        self.fallback = fallback or RuleBasedEngine()
        self.breaker = breaker
        self.memory = AgentMemory(memory_size)
        self._log = log.bind(agent=persona.name, profile=persona.profile.value)

    @property
    def name(self) -> str:
        return self.persona.name

    def act(self, obs: MarketObservation, account: AccountView) -> AgentAction:
        t0 = time.perf_counter()
        p = self.persona
        if account.is_liquidatable or not math.isfinite(account.equity) or account.equity <= 0:
            return AgentAction(p.name, p.profile, "SKIPPED", "Cuenta insolvente/liquidable: decide el motor de riesgo.")
        if p.stress_only and obs.mode is not MarketMode.BLACK_SWAN:
            return AgentAction(p.name, p.profile, "SKIPPED", "Generador de estrés inactivo en modo normal.")

        source = self.engine.name
        if self.breaker is not None and not self.breaker.allow():
            decision = self.fallback.decide(p, obs, account, self.memory)
            source = "RULES_CIRCUIT_OPEN"
        else:
            try:
                decision = self.engine.decide(p, obs, account, self.memory)
                if self.breaker is not None:
                    self.breaker.record_success()
            except Exception as exc:  # noqa: BLE001 - cualquier fallo del LLM degrada a reglas
                self._log.warning("agent.llm_fallback", error=f"{type(exc).__name__}: {str(exc)[:200]}")
                if self.breaker is not None and self.breaker.record_failure():
                    self._log.error("agent.llm_circuit_open", cooldown_s=self.breaker.cooldown_s,
                                    failures=self.breaker.failure_threshold)
                decision = self.fallback.decide(p, obs, account, self.memory)
                source = "RULES_FALLBACK"

        notes: list[str] = []
        if isinstance(decision, ManipulationPlan):
            bursts = self._guard_bursts(decision, obs, account, notes)
            action = AgentAction(p.name, p.profile, source, decision.rationale, bursts=bursts,
                                 guardrail_notes=tuple(notes), latency_ms=(time.perf_counter() - t0) * 1e3)
        else:
            intents = self._guard_targets(decision, obs, account, notes)
            action = AgentAction(p.name, p.profile, source, decision.rationale, intents=intents,
                                 guardrail_notes=tuple(notes), latency_ms=(time.perf_counter() - t0) * 1e3)
        self._log.info("agent.decision", source=source, intents=len(action.intents), bursts=len(action.bursts),
                       notes=list(notes), latency_ms=round(action.latency_ms, 1), rationale=action.rationale[:160])
        return action

    def record_outcome(self, action: AgentAction, account_after: AccountView) -> None:
        self.memory.remember(action, account_after)

    # ------------------------------------------------------------ guardarraíles
    def _guard_targets(self, decision: TradeDecision, obs: MarketObservation, account: AccountView,
                       notes: list[str]) -> tuple[OrderIntent, ...]:
        p = self.persona
        targets: dict[str, float] = {}
        for t in decision.targets:
            if t.symbol not in p.symbols:
                notes.append(f"símbolo {t.symbol} no permitido")
                continue
            if t.symbol not in obs.marks or not math.isfinite(t.target_notional_usd):
                notes.append(f"sin precio para {t.symbol}")
                continue
            targets[t.symbol] = float(t.target_notional_usd)

        gross_cap = p.limits.gross_cap(account.equity)
        gross = sum(abs(v) for v in targets.values())
        if gross > gross_cap > 0:
            k = gross_cap / gross
            targets = {s: v * k for s, v in targets.items()}
            notes.append(f"exposición {gross:,.0f} USD ({gross / account.equity:.1f}x) recortada a {gross_cap:,.0f} USD")

        if p.neutrality_tolerance < 1.0 and targets:
            longs = sum(v for v in targets.values() if v > 0)
            shorts = -sum(v for v in targets.values() if v < 0)
            gross = longs + shorts
            if gross > 0 and abs(longs - shorts) > p.neutrality_tolerance * gross:
                m = min(longs, shorts)
                targets = {s: (v / longs * m if v > 0 else v / shorts * m) if v else 0.0 for s, v in targets.items()}
                notes.append(f"neutralidad forzada: legs igualadas a {m:,.0f} USD")

        band = p.rebalance_band * p.target_leverage * account.equity
        intents: list[OrderIntent] = []
        for sym in sorted(targets):
            mark = float(obs.marks[sym])
            current = account.quantity(sym)
            delta_notional = targets[sym] - current * mark
            if abs(delta_notional) < max(band, 10.0):
                continue
            if abs(delta_notional) > p.max_order_notional_usd:
                delta_notional = math.copysign(p.max_order_notional_usd, delta_notional)
                notes.append(f"{sym}: orden fraccionada a {p.max_order_notional_usd:,.0f} USD")
            dq = _round_lot(sym, delta_notional / mark)
            if dq == 0:
                continue
            reduce_only = current != 0 and np.sign(dq) != np.sign(current) and abs(dq) <= abs(current)
            intents.append(OrderIntent(sym, dq, mark, bool(reduce_only), decision.rationale[:120]))
        return tuple(intents)

    def _guard_bursts(self, plan: ManipulationPlan, obs: MarketObservation, account: AccountView,
                      notes: list[str]) -> tuple[StressBurst, ...]:
        p = self.persona
        current_gross = sum(abs(q) * obs.marks.get(s, 0.0) for s, (q, _, _) in account.positions.items())
        room = p.limits.headroom(current_gross, account.equity)
        bursts: list[StressBurst] = []
        for b in plan.bursts:
            if b.symbol not in p.symbols or b.symbol not in obs.marks:
                notes.append(f"ráfaga {b.symbol} descartada")
                continue
            notional = min(b.notional_usd, room, p.max_order_notional_usd * b.slices)
            if notional < 1_000:
                notes.append(f"ráfaga {b.symbol} sin margen disponible")
                continue
            if notional < b.notional_usd:
                notes.append(f"ráfaga {b.symbol} recortada a {notional:,.0f} USD")
            room -= notional
            bursts.append(b.model_copy(update={"notional_usd": notional}))
        return tuple(bursts)


def _round_lot(symbol: str, qty: float) -> float:
    """Trunca al lot size del exchange (hacia cero: nunca más tamaño del pedido)."""
    d = LOT_DECIMALS.get(symbol, 4)
    return math.trunc(qty * 10**d) / 10**d
