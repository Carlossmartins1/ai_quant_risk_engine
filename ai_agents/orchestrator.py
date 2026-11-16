"""
ai_agents/orchestrator.py
=========================

Orquestador de simulación: conecta mercado (Fases 2-3), agentes (Fase 5) y motor de riesgo (Fase 4)
en tres bucles concurrentes que comparten el mismo ``ShadowMarket`` y la misma base de datos.

::

    ┌─────────────── hilo "market" (cada tick_interval_s) ───────────────┐
    │ PriceSource ── NORMAL: feed Hyperliquid en vivo | Monte Carlo normal │
    │             └─ BLACK_SWAN: trayectoria de cola del Monte Carlo        │
    │                 (black_swan preset arrancado en los marks actuales)  │
    │ ShadowMarket.step()  -> MarketDataStore sombra (L2 + mark + oracle)   │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │ (mismo store)
    ┌──────────── hilo "agents" ────┴──────┐    ┌──────────── hilo "risk" ─────────────┐
    │ MarginSnapshot -> TradingAgent.act() │    │ LiquidationExecutor.run_cycle()       │
    │ (LLM en paralelo, ThreadPool)        │    │ NORMAL  -> regime LIVE_TESTNET        │
    │ OrderGateway / ShadowMarket.flow     │    │ BLACK_SWAN -> regime STRESS_MONTE_CARLO│
    └──────────────────────────────────────┘    └───────────────────────────────────────┘

Los agentes y el liquidador compiten por las mismas filas: el protocolo de locks de la Fase 1
(NOWAIT en el liquidador, lock_timeout corto en la pasarela) resuelve la contención sin deadlocks.

Uso::

    python -m ai_agents.orchestrator --duration 90 --black-swan-at 20          # offline (Monte Carlo)
    python -m ai_agents.orchestrator --live --duration 120 --black-swan-at 30  # precios reales Testnet
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import structlog

from ai_agents.control_plane import ControlPlane, SimulationCommand
from ai_agents.order_gateway import GatewayConfig, OrderGateway
from ai_agents.shadow_market import LiveFeedSource, MonteCarloPathSource, PriceSource, ShadowMarket, ShadowMarketConfig
from ai_agents.trader_agents import (
    AccountView,
    AgentAction,
    MarketMode,
    MarketObservation,
    OrderIntent,
    TradingAgent,
    default_agents,
)
from database.connection import DatabaseManager, lock_account
from quant_models.monte_carlo import ScenarioConfig
from risk_engine.liquidation_executor import (
    LiquidationConfig,
    LiquidationExecutor,
    LiquidationOutcome,
    MarketRegime,
)
from risk_engine.margin_calculator import MarginCalculator

__all__ = ["OrchestratorConfig", "SimulationReport", "Orchestrator"]

log = structlog.get_logger(__name__)


# =============================================================================
# Configuración e informe
# =============================================================================
@dataclass(frozen=True)
class OrchestratorConfig:
    tick_interval_s: float = 0.25          # 1 tick = 1 paso Monte Carlo (5 min simulados por defecto)
    risk_interval_s: float = 0.2
    decision_interval_s: float = 1.0
    black_swan_after_s: float | None = None
    return_to_normal: bool = True          # vuelve a NORMAL al agotar la trayectoria de estrés
    stress_paths: int = 2_000
    stress_path_quantile: float = 0.05     # trayectoria del percentil 5 de BTC (escenario de cola)
    stress_vol_multiplier: float = 1.0
    normal_paths: int = 256
    seed: int | None = None
    initial_prices: Mapping[str, float] = field(default_factory=lambda: {"BTC": 95_000.0, "ETH": 3_500.0})
    reset_accounts: bool = True            # sesión limpia: colateral inicial y posiciones cerradas
    revive_liquidated_after_s: float | None = None   # re-capitaliza agentes liquidados (demos largas)
    history_ticks: int = 600
    momentum_window: int = 12
    llm_workers: int = 3
    control_interval_s: float | None = 1.0  # plano de control (dashboard); None = desactivado


@dataclass(frozen=True)
class SimulationReport:
    timeline: pd.DataFrame
    liquidations: pd.DataFrame
    actions: pd.DataFrame
    mode_changes: tuple[tuple[float, str], ...]
    summary: Mapping[str, Any]

    def print(self) -> None:  # pragma: no cover - salida de consola
        pd.set_option("display.width", 200)
        print("\n=== RESUMEN ===")
        for k, v in self.summary.items():
            print(f"  {k:<28} {v}")
        if not self.liquidations.empty:
            print("\n=== LIQUIDACIONES ===")
            print(self.liquidations.to_string(index=False))


# =============================================================================
# Orquestador
# =============================================================================
class Orchestrator:
    def __init__(
        self,
        db: DatabaseManager,
        *,
        config: OrchestratorConfig | None = None,
        agents: Sequence[TradingAgent] | None = None,
        live_store: Any | None = None,
        shadow_config: ShadowMarketConfig | None = None,
        gateway_config: GatewayConfig | None = None,
        liquidation_config: LiquidationConfig | None = None,
        stress_annual_vols: Mapping[str, float] | None = None,
        control_plane: ControlPlane | None = None,
    ) -> None:
        self.db = db
        self.cfg = config or OrchestratorConfig()
        self.agents = list(agents) if agents is not None else default_agents()
        self.market = ShadowMarket(("BTC", "ETH"), shadow_config)
        self.gateway = OrderGateway(db, self.market.store, gateway_config)
        base_liq = liquidation_config or LiquidationConfig()
        self.executors = {
            MarketMode.NORMAL: LiquidationExecutor(db, self.market.store, replace(base_liq, regime=MarketRegime.LIVE_TESTNET)),
            MarketMode.BLACK_SWAN: LiquidationExecutor(db, self.market.store, replace(base_liq, regime=MarketRegime.STRESS_MONTE_CARLO)),
        }
        self.calculator = MarginCalculator(db, self.market.store, max_price_age_s=5.0)
        self.live_source = LiveFeedSource(live_store) if live_store is not None else None
        self.stress_annual_vols = dict(stress_annual_vols or {})
        self.control = (control_plane or ControlPlane(db)) if self.cfg.control_interval_s is not None else None

        self._mode = MarketMode.NORMAL
        self._mode_lock = threading.Lock()
        self._pending_mode: MarketMode | None = None
        self._source: PriceSource | None = None
        self._stop = threading.Event()
        self._started_at = 0.0
        self._seed_counter = 0
        self._history: dict[str, deque[float]] = {s: deque(maxlen=self.cfg.history_ticks) for s in self.market.symbols}
        self._user_ids: dict[str, int] = {}
        self._liquidated_at: dict[int, float] = {}

        self._rec_lock = threading.Lock()
        self._timeline: list[dict[str, Any]] = []
        self._outcomes: list[tuple[float, LiquidationOutcome, str]] = []
        self._actions: list[dict[str, Any]] = []
        self._mode_changes: list[tuple[float, str]] = []
        self._fund_balance = math.nan
        self._fund_min = math.inf
        self._errors = 0

    # ------------------------------------------------------------- control
    @property
    def mode(self) -> MarketMode:
        return self._mode

    def trigger_black_swan(self) -> None:
        """Thread-safe: se aplica en el siguiente tick del hilo de mercado."""
        with self._mode_lock:
            self._pending_mode = MarketMode.BLACK_SWAN

    def set_mode(self, mode: MarketMode) -> None:
        with self._mode_lock:
            self._pending_mode = mode

    def stop(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------- bootstrap
    def bootstrap(self) -> dict[str, int]:
        """Crea (idempotente) las cuentas de los agentes y, si ``reset_accounts``, las deja limpias."""
        with self.db.transaction() as cur:
            for agent in self.agents:
                p = agent.persona
                cur.execute(
                    """INSERT INTO users_collateral (agent_name, agent_profile, deposited_collateral, max_leverage)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (agent_name) DO NOTHING""",
                    (p.name, p.profile.value, p.initial_collateral, p.max_leverage),
                )
                cur.execute("SELECT user_id FROM users_collateral WHERE agent_name = %s", (p.name,))
                uid = cur.fetchone()["user_id"]
                self._user_ids[p.name] = uid
                if self.cfg.reset_accounts:
                    self._reset_account(cur, uid, p.initial_collateral)
        log.info("orchestrator.bootstrap", accounts=self._user_ids)
        return dict(self._user_ids)

    def reset_agents(self) -> None:
        """
        Recapitaliza a todos los agentes en caliente (comando ``RESET_AGENTS`` del dashboard): cierra sus
        posiciones y les devuelve el colateral inicial con la cuenta ACTIVE. El Insurance Fund no se toca.
        Bloquea las cuentas en orden de ``user_id`` (orden canónico) para no interbloquearse con el
        liquidador ni con el gateway.
        """
        with self.db.transaction() as cur:
            for agent in sorted(self.agents, key=lambda a: self._user_ids[a.name]):
                uid = self._user_ids[agent.name]
                lock_account(cur, uid)
                self._reset_account(cur, uid, agent.persona.initial_collateral)
        self._liquidated_at.clear()
        log.warning("orchestrator.agents_reset", accounts=self._user_ids)

    @staticmethod
    def _reset_account(cur: Any, uid: int, collateral: float) -> None:
        cur.execute("""UPDATE open_positions SET status = 'CLOSED', closed_at = now()
                        WHERE user_id = %s AND status = 'OPEN'""", (uid,))
        cur.execute("""UPDATE users_collateral SET deposited_collateral = %s, account_status = 'ACTIVE'
                        WHERE user_id = %s""", (collateral, uid))

    # ------------------------------------------------------------------- run
    def run(self, duration_s: float = math.inf) -> SimulationReport:
        """
        Ejecuta la simulación durante ``duration_s`` segundos de reloj (``math.inf`` = servicio
        continuo hasta ``stop()``/SIGTERM) y devuelve el informe.
        """
        self.bootstrap()
        self._stop.clear()
        self._started_at = time.monotonic()
        self._switch_mode(MarketMode.NORMAL)
        self._market_tick()  # primer estado publicado antes de arrancar riesgo y agentes
        self._fund_balance = self._fetch_fund_balance()
        fund_start = self._fund_balance

        threads = [
            threading.Thread(target=self._loop, args=("market", self._market_tick, self.cfg.tick_interval_s), daemon=True),
            threading.Thread(target=self._loop, args=("risk", self._risk_tick, self.cfg.risk_interval_s), daemon=True),
            threading.Thread(target=self._loop, args=("agents", self._agents_tick, self.cfg.decision_interval_s), daemon=True),
        ]
        if self.control is not None:
            threads.append(threading.Thread(target=self._loop, args=("control", self._control_tick,
                                                                     self.cfg.control_interval_s), daemon=True))
        for t in threads:
            t.start()
        deadline = self._started_at + duration_s
        bs_at = None if self.cfg.black_swan_after_s is None else self._started_at + self.cfg.black_swan_after_s
        while not self._stop.is_set() and time.monotonic() < deadline:
            if bs_at is not None and time.monotonic() >= bs_at:
                self.trigger_black_swan()
                bs_at = None
            self._stop.wait(0.05)
        self._stop.set()
        for t in threads:
            t.join(timeout=10)
        self._risk_tick()  # barrido final: nada liquidable queda sin procesar
        report = self._build_report(fund_start)
        if self.control is not None:
            self.control.mark_offline()
        return report

    def _loop(self, name: str, fn: Any, interval: float) -> None:
        threading.current_thread().name = name
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                fn()
            except Exception:  # noqa: BLE001 - un fallo de un bucle no derriba la simulación
                self._errors += 1
                log.exception("orchestrator.loop_error", loop=name)
            self._stop.wait(max(0.0, interval - (time.monotonic() - t0)))

    def _elapsed(self) -> float:
        return time.monotonic() - self._started_at

    # ---------------------------------------------------------------- market
    def _current_prices(self) -> dict[str, float]:
        prices = {s: h[-1] for s, h in self._history.items() if h}
        return {**self.cfg.initial_prices, **prices}

    def _next_seed(self) -> int | None:
        if self.cfg.seed is None:
            return None
        self._seed_counter += 1
        return self.cfg.seed * 1_000 + self._seed_counter

    def _switch_mode(self, mode: MarketMode) -> None:
        prices = self._current_prices()
        if mode is MarketMode.BLACK_SWAN:
            cfg = ScenarioConfig.black_swan(n_paths=self.cfg.stress_paths, seed=self._next_seed()).with_prices(prices)
            if self.stress_annual_vols:
                cfg = cfg.with_volatility(self.stress_annual_vols, multiplier=self.cfg.stress_vol_multiplier)
            self._source = MonteCarloPathSource.from_config(cfg, quantile=self.cfg.stress_path_quantile)
        elif self.live_source is not None:
            self._source = self.live_source
        else:
            cfg = ScenarioConfig.normal(n_paths=self.cfg.normal_paths, seed=self._next_seed()).with_prices(prices)
            self._source = MonteCarloPathSource.from_config(cfg, quantile=0.5)
        self._mode = mode
        with self._rec_lock:
            self._mode_changes.append((round(self._elapsed(), 3), mode.value))
        log.warning("orchestrator.mode", mode=mode.value, source=self._source.name, prices=prices)

    def _market_tick(self) -> None:
        with self._mode_lock:
            pending, self._pending_mode = self._pending_mode, None
        if pending is not None and pending is not self._mode:
            self._switch_mode(pending)
        assert self._source is not None
        quotes = self._source.next_quotes()
        if quotes is None:  # trayectoria agotada
            nxt = MarketMode.NORMAL if (self._mode is MarketMode.BLACK_SWAN and self.cfg.return_to_normal) else self._mode
            self._switch_mode(nxt)
            quotes = self._source.next_quotes() or {}
        if not quotes:
            return
        self.market.step(quotes)
        state = self.market.state()
        row: dict[str, Any] = {"t": round(self._elapsed(), 3), "mode": self._mode.value, "fund_balance": self._fund_balance}
        for s, st in state.items():
            self._history[s].append(st["mark"])
            row.update({f"{s}_mark": st["mark"], f"{s}_oracle": st["oracle"], f"{s}_liquidity": st["liquidity"],
                        f"{s}_basis_bps": st["basis_bps"]})
        with self._rec_lock:
            row["liquidations"] = len(self._outcomes)
            self._timeline.append(row)

    # --------------------------------------------------------------- control
    def _control_tick(self) -> None:
        """Consume órdenes del dashboard y publica heartbeat + estado de mercado + KPIs."""
        assert self.control is not None
        for cmd in self.control.consume():
            if cmd is SimulationCommand.TRIGGER_BLACK_SWAN:
                self.trigger_black_swan()
            elif cmd is SimulationCommand.SET_NORMAL:
                self.set_mode(MarketMode.NORMAL)
            elif cmd is SimulationCommand.RESET_AGENTS:
                self.reset_agents()
        with self._rec_lock:
            n_liq = len(self._outcomes)
            deficit = float(sum(o.deficit for _, o, _ in self._outcomes))
        self.control.publish(
            self._mode.value,
            self.market.state(),
            {"elapsed_s": round(self._elapsed(), 1), "liquidations": n_liq, "deficit_usd": round(deficit, 2),
             "fund_balance": self._fund_balance, "source": getattr(self._source, "name", None),
             "agents": {name: uid for name, uid in self._user_ids.items()}},
        )

    # ------------------------------------------------------------------ risk
    def _risk_tick(self) -> None:
        mode = self._mode
        report = self.executors[mode].run_cycle()
        if report.liquidated:
            now = self._elapsed()
            with self._rec_lock:
                for o in report.liquidated:
                    self._outcomes.append((now, o, mode.value))
                    self._liquidated_at[o.user_id] = now
            for o in report.liquidated:
                log.warning("orchestrator.liquidation", user_id=o.user_id, mode=mode.value,
                            deficit=str(o.deficit), fee=str(o.liquidation_fee),
                            fund_balance=str(o.insurance_fund_balance))
        self._fund_balance = self._fetch_fund_balance()
        self._fund_min = min(self._fund_min, self._fund_balance)

    def _fetch_fund_balance(self) -> float:
        with self.db.transaction(readonly=True) as cur:
            cur.execute("SELECT balance FROM insurance_fund WHERE fund_id = 1")
            return float(cur.fetchone()["balance"])

    # ---------------------------------------------------------------- agents
    def _observation(self) -> MarketObservation:
        snaps = self.market.store.snapshot()
        marks, oracles, funding, spread, liq, rets, vols = {}, {}, {}, {}, {}, {}, {}
        w = self.cfg.momentum_window
        for s, snap in snaps.items():
            if snap.ctx is None:
                continue
            marks[s], oracles[s], funding[s] = snap.ctx.mark_price, snap.ctx.oracle_price, snap.ctx.funding_rate_hourly
            spread[s] = snap.book.spread_bps if snap.book and snap.book.spread_bps is not None else math.nan
            liq[s] = self.market.liquidity(s)
            h = np.asarray(self._history[s], dtype=np.float64)
            if len(h) > w:
                rets[s] = float(np.log(h[-1] / h[-1 - w]))
                vols[s] = float(np.std(np.diff(np.log(h[-min(len(h), 120):]))))
        return MarketObservation(self._mode, marks, oracles, funding, spread, liq, rets, vols)

    def _revive(self, now: float) -> None:
        delay = self.cfg.revive_liquidated_after_s
        if delay is None:
            return
        for agent in self.agents:
            uid = self._user_ids[agent.name]
            t_liq = self._liquidated_at.get(uid)
            if t_liq is not None and now - t_liq >= delay:
                with self.db.transaction() as cur:
                    cur.execute("""UPDATE users_collateral SET deposited_collateral = %s, account_status = 'ACTIVE'
                                    WHERE user_id = %s AND account_status = 'LIQUIDATED'""",
                                (agent.persona.initial_collateral, uid))
                self._liquidated_at.pop(uid, None)
                log.info("orchestrator.agent_revived", agent=agent.name)

    def _agents_tick(self) -> None:
        self._revive(self._elapsed())
        snap = self.calculator.snapshot()
        obs = self._observation()
        if not obs.marks:
            return
        tasks = []
        for agent in self.agents:
            uid = self._user_ids[agent.name]
            if uid not in snap.accounts.index:  # LIQUIDATED / FROZEN: fuera del universo monitorizado
                continue
            tasks.append((agent, uid, AccountView.from_snapshot(snap, uid)))
        with ThreadPoolExecutor(max_workers=self.cfg.llm_workers, thread_name_prefix="agent") as pool:
            futures = [(a, uid, view, pool.submit(a.act, obs, view)) for a, uid, view in tasks]
            actions = [(a, uid, view, f.result()) for a, uid, view, f in futures]
        for agent, uid, view, action in actions:
            self._execute(agent, uid, action)
            after = self.calculator.snapshot()
            if uid in after.accounts.index:
                agent.record_outcome(action, AccountView.from_snapshot(after, uid))

    def _execute(self, agent: TradingAgent, uid: int, action: AgentAction) -> None:
        p = agent.persona
        fills: list[Any] = []
        if action.intents:
            fills.extend(self.gateway.execute(uid, action.intents, leverage=p.max_leverage))
        for burst in action.bursts:
            # El flujo tóxico se cruza contra el libro SOMBRA y el manipulador queda con la posición resultante
            for _ in range(burst.slices):
                flow = self.market.execute_flow(burst.symbol, burst.side, burst.notional_usd / burst.slices)
                if flow.vwap is None or flow.filled_quantity <= 0:
                    break
                signed = flow.filled_quantity if burst.side == "BUY" else -flow.filled_quantity
                intent = OrderIntent(burst.symbol, signed, flow.vwap, False, "stress burst")
                fills.extend(self.gateway.execute(uid, [intent], leverage=p.max_leverage,
                                                  price_overrides={burst.symbol: flow.vwap}))
        with self._rec_lock:
            self._actions.append({
                "t": round(self._elapsed(), 3), "agent": p.name, "mode": self._mode.value, "source": action.source,
                "intents": len(action.intents), "bursts": len(action.bursts),
                "filled": sum(1 for f in fills if f.status != "REJECTED"),
                "rejected": sum(1 for f in fills if f.status == "REJECTED"),
                "notes": "; ".join(action.guardrail_notes), "rationale": action.rationale[:200],
            })

    # ---------------------------------------------------------------- report
    def _build_report(self, fund_start: float) -> SimulationReport:
        with self._rec_lock:
            timeline = pd.DataFrame(self._timeline)
            actions = pd.DataFrame(self._actions)
            outcomes = list(self._outcomes)
            mode_changes = tuple(self._mode_changes)
        names = {uid: name for name, uid in self._user_ids.items()}
        liq = pd.DataFrame([
            {"t": t, "mode": mode, "agent": names.get(o.user_id, o.user_id), "user_id": o.user_id,
             "equity_before": float(o.equity_before or 0), "mm_before": float(o.maintenance_margin_before or 0),
             "fee": float(o.liquidation_fee), "deficit": float(o.deficit),
             "fund_after": float(o.insurance_fund_balance or 0), "positions": len(o.fills),
             "max_slippage_bps": max((f.slippage_bps for f in o.fills), default=0.0),
             "latency_ms": round(o.latency_ms, 2)}
            for t, o, mode in outcomes
        ])
        fund_end = self._fetch_fund_balance()
        final = self.calculator.snapshot()
        equities = {}
        for name, uid in self._user_ids.items():
            equities[name] = round(float(final.account(uid)["equity"]), 2) if uid in final.accounts.index else "LIQUIDATED"
        summary = {
            "duration_s": round(self._elapsed(), 2),
            "ticks": len(timeline),
            "mode_changes": [f"{t:.1f}s->{m}" for t, m in mode_changes],
            "liquidations": len(liq),
            "liquidations_by_agent": liq.groupby("agent").size().to_dict() if not liq.empty else {},
            "deficit_absorbed_usd": round(float(liq["deficit"].sum()), 2) if not liq.empty else 0.0,
            "fees_collected_usd": round(float(liq["fee"].sum()), 2) if not liq.empty else 0.0,
            "insurance_fund_start": round(fund_start, 2),
            "insurance_fund_end": round(fund_end, 2),
            "insurance_fund_min": round(min(self._fund_min, fund_end), 2),
            "adl_required": fund_end < 0,
            "agent_equity_end": equities,
            "agent_actions": len(actions),
            "llm_fallbacks": int((actions["source"] == "RULES_FALLBACK").sum()) if not actions.empty else 0,
            "loop_errors": self._errors,
        }
        log.info("orchestrator.report", **{k: v for k, v in summary.items() if k != "agent_equity_end"})
        return SimulationReport(timeline, liq, actions, mode_changes, summary)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:  # pragma: no cover - punto de entrada interactivo
    from common.logging_config import configure_logging
    from database.connection import get_db

    parser = argparse.ArgumentParser(description="Orquestador de agentes + motor de riesgo")
    parser.add_argument("--duration", type=float, default=0.0, help="segundos de reloj (0 = servicio continuo)")
    parser.add_argument("--black-swan-at", type=float, default=None,
                        help="segundo en que se dispara el cisne negro (por defecto: solo desde el dashboard)")
    parser.add_argument("--tick", type=float, default=0.25, help="segundos por paso de 5 min simulado")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--live", action="store_true", help="precios base en vivo de Hyperliquid Testnet")
    parser.add_argument("--insurance-fund", type=float, default=None, help="capitaliza el fondo antes de empezar")
    parser.add_argument("--csv", type=str, default=None, help="prefijo para exportar timeline/liquidaciones")
    parser.add_argument("--whale-bias", choices=["LONG", "SHORT", "MOMENTUM"], default="LONG",
                        help="sesgo de la ballena (política determinista). LONG = objetivo del cisne negro")
    parser.add_argument("--no-llm", action="store_true", help="fuerza la política determinista aunque haya API key")
    args = parser.parse_args()

    configure_logging()
    db = get_db()
    if args.insurance_fund is not None:
        with db.transaction() as cur:
            cur.execute("UPDATE insurance_fund SET balance = balance + %s WHERE fund_id = 1 RETURNING balance",
                        (args.insurance_fund,))
            bal = cur.fetchone()["balance"]
            cur.execute("""INSERT INTO insurance_fund_ledger (entry_type, amount, balance_after, note)
                           VALUES ('CAPITAL_INJECTION', %s, %s, 'CLI')""", (args.insurance_fund, bal))

    feed = None
    if args.live:
        from exchange_connector.hyperliquid_ws import HyperliquidWebSocket

        feed = HyperliquidWebSocket(coins=("BTC", "ETH"))
        feed.start_in_background()
        if not feed.store.wait_until_ready(timeout=20):
            raise SystemExit("Sin datos del feed de Hyperliquid en 20 s")

    cfg = OrchestratorConfig(tick_interval_s=args.tick, black_swan_after_s=args.black_swan_at, seed=args.seed)
    agents = default_agents(whale_bias=args.whale_bias, use_llm=False if args.no_llm else None)
    orch = Orchestrator(db, config=cfg, agents=agents, live_store=feed.store if feed else None)
    for sig in (signal.SIGTERM, signal.SIGINT):  # parada ordenada en `docker stop` / Ctrl-C
        signal.signal(sig, lambda *_: orch.stop())
    try:
        report = orch.run(args.duration if args.duration > 0 else math.inf)
    finally:
        if feed:
            feed.stop_background()
    report.print()
    if args.csv:
        report.timeline.to_csv(f"{args.csv}_timeline.csv", index=False)
        report.liquidations.to_csv(f"{args.csv}_liquidations.csv", index=False)
        report.actions.to_csv(f"{args.csv}_actions.csv", index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
