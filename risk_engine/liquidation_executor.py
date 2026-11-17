"""
risk_engine/liquidation_executor.py
===================================

Motor de liquidación atómica cross-margin.

Ciclo (``run_cycle``)
---------------------
1. **Screening sin locks**: ``MarginCalculator`` evalúa todas las cuentas (snapshot MVCC) y
   devuelve las candidatas (``Equity ≤ MM``).
2. **Liquidación por cuenta** (``liquidate_account``), en UNA transacción ACID:

   ::

       BEGIN (READ COMMITTED, lock_timeout, statement_timeout)
         1. lock_account(NOWAIT)            -- otro liquidador la tiene => SKIPPED_LOCKED
         2. lock_open_positions             -- ORDER BY position_id
         3. RE-EVALUACIÓN bajo lock         -- el screening pudo quedar obsoleto (depósito, cierre...)
         4. precio de bancarrota + ejecución (L2 simulado o Hyperliquid Testnet)
         5. UPDATE open_positions -> LIQUIDATED
         6. INSERT liquidation_events (inmutable, 1 fila/posición, batch UUID común)
         7. lock_insurance_fund             -- ÚLTIMO: fila caliente, se retiene el mínimo tiempo
         8. UPDATE insurance_fund (+fee −déficit) + INSERT insurance_fund_ledger
         9. UPDATE users_collateral -> colateral residual, status LIQUIDATED
       COMMIT      (cualquier excepción => ROLLBACK total: la cuenta queda intacta)

   Orden de bloqueo canónico: ``users_collateral → open_positions → insurance_fund`` (anti-deadlock).
   Deadlocks/serialization failures residuales se reintentan con ``retry_on_transient``.

Economía de la liquidación (por cuenta)
---------------------------------------
::

    realized_pnl_i  = q_i · (exec_px_i − entry_i)
    equity_after    = collateral + Σ realized_pnl_i
    fee             = min(fee_rate · Σ |q_i|·exec_px_i ,  max(equity_after, 0))   -- nunca crea déficit
    deficit         = max(−equity_after, 0)                                       -- bad debt
    collateral_new  = max(equity_after, 0) − fee
    insurance_fund += fee − deficit       (balance < 0 => fondo agotado => ADL, ver risk_engine/adl.py)

Auto-Deleveraging
-----------------
Si una liquidación deja el fondo en negativo, ``run_cycle`` ejecuta a continuación el ADL
(``AdlEngine``) en su propia transacción: recorta pro-rata las posiciones ganadoras y repone el
fondo, con auditoría en ``adl_events``/``adl_fills``. Mientras quede un residual, cada ciclo lo
reintenta con las nuevas ganancias. Se desactiva con ``LiquidationConfig(adl_enabled=False)``.

Precio de ejecución
-------------------
* ``SIMULATED``: recorre el L2 (``L2BookSnapshot.simulate_market_order``). En modo estrés, el
  ``liquidity_factor`` ∈ (0, 1] reduce la profundidad visible (p.ej. ``SimulationResult.liquidity``).
  La parte que el libro no absorbe la asume un **backstop** (estilo vault HLP) a
  ``mark · (1 ∓ backstop_discount)``.
* ``LIVE_TESTNET``: orden market reduce-only en Hyperliquid con ``cloid`` determinista derivado
  del batch (idempotente frente a reintentos). Se exige fill completo; si no, ROLLBACK.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import psycopg2.extras
import structlog

from database.connection import (
    DatabaseManager,
    LockAcquisitionError,
    WaitPolicy,
    lock_account,
    lock_insurance_fund,
    lock_open_positions,
    retry_on_transient,
)
from risk_engine.adl import AdlConfig, AdlEngine, AdlOutcome
from risk_engine.margin_calculator import MarginCalculator, MarginSnapshot, compute_margins

if TYPE_CHECKING:
    from exchange_connector.hyperliquid_client import HyperliquidClient
    from exchange_connector.hyperliquid_ws import MarketDataStore, MarketSnapshot

__all__ = [
    "ExecutionMode",
    "MarketRegime",
    "LiquidationStatus",
    "LiquidationConfig",
    "PositionFill",
    "LiquidationOutcome",
    "CycleReport",
    "LiquidationExecutor",
    "LiveExecutionError",
    "AdlOutcome",
]

log = structlog.get_logger(__name__)

Q = Decimal("0.00000001")  # 1e-8: precisión de persistencia (USDC usa 6 decimales)
ZERO = Decimal(0)
TRIGGER_REASON = "EQUITY_BELOW_MAINTENANCE"


def _d(value: Any) -> Decimal:
    """float/str/Decimal -> Decimal cuantizado a 1e-8 (vía repr: sin artefactos binarios)."""
    if isinstance(value, Decimal):
        return value.quantize(Q, rounding=ROUND_HALF_EVEN)
    return Decimal(repr(float(value))).quantize(Q, rounding=ROUND_HALF_EVEN)


def _allocate(total: Decimal, weights: Sequence[float]) -> list[Decimal]:
    """Reparte ``total`` pro-rata por ``weights``; el residuo de redondeo va a la última fila (suma exacta)."""
    if not weights:
        return []
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    parts = [_d(total * Decimal(repr(float(x)))) for x in w[:-1]]
    parts.append(total - sum(parts, ZERO))
    return parts


class LiveExecutionError(RuntimeError):
    """El exchange no ejecutó el cierre completo: la transacción se revierte."""


class ExecutionMode(str, Enum):
    SIMULATED = "SIMULATED"        # ejecución contra el L2 (shadow ledger)
    LIVE_TESTNET = "LIVE_TESTNET"  # cierre real en Hyperliquid Testnet


class MarketRegime(str, Enum):  # = ENUM market_regime del esquema
    LIVE_TESTNET = "LIVE_TESTNET"
    STRESS_MONTE_CARLO = "STRESS_MONTE_CARLO"


class LiquidationStatus(str, Enum):
    LIQUIDATED = "LIQUIDATED"
    NOT_LIQUIDATABLE = "NOT_LIQUIDATABLE"  # recuperada entre el screening y el lock
    SKIPPED_LOCKED = "SKIPPED_LOCKED"      # otro worker la está procesando
    STALE_PRICES = "STALE_PRICES"          # fail-safe: sin precio fresco para algún activo
    ACCOUNT_INACTIVE = "ACCOUNT_INACTIVE"  # FROZEN / ya LIQUIDATED
    NO_POSITIONS = "NO_POSITIONS"
    EXECUTION_FAILED = "EXECUTION_FAILED"  # fallo del exchange en modo live (ROLLBACK)


@dataclass(frozen=True, slots=True)
class LiquidationConfig:
    mode: ExecutionMode = ExecutionMode.SIMULATED
    regime: MarketRegime = MarketRegime.LIVE_TESTNET
    liquidation_fee_rate: float = 0.005        # 50 bps sobre el nocional ejecutado
    backstop_discount_bps: float = 100.0       # descuento del backstop sobre el mark
    max_slippage_bps: float = 500.0            # alerta (simulado) / tope de la orden IOC (live)
    liquidity_factor: float = 1.0              # (0, 1]: haircut de profundidad del L2 en estrés
    max_price_age_s: float | None = 5.0        # None => sin control de frescura (replays)
    lock_timeout_ms: int = 1_000
    statement_timeout_ms: int = 10_000
    max_retries: int = 3
    adl_enabled: bool = True                   # auto-deleveraging cuando el Insurance Fund queda en negativo
    adl_min_health: float = 1.10               # health ratio mínimo que el ADL respeta en las contrapartes

    def __post_init__(self) -> None:
        if not 0 <= self.liquidation_fee_rate < 0.1:
            raise ValueError("liquidation_fee_rate fuera de [0, 10 %)")
        if not 0 < self.liquidity_factor <= 1:
            raise ValueError("liquidity_factor debe estar en (0, 1]")
        if self.backstop_discount_bps < 0 or self.max_slippage_bps <= 0:
            raise ValueError("backstop_discount_bps >= 0 y max_slippage_bps > 0")


@dataclass(frozen=True, slots=True)
class PositionFill:
    position_id: int
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    oracle_price: Decimal | None
    bankruptcy_price: Decimal
    execution_price: Decimal
    slippage_bps: float
    book_fill_fraction: float
    realized_pnl: Decimal
    notional: Decimal
    source: str                       # L2_BOOK | L2_BOOK+BACKSTOP | BACKSTOP | EXCHANGE
    exchange_oid: int | None = None
    cloid: str | None = None


@dataclass(frozen=True)
class LiquidationOutcome:
    status: LiquidationStatus
    user_id: int
    batch_id: uuid.UUID | None = None
    equity_before: Decimal | None = None
    maintenance_margin_before: Decimal | None = None
    fills: tuple[PositionFill, ...] = ()
    liquidation_fee: Decimal = ZERO
    deficit: Decimal = ZERO
    final_collateral: Decimal | None = None
    insurance_fund_balance: Decimal | None = None
    event_ids: tuple[int, ...] = ()
    reason: str | None = None
    latency_ms: float = 0.0

    @property
    def liquidated(self) -> bool:
        return self.status is LiquidationStatus.LIQUIDATED

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value, "user_id": self.user_id,
            "batch_id": str(self.batch_id) if self.batch_id else None,
            "equity_before": str(self.equity_before) if self.equity_before is not None else None,
            "mm_before": str(self.maintenance_margin_before) if self.maintenance_margin_before is not None else None,
            "positions": len(self.fills), "fee": str(self.liquidation_fee), "deficit": str(self.deficit),
            "final_collateral": str(self.final_collateral) if self.final_collateral is not None else None,
            "fund_balance": str(self.insurance_fund_balance) if self.insurance_fund_balance is not None else None,
            "reason": self.reason, "latency_ms": round(self.latency_ms, 2),
        }


@dataclass(frozen=True)
class CycleReport:
    snapshot: MarginSnapshot
    outcomes: tuple[LiquidationOutcome, ...]
    duration_ms: float
    adl: tuple[AdlOutcome, ...] = ()

    @property
    def liquidated(self) -> tuple[LiquidationOutcome, ...]:
        return tuple(o for o in self.outcomes if o.liquidated)


ClientResolver = Callable[[Mapping[str, Any]], "HyperliquidClient"]


# =============================================================================
# Executor
# =============================================================================
class LiquidationExecutor:
    """
    Args:
        db: gestor transaccional (Fase 1).
        store: fuente de mercado con la interfaz de ``MarketDataStore`` (``get``, ``is_fresh``).
            En modo estrés puede ser un store alimentado por el Monte Carlo.
        config: parámetros de liquidación.
        client_resolver: en ``LIVE_TESTNET``, devuelve el ``HyperliquidClient`` de la cuenta
            (cada agente opera con su propia API wallet). Recibe la fila de ``users_collateral``.
    """

    def __init__(
        self,
        db: DatabaseManager,
        store: "MarketDataStore",
        config: LiquidationConfig | None = None,
        *,
        client_resolver: ClientResolver | None = None,
    ) -> None:
        self.db = db
        self.store = store
        self.config = config or LiquidationConfig()
        self.client_resolver = client_resolver
        self.calculator = MarginCalculator(db, store, max_price_age_s=self.config.max_price_age_s)
        if self.config.mode is ExecutionMode.LIVE_TESTNET and client_resolver is None:
            raise ValueError("El modo LIVE_TESTNET requiere client_resolver")
        self.adl = AdlEngine(
            db, store,
            AdlConfig(lock_timeout_ms=self.config.lock_timeout_ms, statement_timeout_ms=self.config.statement_timeout_ms,
                      max_price_age_s=self.config.max_price_age_s, max_retries=self.config.max_retries,
                      min_health=self.config.adl_min_health),
            regime=self.config.regime.value,
        ) if self.config.adl_enabled else None

    # ------------------------------------------------------------- ciclo
    def run_cycle(self) -> CycleReport:
        """Screening de todas las cuentas + liquidación de las candidatas (peor salud primero)."""
        t0 = time.perf_counter()
        snap = self.calculator.snapshot()
        outcomes = tuple(self.liquidate_account(int(uid)) for uid in snap.liquidatable().index)
        adl = self._run_adl(outcomes)
        report = CycleReport(snap, outcomes, (time.perf_counter() - t0) * 1e3, adl)
        if outcomes:
            log.info(
                "liquidation.cycle", candidates=len(outcomes), liquidated=len(report.liquidated),
                duration_ms=round(report.duration_ms, 2), **snap.summary(),
            )
        return report

    def _run_adl(self, outcomes: Sequence[LiquidationOutcome]) -> tuple[AdlOutcome, ...]:
        """ADL tras la liquidación que agotó el fondo (siempre auditado) o reintento de un residual."""
        if self.adl is None:
            return ()
        trigger = next((o.batch_id for o in reversed(outcomes) if o.liquidated and o.insurance_fund_balance is not None
                        and o.insurance_fund_balance < 0), None)
        try:
            result = self.adl.run(trigger_batch_id=trigger)
        except Exception:  # noqa: BLE001 - un fallo del ADL no invalida las liquidaciones ya confirmadas
            log.exception("adl.error", trigger_batch_id=str(trigger) if trigger else None)
            return ()
        return () if result is None else (result,)

    def run_forever(self, stop: threading.Event, interval_s: float = 1.0) -> None:
        """Bucle de vigilancia. Un fallo de ciclo se registra y no detiene el motor."""
        log.info("liquidation.engine.started", mode=self.config.mode.value, regime=self.config.regime.value)
        while not stop.is_set():
            try:
                self.run_cycle()
            except Exception:  # noqa: BLE001 - el vigilante nunca debe morir
                log.exception("liquidation.cycle.error")
            stop.wait(interval_s)
        log.info("liquidation.engine.stopped")

    # ------------------------------------------------------- por cuenta
    def liquidate_account(self, user_id: int) -> LiquidationOutcome:
        """Liquida la cuenta si sigue siendo liquidable bajo lock. Nunca lanza por condiciones de negocio."""
        t0 = time.perf_counter()
        batch_id = uuid.uuid4()
        attempt = retry_on_transient(max_attempts=self.config.max_retries)(self._liquidate_tx)
        try:
            outcome = attempt(user_id, batch_id)
        except LockAcquisitionError:
            outcome = LiquidationOutcome(LiquidationStatus.SKIPPED_LOCKED, user_id, reason="cuenta bloqueada por otro worker")
        except LiveExecutionError as exc:
            log.error("liquidation.live_execution_failed", user_id=user_id, batch_id=str(batch_id), error=str(exc))
            outcome = LiquidationOutcome(LiquidationStatus.EXECUTION_FAILED, user_id, batch_id=batch_id, reason=str(exc))
        outcome = _with_latency(outcome, t0)
        level = log.warning if outcome.liquidated else log.debug
        level("liquidation.outcome", **outcome.as_dict())
        return outcome

    def _liquidate_tx(self, user_id: int, batch_id: uuid.UUID) -> LiquidationOutcome:
        cfg = self.config
        with self.db.transaction(lock_timeout_ms=cfg.lock_timeout_ms, statement_timeout_ms=cfg.statement_timeout_ms) as cur:
            # 1-2) Bloqueo en orden canónico
            account = lock_account(cur, user_id, wait=WaitPolicy.NOWAIT)
            if str(account["account_status"]) != "ACTIVE":
                return LiquidationOutcome(LiquidationStatus.ACCOUNT_INACTIVE, user_id, reason=str(account["account_status"]))
            positions = lock_open_positions(cur, user_id)
            if not positions:
                return LiquidationOutcome(LiquidationStatus.NO_POSITIONS, user_id)

            # 3) Re-evaluación bajo lock con precios frescos (fail-safe)
            symbols = sorted({p["symbol"] for p in positions})
            markets = {s: self._fresh_market(s) for s in symbols}
            stale = [s for s, m in markets.items() if m is None]
            if stale:
                return LiquidationOutcome(LiquidationStatus.STALE_PRICES, user_id, reason=f"sin precio fresco: {stale}")
            marks = {s: float(m.mark_price) for s, m in markets.items()}  # type: ignore[union-attr]
            snap = compute_margins(
                pd.DataFrame([{"user_id": user_id, "deposited_collateral": float(account["deposited_collateral"])}]),
                pd.DataFrame(positions).astype({"quantity": float, "entry_price": float,
                                                "maintenance_margin_rate": float, "leverage": float}),
                marks,
                statuses=None,
            )
            acc = snap.account(user_id)
            if not bool(acc["is_liquidatable"]):
                return LiquidationOutcome(
                    LiquidationStatus.NOT_LIQUIDATABLE, user_id,
                    equity_before=_d(acc["equity"]), maintenance_margin_before=_d(acc["maintenance_margin"]),
                    reason=f"health_ratio={acc['health_ratio']:.4f} bajo lock",
                )
            bk_by_pos = dict(zip(snap.positions["position_id"], snap.positions["bankruptcy_price"]))

            # 4) Ejecución del cierre de cada posición
            client = self.client_resolver(account) if cfg.mode is ExecutionMode.LIVE_TESTNET else None  # type: ignore[misc]
            fills = tuple(
                self._close_position(p, markets[p["symbol"]], bk_by_pos[p["position_id"]], batch_id, client)  # type: ignore[arg-type]
                for p in positions
            )

            # Economía de la liquidación (Decimal exacto)
            collateral = _d(account["deposited_collateral"])
            equity_after = collateral + sum((f.realized_pnl for f in fills), ZERO)
            notional_total = sum((f.notional for f in fills), ZERO)
            fee = min(_d(notional_total * Decimal(repr(cfg.liquidation_fee_rate))), max(equity_after, ZERO))
            deficit = max(-equity_after, ZERO)
            final_collateral = max(equity_after, ZERO) - fee
            weights = [float(f.notional) for f in fills]
            fee_alloc, deficit_alloc = _allocate(fee, weights), _allocate(deficit, weights)

            equity_before = _d(acc["equity"])
            mm_before = _d(acc["maintenance_margin"])
            health_before = _d(acc["health_ratio"]) if np.isfinite(acc["health_ratio"]) else None

            # 5) Posiciones -> LIQUIDATED
            for f in fills:
                cur.execute(
                    """UPDATE open_positions
                          SET status = 'LIQUIDATED', closed_at = now(), last_mark_price = %s
                        WHERE position_id = %s AND status = 'OPEN'""",
                    (f.mark_price, f.position_id),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(f"Posición {f.position_id} no estaba OPEN bajo lock (invariante roto)")

            # 6) Eventos inmutables
            rows = [
                (
                    str(batch_id), user_id, f.position_id, f.symbol, f.quantity, f.entry_price, f.mark_price,
                    f.oracle_price, f.execution_price, f.bankruptcy_price, _d(f.slippage_bps), collateral,
                    equity_before, mm_before, health_before, f.realized_pnl, fee_i, deficit_i, fee_i - deficit_i,
                    cfg.regime.value, TRIGGER_REASON,
                    psycopg2.extras.Json({
                        "source": f.source, "book_fill_fraction": round(f.book_fill_fraction, 6),
                        "mode": cfg.mode.value, "liquidity_factor": cfg.liquidity_factor,
                        "cloid": f.cloid, "exchange_oid": f.exchange_oid,
                    }),
                )
                for f, fee_i, deficit_i in zip(fills, fee_alloc, deficit_alloc)
            ]
            inserted = psycopg2.extras.execute_values(
                cur,
                """INSERT INTO liquidation_events (
                       liquidation_batch_id, user_id, position_id, symbol, quantity, entry_price, mark_price,
                       oracle_price, execution_price, bankruptcy_price, slippage_bps, collateral_before,
                       account_equity_before, maintenance_margin_before, margin_ratio_before, realized_pnl,
                       liquidation_fee, deficit, insurance_fund_delta, market_regime, trigger_reason, metadata)
                   VALUES %s
                   RETURNING event_id, position_id""",
                rows,
                template="(%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::market_regime,%s,%s)",
                fetch=True,
            )
            event_by_pos = {r["position_id"]: r["event_id"] for r in inserted}
            event_ids = tuple(event_by_pos[f.position_id] for f in fills)

            # 7-8) Insurance Fund: último lock del protocolo
            fund = lock_insurance_fund(cur)
            balance = _d(fund["balance"])
            ledger: list[tuple[str, Decimal, Decimal, int, str]] = []
            if fee > 0:
                balance += fee
                ledger.append(("LIQUIDATION_FEE", fee, balance, event_ids[0], f"batch {batch_id}"))
            if deficit > 0:
                balance -= deficit
                ledger.append(("BANKRUPTCY_DEFICIT", -deficit, balance, event_ids[0], f"batch {batch_id} user {user_id}"))
            if ledger:
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO insurance_fund_ledger (entry_type, amount, balance_after, liquidation_event_id, note)
                       VALUES %s""",
                    ledger,
                    template="(%s::insurance_entry_type,%s,%s,%s,%s)",
                )
                cur.execute("UPDATE insurance_fund SET balance = %s WHERE fund_id = 1", (balance,))
            if balance < 0:
                log.critical("insurance_fund.depleted", balance=str(balance), batch_id=str(batch_id),
                             action="ADL" if cfg.adl_enabled else "ADL desactivado: déficit sin socializar")

            # 9) Cuenta: colateral residual y estado final
            cur.execute(
                """UPDATE users_collateral
                      SET deposited_collateral = %s, account_status = 'LIQUIDATED'
                    WHERE user_id = %s""",
                (final_collateral, user_id),
            )

            return LiquidationOutcome(
                status=LiquidationStatus.LIQUIDATED,
                user_id=user_id,
                batch_id=batch_id,
                equity_before=equity_before,
                maintenance_margin_before=mm_before,
                fills=fills,
                liquidation_fee=fee,
                deficit=deficit,
                final_collateral=final_collateral,
                insurance_fund_balance=balance,
                event_ids=event_ids,
            )
        # COMMIT al salir del bloque `with`

    # -------------------------------------------------------- mercado
    def _fresh_market(self, symbol: str) -> "MarketSnapshot | None":
        snap = self.store.get(symbol)
        if snap is None or snap.mark_price is None or snap.mark_price <= 0:
            return None
        if self.config.max_price_age_s is not None and not self.store.is_fresh(symbol, self.config.max_price_age_s):
            return None
        return snap

    def _close_position(
        self,
        position: Mapping[str, Any],
        market: "MarketSnapshot",
        bankruptcy_price: float,
        batch_id: uuid.UUID,
        client: "HyperliquidClient | None",
    ) -> PositionFill:
        cfg = self.config
        q = _d(position["quantity"])
        entry = _d(position["entry_price"])
        size = float(abs(q))
        is_long = q > 0
        mark = float(market.mark_price)  # type: ignore[arg-type]
        oid: int | None = None
        cloid: str | None = None

        if cfg.mode is ExecutionMode.LIVE_TESTNET:
            from exchange_connector.hyperliquid_client import OrderStatus, Side

            cloid = "0x" + hashlib.blake2b(f"{batch_id}:{position['position_id']}".encode(), digest_size=16).hexdigest()
            try:
                result = client.place_market_order(  # type: ignore[union-attr]
                    position["symbol"], Side.SHORT if is_long else Side.LONG, abs(q),
                    slippage=cfg.max_slippage_bps / 1e4, reference_price=mark, reduce_only=True, cloid=cloid,
                )
            except Exception as exc:  # red, validación, rate limit... => ROLLBACK de toda la liquidación
                raise LiveExecutionError(f"{position['symbol']}: {type(exc).__name__}: {exc}") from exc
            if result.status is not OrderStatus.FILLED or result.avg_fill_price is None:
                raise LiveExecutionError(
                    f"{position['symbol']} pos {position['position_id']}: {result.status.value} "
                    f"filled={result.filled_size}/{abs(q)} error={result.error}"
                )
            exec_px, fill_frac, source, oid = float(result.avg_fill_price), 1.0, "EXCHANGE", result.oid
        else:
            exec_px, fill_frac, source = self._simulate_execution(market, size, is_long, mark)

        slippage_bps = abs(exec_px - mark) / mark * 1e4
        if slippage_bps > cfg.max_slippage_bps:
            log.warning("liquidation.high_slippage", symbol=position["symbol"], position_id=position["position_id"],
                        slippage_bps=round(slippage_bps, 2), source=source)
        exec_d = _d(exec_px)
        oracle = market.oracle_price
        return PositionFill(
            position_id=int(position["position_id"]),
            symbol=str(position["symbol"]),
            quantity=q,
            entry_price=entry,
            mark_price=_d(mark),
            oracle_price=_d(oracle) if oracle else None,
            bankruptcy_price=_d(max(bankruptcy_price, 0.0)),
            execution_price=exec_d,
            slippage_bps=slippage_bps,
            book_fill_fraction=fill_frac,
            realized_pnl=_d(q * (exec_d - entry)),
            notional=_d(abs(q) * exec_d),
            source=source,
            exchange_oid=oid,
            cloid=cloid,
        )

    def _simulate_execution(self, market: "MarketSnapshot", size: float, is_long: bool, mark: float) -> tuple[float, float, str]:
        """
        VWAP contra el L2 con haircut de liquidez. Escalar todos los tamaños del libro por L equivale a
        consumir ``size / L`` del libro original (mismos niveles, mismos pesos relativos).
        """
        cfg = self.config
        L = cfg.liquidity_factor
        side = "sell" if is_long else "buy"
        book = market.book
        vwap, filled = (None, 0.0)
        if book is not None:
            vwap, raw_filled = book.simulate_market_order(side, size / L)
            filled = min(raw_filled * L, size)
        discount = cfg.backstop_discount_bps / 1e4
        backstop_px = mark * (1 - discount) if is_long else mark * (1 + discount)

        if vwap is not None and filled >= size * (1 - 1e-12):
            return vwap, 1.0, "L2_BOOK"
        if vwap is not None and filled > 0:
            px = (vwap * filled + backstop_px * (size - filled)) / size
            return px, filled / size, "L2_BOOK+BACKSTOP"
        return backstop_px, 0.0, "BACKSTOP"


def _with_latency(outcome: LiquidationOutcome, t0: float) -> LiquidationOutcome:
    object.__setattr__(outcome, "latency_ms", (time.perf_counter() - t0) * 1e3)
    return outcome
