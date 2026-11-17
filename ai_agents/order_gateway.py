"""
ai_agents/order_gateway.py
==========================

Pasarela de ejecución de los agentes sobre el **shadow ledger** (PostgreSQL), con la misma disciplina
transaccional que el motor de liquidación:

    BEGIN
      lock_account (lock_timeout corto: si el liquidador la tiene, la orden se rechaza esta ronda)
      lock_open_positions
      por cada intención: precio (L2 sombra o override) -> netting -> control de margen inicial
      UPDATE/INSERT open_positions · UPDATE users_collateral
    COMMIT

Netting (una posición OPEN por usuario/símbolo, invariante de ``idx_user_risk``)
---------------------------------------------------------------------------
* Aumento:   entry' = (q·entry + Δq·px) / (q + Δq)
* Reducción: PnL realizado = q_cerrada · (px − entry) se abona al colateral; entry no cambia
* Giro:      cierra la posición (CLOSED) y abre una nueva por el remanente a ``px``
* Cierre:    status CLOSED, ``closed_at``

Controles pre-trade: cuenta ACTIVE, margen inicial post-trade ≤ equity (salvo reduce-only),
límites de riesgo de la cuenta (``max_leverage`` y ``max_gross_notional_usd`` de ``users_collateral``,
ver ``risk_engine.risk_limits``) y colateral nunca negativo (una cuenta insolvente es asunto del
liquidador, no de la pasarela). Los límites se evalúan bajo el lock de la cuenta: son el control
autoritativo, los guardarraíles del agente solo una primera capa.

Modo live opcional: con ``client_resolver`` las órdenes de Whale y Arbitrageur se envían a
Hyperliquid Testnet (reduce-only cuando procede) y se registran al precio real de ejecución.
El perfil ``MARKET_MANIPULATOR`` está bloqueado por código para cualquier enrutamiento externo.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import structlog

from ai_agents.trader_agents import AgentProfile, OrderIntent
from risk_engine.risk_limits import RiskLimits
from database.connection import (
    DatabaseManager,
    LockAcquisitionError,
    lock_account,
    lock_open_positions,
    retry_on_transient,
)

if TYPE_CHECKING:
    from exchange_connector.hyperliquid_client import HyperliquidClient
    from exchange_connector.hyperliquid_ws import MarketDataStore

__all__ = ["GatewayConfig", "Fill", "OrderGateway"]

log = structlog.get_logger(__name__)
Q = Decimal("0.00000001")


def _d(x: float | Decimal) -> Decimal:
    return (x if isinstance(x, Decimal) else Decimal(repr(float(x)))).quantize(Q, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    taker_fee_bps: float = 4.5                    # comisión taker (orden de magnitud de Hyperliquid)
    maintenance_margin_rates: Mapping[str, float] = field(default_factory=lambda: {"BTC": 0.0125, "ETH": 0.02})
    lock_timeout_ms: int = 300
    max_slippage_bps: float = 300.0               # el resto de la orden no se ejecuta más allá de este tope
    live_slippage: float = 0.01


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    requested: float
    filled: float
    price: float | None
    fee: float
    realized_pnl: float
    position_after: float
    status: str                                   # FILLED | PARTIAL | REJECTED
    reason: str = ""


@dataclass
class _Pos:
    position_id: int | None
    quantity: float
    entry: float
    leverage: float
    mmr: float


class OrderGateway:
    def __init__(
        self,
        db: DatabaseManager,
        store: "MarketDataStore",
        config: GatewayConfig | None = None,
        *,
        client_resolver: Callable[[Mapping[str, Any]], "HyperliquidClient"] | None = None,
    ) -> None:
        self.db = db
        self.store = store
        self.config = config or GatewayConfig()
        self.client_resolver = client_resolver

    def execute(
        self,
        user_id: int,
        intents: Sequence[OrderIntent],
        *,
        leverage: float,
        price_overrides: Mapping[str, float] | None = None,
    ) -> tuple[Fill, ...]:
        """Ejecuta las intenciones en una transacción. ``price_overrides`` fija el precio (flujo ya cruzado)."""
        if not intents:
            return ()
        try:
            return retry_on_transient(max_attempts=3)(self._execute_tx)(user_id, intents, leverage, price_overrides or {})
        except LockAcquisitionError:
            return tuple(Fill(i.symbol, i.delta_quantity, 0.0, None, 0.0, 0.0, math.nan, "REJECTED", "LOCKED")
                         for i in intents)

    def headroom_usd(self, user_id: int) -> float:
        """
        Nocional adicional que la cuenta aún puede abrir según sus límites (lectura MVCC, sin locks).
        Pre-check para dimensionar el flujo ANTES de cruzarlo contra el libro sombra; el control
        autoritativo sigue siendo el de ``_apply`` bajo lock.
        """
        with self.db.transaction(readonly=True) as cur:
            cur.execute("""SELECT deposited_collateral, max_leverage, max_gross_notional_usd, account_status
                             FROM users_collateral WHERE user_id = %s""", (user_id,))
            account = cur.fetchone()
            if account is None or str(account["account_status"]) != "ACTIVE":
                return 0.0
            cur.execute("""SELECT symbol, quantity, entry_price FROM open_positions
                            WHERE user_id = %s AND status = 'OPEN'""", (user_id,))
            rows = cur.fetchall()
        equity, gross = float(account["deposited_collateral"]), 0.0
        for r in rows:
            snap = self.store.get(r["symbol"])
            mark = snap.mark_price if snap is not None and snap.mark_price else float(r["entry_price"])
            equity += float(r["quantity"]) * (mark - float(r["entry_price"]))
            gross += abs(float(r["quantity"])) * mark
        return RiskLimits.from_row(account).headroom(gross, equity)

    # ------------------------------------------------------------------ núcleo
    def _execute_tx(self, user_id: int, intents: Sequence[OrderIntent], leverage: float,
                    overrides: Mapping[str, float]) -> tuple[Fill, ...]:
        cfg = self.config
        with self.db.transaction(lock_timeout_ms=cfg.lock_timeout_ms) as cur:
            account = lock_account(cur, user_id)
            if str(account["account_status"]) != "ACTIVE":
                return tuple(Fill(i.symbol, i.delta_quantity, 0.0, None, 0.0, 0.0, math.nan, "REJECTED",
                                  f"cuenta {account['account_status']}") for i in intents)
            rows = lock_open_positions(cur, user_id)
            book: dict[str, _Pos] = {
                r["symbol"]: _Pos(r["position_id"], float(r["quantity"]), float(r["entry_price"]),
                                  float(r["leverage"]), float(r["maintenance_margin_rate"])) for r in rows
            }
            collateral = float(account["deposited_collateral"])
            is_manipulator = str(account["agent_profile"]) == AgentProfile.MARKET_MANIPULATOR.value
            fills: list[Fill] = []

            for intent in intents:
                fill, collateral = self._apply(cur, account, user_id, intent, book, collateral, leverage,
                                               overrides.get(intent.symbol), is_manipulator)
                fills.append(fill)

            cur.execute("UPDATE users_collateral SET deposited_collateral = %s WHERE user_id = %s",
                        (_d(collateral), user_id))
        for f in fills:
            (log.info if f.status != "REJECTED" else log.warning)(
                "gateway.fill", user_id=user_id, symbol=f.symbol, requested=f.requested, filled=f.filled,
                price=None if f.price is None else round(f.price, 2), fee=round(f.fee, 4),
                realized_pnl=round(f.realized_pnl, 2), position_after=f.position_after, status=f.status,
                reason=f.reason or None,
            )
        return tuple(fills)

    def _price(self, intent: OrderIntent, override: float | None) -> tuple[float | None, float, str]:
        """(precio, cantidad ejecutable, motivo). Recorre el L2 sombra hasta ``max_slippage_bps``."""
        if override is not None:
            return override, abs(intent.delta_quantity), ""
        snap = self.store.get(intent.symbol)
        if snap is None or snap.book is None or snap.book.mid is None:
            return None, 0.0, "sin libro"
        book = snap.book
        side = "buy" if intent.delta_quantity > 0 else "sell"
        levels = book.asks if side == "buy" else book.bids
        mid = book.mid
        limit = mid * (1 + self.config.max_slippage_bps / 1e4) if side == "buy" else mid * (1 - self.config.max_slippage_bps / 1e4)
        mask = levels[:, 0] <= limit if side == "buy" else levels[:, 0] >= limit
        available = float(levels[mask, 1].sum())
        size = min(abs(intent.delta_quantity), available)
        if size <= 0:
            return None, 0.0, "sin liquidez dentro del tope de slippage"
        vwap, filled = book.simulate_market_order(side, size)
        return vwap, filled, "" if filled >= abs(intent.delta_quantity) else "libro insuficiente"

    def _apply(self, cur: Any, account: Mapping[str, Any], user_id: int, intent: OrderIntent,
               book: dict[str, _Pos], collateral: float, leverage: float, override: float | None,
               is_manipulator: bool) -> tuple[Fill, float]:
        cfg = self.config
        sym = intent.symbol
        reject = lambda reason: (Fill(sym, intent.delta_quantity, 0.0, None, 0.0, 0.0,  # noqa: E731
                                      book[sym].quantity if sym in book else 0.0, "REJECTED", reason), collateral)

        px, size, partial_reason = self._price(intent, override)
        if px is None or size <= 0:
            return reject(partial_reason or "sin precio")
        dq = math.copysign(size, intent.delta_quantity)

        # Modo live (nunca para el manipulador)
        if self.client_resolver is not None and override is None:
            if is_manipulator:
                return reject("enrutamiento externo prohibido para MARKET_MANIPULATOR")
            px_live = self._route_live(account, intent, dq, px)
            if px_live is None:
                return reject("rechazada por el exchange")
            px = px_live

        pos = book.get(sym) or _Pos(None, 0.0, 0.0, leverage, cfg.maintenance_margin_rates.get(sym, 0.02))
        q0, new_q = pos.quantity, pos.quantity + dq
        flat_after = abs(new_q) < 1e-12
        same_side = q0 != 0 and not flat_after and math.copysign(1, new_q) == math.copysign(1, q0)
        realized = 0.0
        if q0 != 0 and math.copysign(1, dq) != math.copysign(1, q0):
            closed = math.copysign(min(abs(dq), abs(q0)), q0)
            realized = closed * (px - pos.entry)
        if q0 == 0 or (not same_side and not flat_after):
            new_entry, new_lev = px, leverage                       # apertura o giro
        elif abs(new_q) > abs(q0):
            new_entry, new_lev = (q0 * pos.entry + dq * px) / new_q, pos.leverage  # aumento
        else:
            new_entry, new_lev = pos.entry, pos.leverage            # reducción
        fee = abs(dq) * px * cfg.taker_fee_bps / 1e4
        new_collateral = collateral + realized - fee
        if new_collateral < 0:
            return reject("colateral negativo tras la operación (cuenta insolvente)")

        # Controles post-trade (solo si la orden aumenta exposición): margen inicial y límites de la cuenta
        if abs(new_q) > abs(q0) + 1e-12:
            hypothetical = {s_: (p_.quantity, p_.entry, p_.leverage) for s_, p_ in book.items()}
            hypothetical[sym] = (new_q, new_entry, new_lev)
            equity, im, gross = new_collateral, 0.0, 0.0
            for s_, (q_, e_, lev_) in hypothetical.items():
                snap = self.store.get(s_)
                m_ = snap.mark_price if snap is not None and snap.mark_price else px
                equity += q_ * (m_ - e_)
                im += abs(q_) * m_ / lev_
                gross += abs(q_) * m_
            if im > equity:
                return reject(f"margen inicial insuficiente (IM {im:,.2f} > equity {equity:,.2f})")
            breach = RiskLimits.from_row(account).breach(gross, equity)
            if breach is not None:
                return reject(str(breach))

        snap = self.store.get(sym)
        mark = snap.mark_price if snap is not None and snap.mark_price else px
        if flat_after:                                            # cierre total
            self._close(cur, pos, mark)
            book.pop(sym, None)
        elif same_side:                                           # aumento o reducción
            cur.execute(
                """UPDATE open_positions SET quantity = %s, entry_price = %s, last_mark_price = %s
                    WHERE position_id = %s""",
                (_d(new_q), _d(new_entry), _d(mark), pos.position_id),
            )
            book[sym] = _Pos(pos.position_id, new_q, new_entry, new_lev, pos.mmr)
        else:                                                     # apertura o giro
            if q0 != 0:
                self._close(cur, pos, mark)
            cur.execute(
                """INSERT INTO open_positions (user_id, symbol, quantity, entry_price, leverage,
                                               maintenance_margin_rate, last_mark_price)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING position_id""",
                (user_id, sym, _d(new_q), _d(new_entry), _d(new_lev), _d(pos.mmr), _d(mark)),
            )
            book[sym] = _Pos(cur.fetchone()["position_id"], new_q, new_entry, new_lev, pos.mmr)

        status = "FILLED" if size >= abs(intent.delta_quantity) - 1e-12 else "PARTIAL"
        return Fill(sym, intent.delta_quantity, dq, px, fee, realized, new_q, status, partial_reason), new_collateral

    @staticmethod
    def _close(cur: Any, pos: _Pos, mark: float) -> None:
        cur.execute(
            """UPDATE open_positions SET status = 'CLOSED', closed_at = now(), last_mark_price = %s
                WHERE position_id = %s""",
            (_d(mark), pos.position_id),
        )

    def _route_live(self, account: Mapping[str, Any], intent: OrderIntent, dq: float, ref_px: float) -> float | None:
        from exchange_connector.hyperliquid_client import OrderStatus, Side

        client = self.client_resolver(account)  # type: ignore[misc]
        cloid = "0x" + hashlib.blake2b(f"{account['user_id']}:{uuid.uuid4()}".encode(), digest_size=16).hexdigest()
        try:
            res = client.place_market_order(intent.symbol, Side.LONG if dq > 0 else Side.SHORT, abs(dq),
                                            slippage=self.config.live_slippage, reference_price=ref_px,
                                            reduce_only=intent.reduce_only, cloid=cloid)
        except Exception as exc:  # noqa: BLE001
            log.warning("gateway.live_error", symbol=intent.symbol, error=str(exc)[:200])
            return None
        if res.status not in (OrderStatus.FILLED,) or res.avg_fill_price is None:
            return None
        return float(res.avg_fill_price)
