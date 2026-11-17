"""
risk_engine/adl.py
==================

Auto-Deleveraging (ADL): última línea de defensa cuando el Insurance Fund no cubre el bad debt.

Por qué
-------
Una liquidación que se ejecuta peor que el precio de bancarrota deja un déficit que absorbe el
Insurance Fund. Si el fondo queda en negativo, el exchange es insolvente frente a los ganadores:
alguien tiene que pagar. El ADL socializa ese déficit entre las posiciones ganadoras en lugar de
dejarlo como una alerta pasiva.

Modelo (pro-rata con water-filling)
-----------------------------------
``S = −balance`` del fondo. Candidatos: posiciones OPEN con PnL no realizado ``u_i > 0`` de cuentas
ACTIVE sanas (``equity > MM``). Cada cuenta ``a`` cierra la misma fracción ``φ_a`` de TODAS sus
posiciones ganadoras **al precio de entrada**: el beneficio de la parte cerrada, ``φ_a·u_i``, se
transfiere al fondo; el principal nunca se toca (el colateral de la cuenta no cambia).

::

    U_a  = Σ_{i∈a, u_i>0} u_i            MMw_a = MM de esas posiciones
    φ_a  = min(φ, φmax_a)                Σ_a φ_a · U_a = S   (φ mínimo que cubre S)

``φmax_a`` impide que el ADL deje a una cuenta con piernas perdedoras (cross-margin) por debajo de
un health ratio mínimo ``h`` (1.10 por defecto; ``h = 1`` sería dejarla justo en el umbral de
liquidación):

::

    E'_a − h·MM'_a = (E_a − h·MM_a) − φ_a·(U_a − h·MMw_a) ≥ 0
    ⇒  φmax_a = (E_a − h·MM_a) / (U_a − h·MMw_a)   si U_a > h·MMw_a ;  1 en otro caso ;  0 si E_a < h·MM_a

``Σ_a min(φ, φmax_a)·U_a`` es lineal a trozos y creciente en ``φ``: se resuelve de forma exacta
ordenando los ``φmax_a`` (sin bisección). Si ni con ``φ = 1`` se cubre ``S``, el ADL es
``PARTIAL`` y el residual se reintenta en ciclos posteriores; sin candidatos, ``NO_COUNTERPARTIES``
(también se registra: un fondo en negativo nunca pasa sin dejar rastro).

Propiedades: la misma fracción para todos (justo e independiente del orden de procesamiento), el
principal nunca se toca, recuperación acotada por el beneficio no realizado total y sin cascadas
inducidas (ninguna cuenta queda bajo mantenimiento por el ADL).

Concurrencia
------------
Transacción propia, posterior al COMMIT de la liquidación, con el orden de locks canónico:

    users_collateral (ORDER BY user_id, SKIP LOCKED) → open_positions (ORDER BY position_id) → insurance_fund

``SKIP LOCKED``: una cuenta que el gateway o el liquidador tienen bloqueada en ese instante no se
espera (sin deadlocks ni latencia en cola); queda fuera de esta ronda y el residual se reintenta.
El saldo del fondo se relee bajo lock: si otro proceso ya lo repuso, el ADL no hace nada.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd
import psycopg2.extras
import structlog

from database.connection import DatabaseManager, LockAcquisitionError, lock_insurance_fund, retry_on_transient
from risk_engine.margin_calculator import compute_margins

if TYPE_CHECKING:
    from exchange_connector.hyperliquid_ws import MarketDataStore

__all__ = ["AdlConfig", "AdlPlan", "AdlFill", "AdlOutcome", "AdlEngine", "plan_adl"]

log = structlog.get_logger(__name__)

Q = Decimal("0.00000001")
ZERO = Decimal(0)
PHI_Q = Decimal("0.000000000001")


def _d(x: Any) -> Decimal:
    return (x if isinstance(x, Decimal) else Decimal(repr(float(x)))).quantize(Q, rounding=ROUND_HALF_EVEN)


# =============================================================================
# Planificador (puro, sin BD)
# =============================================================================
@dataclass(frozen=True)
class AdlPlan:
    shortfall: float
    phi: float                              # fracción común (antes de los topes por cuenta)
    account_phi: Mapping[int, float]        # φ_a efectiva por cuenta (> 0)
    haircuts: np.ndarray                    # beneficio transferido por fila candidata
    recovered: float

    @property
    def residual(self) -> float:
        return max(self.shortfall - self.recovered, 0.0)

    @property
    def status(self) -> str:
        if self.recovered <= 0:
            return "NO_COUNTERPARTIES"
        return "COVERED" if self.residual <= 1e-6 else "PARTIAL"


def plan_adl(shortfall: float, candidates: pd.DataFrame, *, min_health: float = 1.10) -> AdlPlan:
    """
    ``candidates``: una fila por posición ganadora con columnas ``user_id, unrealized_pnl (>0),
    maintenance_margin, account_equity, account_mm``. Devuelve el reparto de mínimo ``φ``.
    """
    if not shortfall > 0:
        raise ValueError("shortfall debe ser > 0")
    if min_health < 1:
        raise ValueError("min_health debe ser >= 1")
    n = len(candidates)
    if n == 0:
        return AdlPlan(shortfall, 0.0, {}, np.zeros(0), 0.0)
    uid = candidates["user_id"].to_numpy(dtype=np.int64)
    u = candidates["unrealized_pnl"].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(u)) or np.any(u <= 0):
        raise ValueError("los candidatos deben tener PnL no realizado finito y > 0")

    accounts, inv = np.unique(uid, return_inverse=True)
    U = np.bincount(inv, weights=u)
    MMw = np.bincount(inv, weights=candidates["maintenance_margin"].to_numpy(dtype=np.float64))
    first = np.unique(inv, return_index=True)[1]
    buffer = (candidates["account_equity"].to_numpy(dtype=np.float64)
              - min_health * candidates["account_mm"].to_numpy(dtype=np.float64))[first]
    denom = U - min_health * MMw
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_max = np.where(denom > 0, buffer / denom, 1.0)
    phi_max = np.where(buffer > 0, np.clip(np.nan_to_num(phi_max, nan=0.0), 0.0, 1.0), 0.0)

    capacity = float(phi_max @ U)
    if capacity <= shortfall:                                  # ni con φ = 1 se cubre: todos a su máximo
        phi, phi_a = 1.0, phi_max
    else:                                                       # water-filling exacto
        order = np.argsort(phi_max, kind="stable")
        fixed, rest, phi = 0.0, float(U.sum()), 1.0
        for k in order:
            candidate = (shortfall - fixed) / rest
            if candidate <= phi_max[k]:
                phi = candidate
                break
            fixed += phi_max[k] * U[k]
            rest -= U[k]
        phi_a = np.minimum(phi, phi_max)

    haircuts = phi_a[inv] * u
    recovered = float(min(haircuts.sum(), shortfall))
    account_phi = {int(a): float(p) for a, p in zip(accounts, phi_a) if p > 0}
    return AdlPlan(shortfall, float(phi), account_phi, haircuts, recovered)


# =============================================================================
# Motor transaccional
# =============================================================================
@dataclass(frozen=True, slots=True)
class AdlConfig:
    lock_timeout_ms: int = 1_000
    statement_timeout_ms: int = 10_000
    max_price_age_s: float | None = 5.0
    max_retries: int = 3
    min_health: float = 1.10               # health ratio mínimo que el ADL respeta en las contrapartes
    min_retry_recovery_usd: float = 100.0  # reintentos periódicos: no registrar recuperaciones marginales


@dataclass(frozen=True, slots=True)
class AdlFill:
    user_id: int
    position_id: int
    symbol: str
    quantity_before: Decimal
    quantity_closed: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl_before: Decimal
    haircut: Decimal
    account_phi: float


@dataclass(frozen=True)
class AdlOutcome:
    adl_batch_id: uuid.UUID
    trigger_batch_id: uuid.UUID | None
    status: str
    shortfall: Decimal
    recovered: Decimal
    residual: Decimal
    phi: float
    fund_before: Decimal
    fund_after: Decimal
    fills: tuple[AdlFill, ...] = ()
    event_id: int | None = None

    @property
    def counterparties(self) -> int:
        return len({f.user_id for f in self.fills})

    def as_dict(self) -> dict[str, Any]:
        return {"adl_batch_id": str(self.adl_batch_id), "status": self.status,
                "trigger_batch_id": str(self.trigger_batch_id) if self.trigger_batch_id else None,
                "shortfall": str(self.shortfall), "recovered": str(self.recovered), "residual": str(self.residual),
                "phi": round(self.phi, 6), "counterparties": self.counterparties, "positions": len(self.fills),
                "fund_before": str(self.fund_before), "fund_after": str(self.fund_after)}


class AdlEngine:
    """Ejecuta el ADL cuando el Insurance Fund está en negativo. Ver docstring del módulo."""

    def __init__(self, db: DatabaseManager, store: "MarketDataStore", config: AdlConfig | None = None, *,
                 regime: str = "LIVE_TESTNET") -> None:
        self.db = db
        self.store = store
        self.config = config or AdlConfig()
        self.regime = regime

    def fund_balance(self) -> Decimal:
        with self.db.transaction(readonly=True) as cur:
            cur.execute("SELECT balance FROM insurance_fund WHERE fund_id = 1")
            return Decimal(cur.fetchone()["balance"])

    def run(self, *, trigger_batch_id: uuid.UUID | None = None) -> AdlOutcome | None:
        """
        ``None`` si el fondo no está en negativo (o no hay nada que hacer ni registrar). Con
        ``trigger_batch_id`` (llamada tras una liquidación que agotó el fondo) SIEMPRE registra un
        ``adl_events``, aunque no haya contrapartes. Sin él (reintento periódico de un residual) solo
        registra si recupera algo, para no llenar la auditoría de filas vacías.
        """
        if self.fund_balance() >= 0:          # lectura barata sin lock: el caso normal sale aquí
            return None
        attempt = retry_on_transient(max_attempts=self.config.max_retries)(self._run_tx)
        try:
            outcome = attempt(trigger_batch_id)
        except LockAcquisitionError:
            log.warning("adl.skipped_locked", trigger_batch_id=str(trigger_batch_id) if trigger_batch_id else None)
            return None
        if outcome is not None:
            level = log.warning if outcome.status == "COVERED" else log.critical
            level("adl.executed", **outcome.as_dict())
        return outcome

    def _fresh_marks(self, symbols: set[str]) -> dict[str, float]:
        marks: dict[str, float] = {}
        for s in symbols:
            snap = self.store.get(s)
            if snap is None or not snap.mark_price or snap.mark_price <= 0:
                continue
            if self.config.max_price_age_s is not None and not self.store.is_fresh(s, self.config.max_price_age_s):
                continue
            marks[s] = float(snap.mark_price)
        return marks

    def _run_tx(self, trigger_batch_id: uuid.UUID | None) -> AdlOutcome | None:
        cfg = self.config
        with self.db.transaction(lock_timeout_ms=cfg.lock_timeout_ms, statement_timeout_ms=cfg.statement_timeout_ms) as cur:
            # 1) Cuentas candidatas, orden canónico, sin esperar a las que estén operando/liquidándose
            cur.execute(
                """SELECT uc.user_id, uc.deposited_collateral
                     FROM users_collateral uc
                    WHERE uc.account_status = 'ACTIVE'
                      AND EXISTS (SELECT 1 FROM open_positions op
                                   WHERE op.user_id = uc.user_id AND op.status = 'OPEN')
                    ORDER BY uc.user_id
                      FOR UPDATE SKIP LOCKED"""
            )
            accounts = [dict(r) for r in cur.fetchall()]
            positions: list[dict[str, Any]] = []
            if accounts:
                # 2) Sus posiciones (orden de PK)
                cur.execute(
                    """SELECT position_id, user_id, symbol, quantity, entry_price, maintenance_margin_rate, leverage
                         FROM open_positions
                        WHERE user_id = ANY(%s) AND status = 'OPEN'
                        ORDER BY position_id
                          FOR UPDATE""",
                    ([a["user_id"] for a in accounts],),
                )
                positions = [dict(r) for r in cur.fetchall()]

            # 3) Insurance Fund: último lock; se relee bajo lock
            fund_before = Decimal(lock_insurance_fund(cur)["balance"])
            if fund_before >= 0:
                return None
            shortfall = -fund_before

            cands = self._candidates(accounts, positions)
            plan = (plan_adl(float(shortfall), cands, min_health=cfg.min_health) if not cands.empty
                    else AdlPlan(float(shortfall), 0.0, {}, np.zeros(0), 0.0))
            fills = self._fills(cands, plan, shortfall)
            recovered = sum((f.haircut for f in fills), ZERO)
            if trigger_batch_id is None and recovered < Decimal(repr(cfg.min_retry_recovery_usd)) and recovered < shortfall:
                return None    # reintento periódico sin nada material que recuperar: no se toca nada
            residual = shortfall - recovered
            status = "NO_COUNTERPARTIES" if recovered <= 0 else ("COVERED" if residual == 0 else "PARTIAL")
            fund_after = fund_before + recovered
            phi = Decimal(repr(plan.phi)).quantize(PHI_Q) if fills else ZERO
            adl_batch_id = uuid.uuid4()

            # 4) Posiciones: reducción a q·(1 − φ_a) o cierre total; el colateral no cambia
            for f in fills:
                remaining = f.quantity_before - f.quantity_closed
                if remaining == 0:
                    cur.execute("""UPDATE open_positions SET status = 'CLOSED', closed_at = now(), last_mark_price = %s
                                    WHERE position_id = %s AND status = 'OPEN'""", (f.mark_price, f.position_id))
                else:
                    cur.execute("""UPDATE open_positions SET quantity = %s, last_mark_price = %s
                                    WHERE position_id = %s AND status = 'OPEN'""",
                                (remaining, f.mark_price, f.position_id))
                if cur.rowcount != 1:
                    raise RuntimeError(f"Posición {f.position_id} no estaba OPEN bajo lock (invariante roto)")

            # 5) Auditoría inmutable
            cur.execute(
                """INSERT INTO adl_events (adl_batch_id, trigger_batch_id, status, shortfall, recovered,
                                           residual_shortfall, phi, counterparties, positions, fund_balance_before,
                                           fund_balance_after, market_regime, metadata)
                   VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::market_regime, %s)
                   RETURNING adl_event_id""",
                (str(adl_batch_id), str(trigger_batch_id) if trigger_batch_id else None, status, shortfall, recovered,
                 residual, phi, len({f.user_id for f in fills}), len(fills), fund_before, fund_after, self.regime,
                 psycopg2.extras.Json({"model": "pro_rata_water_filling", "candidates": int(len(cands)),
                                       "accounts_locked": len(accounts),
                                       "capacity": round(float(cands["unrealized_pnl"].sum()), 2) if len(cands) else 0.0})),
            )
            event_id = int(cur.fetchone()["adl_event_id"])
            if fills:
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO adl_fills (adl_event_id, user_id, position_id, symbol, quantity_before,
                                              quantity_closed, entry_price, mark_price, unrealized_pnl_before,
                                              haircut, account_phi)
                       VALUES %s""",
                    [(event_id, f.user_id, f.position_id, f.symbol, f.quantity_before, f.quantity_closed, f.entry_price,
                      f.mark_price, f.unrealized_pnl_before, f.haircut, Decimal(repr(f.account_phi)).quantize(PHI_Q))
                     for f in fills],
                )
            if recovered > 0:
                cur.execute(
                    """INSERT INTO insurance_fund_ledger (entry_type, amount, balance_after, note)
                       VALUES ('ADL_RECOVERY', %s, %s, %s)""",
                    (recovered, fund_after, f"ADL {adl_batch_id} φ={float(phi):.6f} ({status})"),
                )
                cur.execute("UPDATE insurance_fund SET balance = %s WHERE fund_id = 1", (fund_after,))

            return AdlOutcome(adl_batch_id, trigger_batch_id, status, shortfall, recovered, residual, float(phi),
                              fund_before, fund_after, tuple(fills), event_id)

    def _candidates(self, accounts: list[dict[str, Any]], positions: list[dict[str, Any]]) -> pd.DataFrame:
        cols = ["user_id", "position_id", "symbol", "quantity", "entry_price", "mark_price", "unrealized_pnl",
                "maintenance_margin", "account_equity", "account_mm"]
        if not positions:
            return pd.DataFrame(columns=cols)
        marks = self._fresh_marks({p["symbol"] for p in positions})
        pos = pd.DataFrame(positions).astype({"quantity": float, "entry_price": float,
                                              "maintenance_margin_rate": float, "leverage": float})
        snap = compute_margins(pd.DataFrame(accounts).astype({"deposited_collateral": float}), pos, marks, statuses=None)
        acc = snap.accounts
        healthy = acc.index[np.isfinite(acc["equity"]) & (acc["equity"] > acc["maintenance_margin"])]
        p = snap.positions
        p = p[(p["unrealized_pnl"] > 0) & p["user_id"].isin(healthy) & np.isfinite(p["mark_price"])].copy()
        p["account_equity"] = acc.loc[p["user_id"], "equity"].to_numpy()
        p["account_mm"] = acc.loc[p["user_id"], "maintenance_margin"].to_numpy()
        return p[cols].reset_index(drop=True)

    @staticmethod
    def _fills(cands: pd.DataFrame, plan: AdlPlan, shortfall: Decimal) -> list[AdlFill]:
        """Cantidades y haircuts exactos (Decimal). La suma de haircuts nunca excede el déficit."""
        rows = []
        for i, r in enumerate(cands.itertuples(index=False)):
            phi_a = plan.account_phi.get(int(r.user_id), 0.0)
            if phi_a <= 0:
                continue
            q = _d(r.quantity)
            closed = q if phi_a >= 1.0 else _d(r.quantity * phi_a)
            if closed == 0:
                continue
            rows.append((r, i, q, closed, phi_a))
        if not rows:
            return []
        weights = [float(plan.haircuts[i]) for _, i, *_ in rows]
        total = min(shortfall, _d(sum(weights)))
        parts = _allocate(total, weights)
        fills = []
        for (r, _, q, closed, phi_a), h in zip(rows, parts):
            upnl = _d(r.unrealized_pnl)
            fills.append(AdlFill(int(r.user_id), int(r.position_id), str(r.symbol), q, closed, _d(r.entry_price),
                                 _d(r.mark_price), upnl, max(min(h, upnl), ZERO), phi_a))
        return fills


def _allocate(total: Decimal, weights: list[float]) -> list[Decimal]:
    """Reparte ``total`` pro-rata por ``weights`` con suma exacta (residuo de redondeo en la última fila)."""
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))
    parts = [_d(total * Decimal(repr(float(x)))) for x in w[:-1]]
    parts.append(total - sum(parts, ZERO))
    return parts

