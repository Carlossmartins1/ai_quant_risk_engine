"""
risk_engine/margin_calculator.py
================================

Shadow Risk Engine: cálculo vectorizado del estado de margen cross de TODAS las cuentas.

Modelo (idéntico al de ``database/schema.sql`` y a ``quant_models.monte_carlo.margin_stress``)
-------------------------------------------------------------------------------------------
Para cada posición *i* (cantidad con signo ``q``: +long / −short)::

    uPnL_i     = q_i · (mark_i − entry_i)
    Notional_i = |q_i| · mark_i
    MM_i       = Notional_i · MMR_i                    (margen de mantenimiento)
    IM_i       = Notional_i / leverage_i               (margen inicial consumido)

Para cada cuenta *u* (netting cross-margin: todas sus posiciones comparten colateral)::

    Equity_u   = deposited_collateral_u + Σ_{i∈u} uPnL_i
    MM_u       = Σ_{i∈u} MM_i
    Buffer_u   = Equity_u − MM_u                       (colchón en USD; ≤ 0 => liquidable)
    liquidable  <=>  Equity_u ≤ MM_u

Ratios (se exponen ambas convenciones de la industria para evitar ambigüedad)::

    margin_ratio = MM / Equity     (convención Hyperliquid:   ≥ 1 => liquidable)
    health_ratio = Equity / MM     (convención del esquema:   ≤ 1 => liquidable)

Precios por posición con margen cruzado (resto de la cartera congelado en el mark actual)::

    P_liq,i = (MM_{−i} − Eq_{−i} + q_i·entry_i) / (q_i − |q_i|·MMR_i)
    P_bk,i  = entry_i − Eq_{−i} / q_i                  (precio de bancarrota: Equity = 0)
    donde Eq_{−i} = Equity − uPnL_i  y  MM_{−i} = MM − MM_i

Rendimiento
-----------
Sin ``groupby`` ni bucles: las agregaciones por usuario son ``np.bincount`` sobre un índice entero
(O(n), en C). 300 000 posiciones / 100 000 cuentas se evalúan en ~20-40 ms.

Seguridad operativa
-------------------
**Fail-safe ante datos de mercado ausentes o rancios**: si falta el mark de cualquier activo de una
cuenta, su equity queda en NaN, se marca ``missing_prices`` y NUNCA se considera liquidable. Liquidar
con un precio obsoleto es peor que retrasar una liquidación unos segundos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray

from database.connection import DatabaseManager, IsolationLevel

if TYPE_CHECKING:
    from exchange_connector.hyperliquid_ws import MarketDataStore

__all__ = [
    "MarginSnapshot",
    "compute_margins",
    "marks_from_store",
    "load_risk_inputs",
    "MarginCalculator",
    "ACCOUNT_COLUMNS",
    "POSITION_COLUMNS",
]

log = structlog.get_logger(__name__)
FloatArray = NDArray[np.float64]

ACCOUNT_COLUMNS = ("user_id", "deposited_collateral")
POSITION_COLUMNS = ("position_id", "user_id", "symbol", "quantity", "entry_price", "maintenance_margin_rate")
MONITORED_STATUSES = ("ACTIVE", "LIQUIDATING")


# =============================================================================
# Resultado
# =============================================================================
@dataclass(frozen=True)
class MarginSnapshot:
    """Fotografía inmutable del riesgo de margen en un instante."""

    accounts: pd.DataFrame          # índice user_id
    positions: pd.DataFrame         # una fila por posición abierta evaluada
    mark_prices: Mapping[str, float]
    computed_at: float = field(default_factory=time.time)
    compute_ms: float = 0.0

    def liquidatable(self) -> pd.DataFrame:
        """Cuentas liquidables, de la más deteriorada (menor health_ratio) a la menos."""
        return self.accounts[self.accounts["is_liquidatable"]].sort_values("health_ratio")

    def at_risk(self, health_threshold: float = 1.2) -> pd.DataFrame:
        """Cuentas con health_ratio por debajo de un umbral de alerta temprana (incluye liquidables)."""
        acc = self.accounts
        return acc[(acc["n_positions"] > 0) & (acc["health_ratio"] <= health_threshold)].sort_values("health_ratio")

    def account(self, user_id: int) -> pd.Series:
        return self.accounts.loc[user_id]

    def summary(self) -> dict[str, Any]:
        acc = self.accounts
        return {
            "accounts": int(len(acc)),
            "positions": int(len(self.positions)),
            "liquidatable": int(acc["is_liquidatable"].sum()),
            "missing_prices": int(acc["missing_prices"].sum()),
            "total_equity": float(np.nansum(acc["equity"])),
            "total_maintenance_margin": float(np.nansum(acc["maintenance_margin"])),
            "total_unrealized_pnl": float(np.nansum(acc["unrealized_pnl"])),
            "compute_ms": round(self.compute_ms, 3),
        }


# =============================================================================
# Núcleo vectorizado
# =============================================================================
def _require(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}")


def compute_margins(
    accounts: pd.DataFrame,
    positions: pd.DataFrame,
    mark_prices: Mapping[str, float] | pd.Series,
    *,
    statuses: Iterable[str] | None = MONITORED_STATUSES,
) -> MarginSnapshot:
    """
    Evalúa margen, equity y liquidabilidad de todas las cuentas en una sola pasada vectorizada.

    Args:
        accounts: columnas ``user_id`` (o índice), ``deposited_collateral`` y opcionalmente
            ``account_status``, ``agent_name``, ``agent_profile``.
        positions: columnas ``position_id, user_id, symbol, quantity, entry_price,
            maintenance_margin_rate`` y opcionalmente ``leverage``. Solo posiciones OPEN.
        mark_prices: ``{symbol: mark}``. Símbolos ausentes o no positivos => precio desconocido.
        statuses: estados de cuenta a evaluar (``None`` = todos).

    Returns:
        ``MarginSnapshot`` con un DataFrame por cuenta y otro por posición.
    """
    t0 = time.perf_counter()
    acc = accounts.set_index("user_id") if "user_id" in accounts.columns else accounts.copy()
    acc.index = acc.index.astype(np.int64)
    acc.index.name = "user_id"
    _require(acc, ACCOUNT_COLUMNS[1:], "accounts")
    _require(positions, POSITION_COLUMNS, "positions")
    if statuses is not None and "account_status" in acc.columns:
        acc = acc[acc["account_status"].astype(str).isin(tuple(statuses))]
    if acc.index.has_duplicates:
        raise ValueError("accounts contiene user_id duplicados")

    # -- alinear posiciones con cuentas (índice entero denso para bincount) -------------------
    idx_all = acc.index.get_indexer(positions["user_id"].astype(np.int64))
    keep = idx_all >= 0
    pos = positions.loc[keep].reset_index(drop=True)
    idx = idx_all[keep]
    U = len(acc)

    q = pos["quantity"].to_numpy(dtype=np.float64)
    entry = pos["entry_price"].to_numpy(dtype=np.float64)
    mmr = pos["maintenance_margin_rate"].to_numpy(dtype=np.float64)
    lev = pos["leverage"].to_numpy(dtype=np.float64) if "leverage" in pos.columns else np.full(len(pos), np.nan)
    marks = pd.Series(mark_prices, dtype=np.float64)
    marks = marks.where(marks > 0)                                         # no positivos => NaN
    mark = marks.reindex(pos["symbol"].astype(str)).to_numpy(dtype=np.float64)

    # -- métricas por posición ---------------------------------------------------------------
    upnl = q * (mark - entry)
    notional = np.abs(q) * mark
    mm = notional * mmr
    im = notional / lev

    def agg(weights: FloatArray) -> FloatArray:
        # astype: con 0 posiciones numpy devuelve int64 aunque se pasen pesos float
        return np.bincount(idx, weights=weights, minlength=U).astype(np.float64, copy=False)

    unknown = np.isnan(mark)
    missing = agg(unknown.astype(np.float64)) > 0
    n_pos = np.bincount(idx, minlength=U)
    upnl_u = agg(np.where(unknown, 0.0, upnl))
    mm_u = agg(np.where(unknown, 0.0, mm))
    im_u = agg(np.where(unknown | np.isnan(im), 0.0, im))
    gross_u = agg(np.where(unknown, 0.0, notional))
    net_u = agg(np.where(unknown, 0.0, q * mark))

    collateral = acc["deposited_collateral"].to_numpy(dtype=np.float64)
    equity = collateral + upnl_u
    for arr in (equity, mm_u, upnl_u):
        arr[missing] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        buffer = equity - mm_u
        health = np.where(mm_u > 0, equity / mm_u, np.inf)
        margin_ratio = np.where(mm_u > 0, np.where(equity > 0, mm_u / equity, np.inf), 0.0)
        eff_lev = np.where(equity > 0, gross_u / equity, np.inf)
    health[missing] = np.nan
    margin_ratio[missing] = np.nan
    liquidatable = (n_pos > 0) & ~missing & (equity <= mm_u)

    # -- precios de liquidación y bancarrota por posición (cross margin) ----------------------
    other_eq = equity[idx] - upnl
    other_mm = mm_u[idx] - mm
    with np.errstate(divide="ignore", invalid="ignore"):
        liq = (other_mm - other_eq + q * entry) / (q - np.abs(q) * mmr)
        liq = np.where(np.isfinite(liq) & (liq > 0), liq, np.nan)
        bk = np.maximum(entry - other_eq / q, 0.0)
        liq_dist = np.sign(q) * (mark - liq) / mark

    positions_out = pd.DataFrame(
        {
            "position_id": pos["position_id"].to_numpy(),
            "user_id": pos["user_id"].to_numpy(dtype=np.int64),
            "symbol": pos["symbol"].astype(str).to_numpy(),
            "quantity": q,
            "entry_price": entry,
            "mark_price": mark,
            "notional": notional,
            "unrealized_pnl": upnl,
            "maintenance_margin": mm,
            "initial_margin": im,
            "liquidation_price": liq,
            "bankruptcy_price": bk,
            "liq_distance_pct": liq_dist,
        }
    )

    accounts_out = pd.DataFrame(
        {
            "collateral": collateral,
            "unrealized_pnl": upnl_u,
            "equity": equity,
            "maintenance_margin": mm_u,
            "initial_margin": im_u,
            "margin_buffer": buffer,
            "margin_ratio": margin_ratio,
            "health_ratio": health,
            "gross_notional": gross_u,
            "net_notional": net_u,
            "effective_leverage": eff_lev,
            "n_positions": n_pos,
            "missing_prices": missing,
            "is_liquidatable": liquidatable,
        },
        index=acc.index,
    )
    for col in ("agent_name", "agent_profile", "account_status"):
        if col in acc.columns:
            accounts_out.insert(0, col, acc[col].to_numpy())

    return MarginSnapshot(
        accounts=accounts_out,
        positions=positions_out,
        mark_prices=dict(marks.dropna()),
        compute_ms=(time.perf_counter() - t0) * 1e3,
    )


# =============================================================================
# Fuentes de datos
# =============================================================================
def marks_from_store(
    store: "MarketDataStore", symbols: Iterable[str], *, max_age_s: float | None = 5.0
) -> dict[str, float]:
    """
    Mark prices del ``MarketDataStore`` (WebSocket). Solo incluye activos con datos frescos:
    los que falten provocarán ``missing_prices`` (fail-safe) en ``compute_margins``.
    ``max_age_s=None`` desactiva el control de frescura (replays / modo estrés).
    """
    marks: dict[str, float] = {}
    for sym in set(symbols):
        snap = store.get(sym)
        if snap is None or snap.mark_price is None:
            continue
        if max_age_s is not None and not store.is_fresh(sym, max_age_s):
            continue
        marks[sym] = float(snap.mark_price)
    return marks


_ACCOUNTS_SQL = """
    SELECT user_id, agent_name, agent_profile::text AS agent_profile, account_status::text AS account_status,
           deposited_collateral
      FROM users_collateral
     WHERE account_status IN ('ACTIVE', 'LIQUIDATING')
"""
_POSITIONS_SQL = """
    SELECT position_id, user_id, symbol, quantity, entry_price, leverage, maintenance_margin_rate
      FROM open_positions
     WHERE status = 'OPEN'
"""


def load_risk_inputs(db: DatabaseManager) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee cuentas y posiciones abiertas SIN bloquear filas (MVCC) en una transacción
    REPEATABLE READ de solo lectura: ambas consultas ven exactamente el mismo snapshot
    (no puede aparecer una posición de una cuenta que la primera consulta no vio).
    """
    with db.transaction(isolation=IsolationLevel.REPEATABLE_READ, readonly=True) as cur:
        cur.execute(_ACCOUNTS_SQL)
        accounts = pd.DataFrame(cur.fetchall(), columns=["user_id", "agent_name", "agent_profile",
                                                         "account_status", "deposited_collateral"])
        cur.execute(_POSITIONS_SQL)
        positions = pd.DataFrame(cur.fetchall(), columns=["position_id", "user_id", "symbol", "quantity",
                                                          "entry_price", "leverage", "maintenance_margin_rate"])
    # NUMERIC -> Decimal en psycopg2: a float64 para el cómputo vectorizado
    for col in ("deposited_collateral",):
        accounts[col] = accounts[col].astype(np.float64)
    for col in ("quantity", "entry_price", "leverage", "maintenance_margin_rate"):
        positions[col] = positions[col].astype(np.float64)
    return accounts, positions


class MarginCalculator:
    """Fachada: base de datos + feed de mercado -> ``MarginSnapshot``."""

    def __init__(self, db: DatabaseManager, store: "MarketDataStore", *, max_price_age_s: float | None = 5.0) -> None:
        self.db = db
        self.store = store
        self.max_price_age_s = max_price_age_s

    def snapshot(self) -> MarginSnapshot:
        accounts, positions = load_risk_inputs(self.db)
        marks = marks_from_store(self.store, positions["symbol"].unique(), max_age_s=self.max_price_age_s)
        snap = compute_margins(accounts, positions, marks)
        summary = snap.summary()
        if summary["missing_prices"]:
            log.warning("margin.missing_prices", accounts=summary["missing_prices"], marks=sorted(marks))
        log.debug("margin.snapshot", **summary)
        return snap
