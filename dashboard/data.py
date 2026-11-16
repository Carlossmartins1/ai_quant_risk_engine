"""
dashboard/data.py
=================

Capa de acceso a datos del dashboard (sin dependencias de Streamlit: testeable de forma aislada).

Toda lectura es de solo lectura y sin locks de fila (MVCC): el dashboard nunca compite con el
motor de liquidación. Los health ratios se calculan con ``risk_engine.margin_calculator`` (misma
fórmula que el motor) usando los marks que publica el orquestador en ``simulation_control``; si el
orquestador no está vivo, se recurre al último ``last_mark_price`` conocido y se avisa de ello.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ai_agents.control_plane import ControlPlane, ControlState, SimulationCommand
from database.connection import DatabaseManager
from risk_engine.margin_calculator import compute_margins, load_risk_inputs

__all__ = ["FundOverview", "AccountsHealth", "DashboardRepository", "health_status"]

HEALTH_WARNING = 1.2   # alerta temprana: < 20 % de colchón sobre el margen de mantenimiento


@dataclass(frozen=True)
class FundOverview:
    balance: float
    updated_at: datetime | None
    capital_injected: float
    fees_collected: float
    deficits_absorbed: float
    liquidations: int
    last_liquidation_at: datetime | None

    @property
    def net_pnl(self) -> float:
        """Resultado del fondo: comisiones − déficits (excluye capitalizaciones)."""
        return self.fees_collected - self.deficits_absorbed


@dataclass(frozen=True)
class AccountsHealth:
    table: pd.DataFrame
    marks: dict[str, float]
    marks_source: str           # "orchestrator" | "last_mark_price" | "none"
    marks_live: bool


def health_status(row: pd.Series) -> str:
    """Estado legible con icono + texto (nunca solo color)."""
    if row["missing_prices"]:
        return "⚪ Sin precio"
    if row["n_positions"] == 0:
        return "➖ Sin posiciones"
    if row["is_liquidatable"]:
        return "🔴 Liquidable"
    if row["health_ratio"] <= HEALTH_WARNING:
        return "🟠 En riesgo"
    return "🟢 Sana"


class DashboardRepository:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.control = ControlPlane(db, instance_id="dashboard")

    def _frame(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self.db.transaction(readonly=True) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------ Insurance Fund
    def fund_overview(self) -> FundOverview:
        with self.db.transaction(readonly=True) as cur:
            cur.execute(
                """SELECT f.balance, f.updated_at,
                          COALESCE(SUM(l.amount) FILTER (WHERE l.entry_type = 'CAPITAL_INJECTION'), 0) AS injected,
                          COALESCE(SUM(l.amount) FILTER (WHERE l.entry_type = 'LIQUIDATION_FEE'), 0)   AS fees,
                          COALESCE(-SUM(l.amount) FILTER (WHERE l.entry_type = 'BANKRUPTCY_DEFICIT'), 0) AS deficits
                     FROM insurance_fund f
                     LEFT JOIN insurance_fund_ledger l ON TRUE
                    WHERE f.fund_id = 1
                    GROUP BY f.balance, f.updated_at"""
            )
            f = cur.fetchone()
            cur.execute("SELECT count(DISTINCT liquidation_batch_id) AS n, max(created_at) AS last FROM liquidation_events")
            liq = cur.fetchone()
        return FundOverview(
            balance=float(f["balance"]), updated_at=f["updated_at"], capital_injected=float(f["injected"]),
            fees_collected=float(f["fees"]), deficits_absorbed=float(f["deficits"]),
            liquidations=int(liq["n"]), last_liquidation_at=liq["last"],
        )

    def fund_ledger(self, limit: int = 1_000) -> pd.DataFrame:
        df = self._frame(
            """SELECT * FROM (
                   SELECT l.entry_id, l.created_at, l.entry_type::text AS entry_type, l.amount, l.balance_after,
                          l.liquidation_event_id, u.agent_name, l.note
                     FROM insurance_fund_ledger l
                     LEFT JOIN liquidation_events e ON e.event_id = l.liquidation_event_id
                     LEFT JOIN users_collateral u ON u.user_id = e.user_id
                    ORDER BY l.entry_id DESC LIMIT %s) t
               ORDER BY entry_id""",
            (limit,),
        )
        for col in ("amount", "balance_after"):
            df[col] = df[col].astype(float)
        return df

    # ----------------------------------------------------------------- cuentas
    def latest_marks(self, state: ControlState | None = None, *, max_age_s: float = 10.0) -> tuple[dict[str, float], str]:
        state = state or self.control.state()
        if state.is_online(max_age_s):
            marks = {s: float(v["mark"]) for s, v in state.market_state.items()
                     if isinstance(v, dict) and v.get("mark") is not None}
            if marks:
                return marks, "orchestrator"
        df = self._frame(
            """SELECT DISTINCT ON (symbol) symbol, last_mark_price
                 FROM open_positions WHERE last_mark_price IS NOT NULL
                ORDER BY symbol, updated_at DESC"""
        )
        if df.empty:
            return {}, "none"
        return {r.symbol: float(r.last_mark_price) for r in df.itertuples()}, "last_mark_price"

    def accounts_health(self, state: ControlState | None = None) -> AccountsHealth:
        state = state or self.control.state()
        marks, source = self.latest_marks(state)
        accounts, positions = load_risk_inputs(self.db)
        snap = compute_margins(accounts, positions, marks)
        table = snap.accounts.reset_index()
        if not table.empty:
            table["status"] = table.apply(health_status, axis=1)
            table["health_ratio"] = table["health_ratio"].replace([np.inf, -np.inf], np.nan)
            pos = snap.positions
            labels = pos["quantity"].map(lambda q: f"{q:+.4g}") + " " + pos["symbol"]
            exposure = labels.groupby(pos["user_id"]).agg(", ".join)
            table["positions"] = table["user_id"].map(exposure).fillna("—")
            table = table.sort_values("health_ratio", na_position="last").reset_index(drop=True)
        return AccountsHealth(table, marks, source, source == "orchestrator")

    def account_status_counts(self) -> dict[str, int]:
        df = self._frame("SELECT account_status::text AS status, count(*) AS n FROM users_collateral GROUP BY 1")
        return {r.status: int(r.n) for r in df.itertuples()}

    # ------------------------------------------------------------ liquidaciones
    def liquidations(self, limit: int = 500) -> pd.DataFrame:
        """Una fila por liquidación de cuenta (batch), con métricas de déficit y slippage."""
        df = self._frame(
            """SELECT e.liquidation_batch_id::text                    AS batch_id,
                      max(e.created_at)                                AS liquidated_at,
                      e.user_id, u.agent_name, u.agent_profile::text   AS agent_profile,
                      e.market_regime::text                            AS regime,
                      count(*)                                         AS positions,
                      string_agg(e.symbol, ', ' ORDER BY e.symbol)     AS symbols,
                      max(e.account_equity_before)                     AS equity_before,
                      max(e.maintenance_margin_before)                 AS maintenance_margin_before,
                      max(e.margin_ratio_before)                       AS health_before,
                      sum(e.realized_pnl)                              AS realized_pnl,
                      sum(e.liquidation_fee)                           AS fee,
                      sum(e.deficit)                                   AS deficit,
                      sum(e.insurance_fund_delta)                      AS fund_delta,
                      max(e.slippage_bps)                              AS max_slippage_bps,
                      sum(e.slippage_bps * abs(e.quantity) * e.execution_price)
                        / NULLIF(sum(abs(e.quantity) * e.execution_price), 0) AS wavg_slippage_bps,
                      string_agg(DISTINCT e.metadata->>'source', ', ') AS execution_source
                 FROM liquidation_events e
                 JOIN users_collateral u USING (user_id)
                GROUP BY e.liquidation_batch_id, e.user_id, u.agent_name, u.agent_profile, e.market_regime
                ORDER BY liquidated_at DESC
                LIMIT %s""",
            (limit,),
        )
        num = ["equity_before", "maintenance_margin_before", "health_before", "realized_pnl", "fee", "deficit",
               "fund_delta", "max_slippage_bps", "wavg_slippage_bps"]
        for col in num:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        return df

    # ----------------------------------------------------------- plano de control
    def control_state(self) -> ControlState:
        return self.control.state()

    def request(self, command: SimulationCommand, requested_by: str) -> int:
        return self.control.enqueue(command, requested_by=requested_by, note="dashboard")

    def recent_commands(self, limit: int = 15) -> pd.DataFrame:
        return self._frame(
            """SELECT command_id, command::text AS command, requested_by, requested_at, consumed_at, consumed_by
                 FROM simulation_commands ORDER BY command_id DESC LIMIT %s""",
            (limit,),
        )


def fmt_usd(value: float | None, *, compact: bool = False) -> str:
    """``compact``: sin céntimos a partir de 10 000 USD (KPIs con espacio limitado)."""
    if value is None or not math.isfinite(value):
        return "—"
    sign = "-" if value < 0 else ""
    if compact and abs(value) >= 10_000:
        return f"{sign}${abs(value):,.0f}"
    return f"{sign}${abs(value):,.2f}"

