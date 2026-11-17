"""
dashboard/data.py
=================

Capa de acceso a datos del dashboard (sin dependencias de Dash: testeable de forma aislada).

Toda lectura es de solo lectura y sin locks de fila (MVCC): el dashboard nunca compite con el
motor de liquidación. Los health ratios se calculan con ``risk_engine.margin_calculator`` (misma
fórmula que el motor) usando los marks que publica el orquestador en ``simulation_control``; si el
orquestador no está vivo, se recurre al último ``last_mark_price`` conocido y se avisa de ello.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ai_agents.control_plane import ControlPlane, ControlState, SimulationCommand
from database.connection import DatabaseManager
from risk_engine.margin_calculator import compute_margins, load_risk_inputs

__all__ = ["FundOverview", "AccountsHealth", "RiskOverview", "DashboardRepository", "MarketTape", "health_status",
           "risk_overview", "feed_view", "liquidation_map", "fmt_usd", "fmt_short"]

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
    adl_recovered: float = 0.0
    adl_events: int = 0

    @property
    def net_pnl(self) -> float:
        """Resultado del fondo: comisiones + ADL − déficits (excluye capitalizaciones)."""
        return self.fees_collected + self.adl_recovered - self.deficits_absorbed


@dataclass(frozen=True)
class RiskOverview:
    """Métricas de riesgo agregadas para los KPIs (sin dependencias de UI: testeable)."""

    open_interest: float                 # Σ |q|·mark de las cuentas activas
    fund_balance: float
    var: float | None                    # VaR de la cartera agregada (publicado por el orquestador)
    es: float | None
    alpha: float | None
    horizon_ticks: int | None
    var_by_account: dict[str, dict[str, float]]
    stress_coverage: dict | None         # chequeo pre-flight del orquestador

    @property
    def coverage_ratio(self) -> float | None:
        """Insurance Fund / Open Interest (los exchanges lo vigilan como % del OI)."""
        return None if self.open_interest <= 0 else self.fund_balance / self.open_interest

    @property
    def fund_to_es(self) -> float | None:
        """Cuántas veces cubre el fondo la pérdida esperada en la cola (ES) de la cartera agregada."""
        return None if not self.es or self.es <= 0 else self.fund_balance / self.es


def risk_overview(state: ControlState, accounts: pd.DataFrame, fund_balance: float) -> RiskOverview:
    risk = (state.stats or {}).get("risk") or {}
    oi = float(accounts["gross_notional"].sum()) if "gross_notional" in accounts.columns and not accounts.empty else 0.0
    return RiskOverview(
        open_interest=oi, fund_balance=fund_balance,
        var=risk.get("var") if state.is_online() else None, es=risk.get("es") if state.is_online() else None,
        alpha=risk.get("alpha"), horizon_ticks=risk.get("horizon_ticks"),
        var_by_account=dict(risk.get("by_account") or {}) if state.is_online() else {},
        stress_coverage=(state.stats or {}).get("stress_coverage"),
    )


@dataclass(frozen=True)
class AccountsHealth:
    table: pd.DataFrame
    marks: dict[str, float]
    marks_source: str           # "orchestrator" | "last_mark_price" | "none"
    marks_live: bool
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)


def liquidation_map(health: AccountsHealth, *, band: float = 0.15, bins: int = 60) -> pd.DataFrame:
    """
    Nocional que entraría en liquidación por tramo de precio (``±band`` alrededor del mark), por activo y lado.

    Usa el precio de liquidación cross-margin de cada posición (``compute_margins``, misma fórmula que el
    motor): los largos se liquidan por debajo del mark y los cortos por encima. Devuelve una fila por
    (activo, lado, tramo) con ``price_lo/price_hi/price_mid`` y ``notional`` (0 en tramos vacíos).
    """
    cols = ["symbol", "side", "price_lo", "price_hi", "price_mid", "notional", "positions", "mark"]
    pos = health.positions
    if pos.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for sym, grp in pos.groupby("symbol"):
        mark = health.marks.get(sym)
        if mark is None or not math.isfinite(mark) or mark <= 0:
            continue
        edges = np.linspace(mark * (1 - band), mark * (1 + band), bins + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        for side, mask in (("Largos", grp["quantity"] > 0), ("Cortos", grp["quantity"] < 0)):
            g = grp[mask & np.isfinite(grp["liquidation_price"])]
            idx = np.searchsorted(edges, g["liquidation_price"].to_numpy(), side="right") - 1
            inside = (idx >= 0) & (idx < bins)
            notional = np.bincount(idx[inside], weights=g["notional"].to_numpy()[inside], minlength=bins)
            count = np.bincount(idx[inside], minlength=bins)
            rows.extend({"symbol": sym, "side": side, "price_lo": edges[i], "price_hi": edges[i + 1],
                         "price_mid": mids[i], "notional": float(notional[i]), "positions": int(count[i]),
                         "mark": mark} for i in range(bins))
    return pd.DataFrame(rows, columns=cols)


def feed_view(state: ControlState) -> pd.DataFrame:
    """
    Mercado sombra (lo que ve el motor) frente a la referencia real de Hyperliquid, por activo.

    ``deviation_bps`` = (mark simulado − mark real) / mark real: ≈ 0 en Modo Normal con feed en vivo y
    se dispara durante el Cisne Negro, cuando el motor pasa a la trayectoria de cola del Monte Carlo.
    """
    feed = (state.stats or {}).get("feed") or {}
    live = feed.get("symbols") or {}
    rows = []
    for sym in sorted(set(state.market_state or {}) | set(live)):
        sim = (state.market_state or {}).get(sym) or {}
        ref = live.get(sym) or {}
        sim_mark, ref_mark = sim.get("mark"), ref.get("mark")
        dev = (sim_mark - ref_mark) / ref_mark * 1e4 if sim_mark and ref_mark else None
        rows.append({
            "symbol": sym, "sim_mark": sim_mark, "sim_oracle": sim.get("oracle"), "basis_bps": sim.get("basis_bps"),
            "spread_bps": sim.get("spread_bps"), "liquidity": sim.get("liquidity"),
            "sim_funding_hourly": sim.get("funding_hourly"), "live_mark": ref_mark, "live_oracle": ref.get("oracle"),
            "live_funding_hourly": ref.get("funding_hourly"), "deviation_bps": dev,
            "latency_ms": ref.get("latency_ms"), "age_ms": ref.get("age_ms"),
        })
    cols = ["symbol", "sim_mark", "sim_oracle", "basis_bps", "spread_bps", "liquidity", "sim_funding_hourly",
            "live_mark", "live_oracle", "live_funding_hourly", "deviation_bps", "latency_ms", "age_ms"]
    return pd.DataFrame(rows, columns=cols)


class MarketTape:
    """
    Serie temporal en memoria de precios simulados vs. reales (una muestra por heartbeat del orquestador).

    El orquestador solo publica el último estado; el dashboard reconstruye la serie muestreando cada
    refresco y deduplicando por ``heartbeat_at``. Thread-safe (callbacks concurrentes del servidor).
    """

    def __init__(self, maxlen: int = 1_800) -> None:
        self._rows: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._last: datetime | None = None
        self._lock = threading.Lock()

    def record(self, state: ControlState) -> None:
        if state.heartbeat_at is None or not state.market_state:
            return
        view = feed_view(state)
        with self._lock:
            if self._last is not None and state.heartbeat_at <= self._last:
                return
            self._last = state.heartbeat_at
            for r in view.itertuples():
                self._rows.append({"t": state.heartbeat_at, "mode": state.current_mode, "symbol": r.symbol,
                                   "sim_mark": r.sim_mark, "live_mark": r.live_mark, "deviation_bps": r.deviation_bps})

    def frame(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._rows)
        return pd.DataFrame(rows, columns=["t", "mode", "symbol", "sim_mark", "live_mark", "deviation_bps"])


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
                          COALESCE(-SUM(l.amount) FILTER (WHERE l.entry_type = 'BANKRUPTCY_DEFICIT'), 0) AS deficits,
                          COALESCE(SUM(l.amount) FILTER (WHERE l.entry_type = 'ADL_RECOVERY'), 0)      AS adl
                     FROM insurance_fund f
                     LEFT JOIN insurance_fund_ledger l ON TRUE
                    WHERE f.fund_id = 1
                    GROUP BY f.balance, f.updated_at"""
            )
            f = cur.fetchone()
            cur.execute("SELECT count(DISTINCT liquidation_batch_id) AS n, max(created_at) AS last FROM liquidation_events")
            liq = cur.fetchone()
            cur.execute("SELECT count(*) AS n FROM adl_events")
            n_adl = int(cur.fetchone()["n"])
        return FundOverview(
            balance=float(f["balance"]), updated_at=f["updated_at"], capital_injected=float(f["injected"]),
            fees_collected=float(f["fees"]), deficits_absorbed=float(f["deficits"]),
            liquidations=int(liq["n"]), last_liquidation_at=liq["last"],
            adl_recovered=float(f["adl"]), adl_events=n_adl,
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
            table["state"] = table["status"].str.split(" ", n=1).str[1]      # texto sin icono
            table["health_ratio"] = table["health_ratio"].replace([np.inf, -np.inf], np.nan)
            pos = snap.positions
            labels = pos["quantity"].map(lambda q: f"{q:+.4g}") + " " + pos["symbol"]
            exposure = labels.groupby(pos["user_id"]).agg(", ".join)
            table["positions"] = table["user_id"].map(exposure).fillna("—")
            table = table.sort_values("health_ratio", na_position="last").reset_index(drop=True)
        positions = snap.positions
        if not positions.empty and not table.empty:
            positions = positions.merge(table[["user_id", "agent_name"]], on="user_id", how="left")
        return AccountsHealth(table, marks, source, source == "orchestrator", positions)

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

    # ---------------------------------------------------------------------- ADL
    def adl_history(self, limit: int = 200) -> pd.DataFrame:
        """Una fila por ejecución de ADL, enlazada con la liquidación que la disparó."""
        df = self._frame(
            """SELECT a.adl_event_id, a.created_at, a.status, a.market_regime::text AS regime,
                      a.shortfall, a.recovered, a.residual_shortfall, a.phi, a.counterparties, a.positions,
                      a.fund_balance_before, a.fund_balance_after, a.trigger_batch_id::text AS trigger_batch_id,
                      (SELECT u.agent_name FROM liquidation_events e JOIN users_collateral u USING (user_id)
                        WHERE e.liquidation_batch_id = a.trigger_batch_id LIMIT 1) AS liquidated_agent
                 FROM adl_events a
                ORDER BY a.adl_event_id DESC
                LIMIT %s""",
            (limit,),
        )
        for col in ("shortfall", "recovered", "residual_shortfall", "phi", "fund_balance_before", "fund_balance_after"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        return df

    def adl_fills(self, limit: int = 500) -> pd.DataFrame:
        """Posiciones desapalancadas: quién pagó el ADL y cuánto."""
        df = self._frame(
            """SELECT f.adl_event_id, f.created_at, u.agent_name, f.symbol, f.quantity_before, f.quantity_closed,
                      f.entry_price, f.mark_price, f.unrealized_pnl_before, f.haircut, f.account_phi
                 FROM adl_fills f JOIN users_collateral u USING (user_id)
                ORDER BY f.adl_fill_id DESC
                LIMIT %s""",
            (limit,),
        )
        for col in ("quantity_before", "quantity_closed", "entry_price", "mark_price", "unrealized_pnl_before",
                    "haircut", "account_phi"):
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



def fmt_short(value: float | None) -> str:
    """Importe abreviado para KPIs estrechos: ``$276.3k``, ``$10.4M``."""
    if value is None or not math.isfinite(value):
        return "—"
    sign, v = ("-" if value < 0 else ""), abs(value)
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{sign}${v / div:,.1f}{unit}"
    return f"{sign}${v:,.0f}"
