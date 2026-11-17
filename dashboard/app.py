"""
dashboard/app.py
================

Centro de control en tiempo real del AI-Driven Quant Risk & Liquidation Engine (Streamlit).

Paneles
-------
* **KPIs**: Insurance Fund, cobertura del fondo frente al Open Interest, VaR/ES 99 % de la cartera agregada,
  liquidaciones, Auto-Deleveraging y cuentas en riesgo.
* **Insurance Fund**: evolución del balance (libro mayor) y movimientos.
* **Cuentas**: health ratio en tiempo real de cada cuenta (misma fórmula que el motor de riesgo) y VaR/ES por cuenta.
* **Liquidaciones**: histórico con déficit, slippage y régimen de mercado.
* **ADL**: ejecuciones de Auto-Deleveraging (déficit, recuperado, φ) y posiciones desapalancadas.
* **Mercado**: mark / oracle / basis / liquidez publicados por el orquestador.
* **Control**: Modo Normal ↔ Cisne Negro y recapitalización de agentes vía la cola ``simulation_commands``
  (plano de control en PostgreSQL).

Ejecución::

    streamlit run dashboard/app.py

Seguridad: si ``DASHBOARD_ADMIN_PASSWORD`` está definida, los controles exigen autenticación de
operador (comparación en tiempo constante). Sin ella, los controles quedan abiertos (solo desarrollo).
Las lecturas son de solo lectura y sin locks: el dashboard no interfiere con el motor de liquidación.
"""

from __future__ import annotations

import hmac
import math
from dataclasses import replace
import os
import pathlib
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run dashboard/app.py` desde la raíz del repo
    sys.path.insert(0, str(ROOT))

from ai_agents.control_plane import SimulationCommand  # noqa: E402
from dashboard.data import (  # noqa: E402
    HEALTH_WARNING,
    DashboardRepository,
    RiskOverview,
    fmt_short,
    fmt_usd,
    risk_overview,
)
from database.connection import DatabaseManager, DBConfig  # noqa: E402

# -----------------------------------------------------------------------------
# Paleta (instancia de referencia validada: categórica en orden fijo + estados reservados)
# -----------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # slots 1-4
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b", "neutral": "#898781"}
INK_MUTED = "#898781"
SERIES_5 = "#e87ba4"                                           # slot 5 (magenta), solo para ADL_RECOVERY
LEDGER_COLORS = alt.Scale(  # el color sigue a la entidad: los 4 tipos históricos conservan su slot
    domain=["CAPITAL_INJECTION", "LIQUIDATION_FEE", "BANKRUPTCY_DEFICIT", "ADJUSTMENT", "ADL_RECOVERY"],
    range=[*SERIES, SERIES_5],
)
ADL_STATUS_COLORS = alt.Scale(domain=["Recuperado", "Residual"], range=[STATUS["good"], STATUS["critical"]])
REGIME_COLORS = alt.Scale(domain=["LIVE_TESTNET", "STRESS_MONTE_CARLO"], range=SERIES[:2])
STATUS_COLORS = alt.Scale(
    domain=["Sana", "En riesgo", "Liquidable", "Sin precio"],
    range=[STATUS["good"], STATUS["warning"], STATUS["critical"], STATUS["neutral"]],
)
MODE_LABEL = {"NORMAL": "🟢 Modo Normal", "BLACK_SWAN": "🦢 Cisne Negro", "OFFLINE": "⚫ Orquestador detenido"}

st.set_page_config(page_title="Quant Risk Engine · Control", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_repository() -> DashboardRepository:
    """Un pool de conexiones pequeño por proceso de Streamlit (compartido entre sesiones)."""
    cfg = replace(DBConfig.from_env(), pool_min=1, pool_max=int(os.getenv("DASHBOARD_DB_POOL_MAX", "5")),
                  application_name="risk_dashboard")
    return DashboardRepository(DatabaseManager(cfg))


# -----------------------------------------------------------------------------
# Autorización de operador
# -----------------------------------------------------------------------------
def operator_authorized() -> bool:
    expected = os.getenv("DASHBOARD_ADMIN_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("operator_ok"):
        return True
    supplied = st.text_input("Contraseña de operador", type="password", key="operator_pwd")
    if supplied and hmac.compare_digest(supplied.encode(), expected.encode()):
        st.session_state["operator_ok"] = True
        st.session_state["operator_name"] = os.getenv("DASHBOARD_OPERATOR_NAME", "operator")
        return True
    if supplied:
        st.error("Credencial incorrecta")
    return False


COMMAND_LABELS = {
    SimulationCommand.SET_NORMAL: ("volver a Modo Normal", "🟢"),
    SimulationCommand.TRIGGER_BLACK_SWAN: ("Cisne Negro disparado", "🦢"),
    SimulationCommand.RESET_AGENTS: ("agentes recapitalizados", "♻️"),
}


def _send_command(repo: DashboardRepository, command: SimulationCommand, operator: str) -> None:
    """Callback de botón: se ejecuta ANTES del rerun, por lo que puede resetear widgets (confirmación)."""
    cid = repo.request(command, operator)
    label, icon = COMMAND_LABELS[command]
    st.session_state["last_command_msg"] = (f"Comando #{cid}: {label}", icon)
    st.session_state["confirm_bs"] = False
    st.session_state["confirm_reset"] = False


# -----------------------------------------------------------------------------
# Barra lateral: estado del orquestador y controles
# -----------------------------------------------------------------------------
def render_control_panel(repo: DashboardRepository) -> None:
    """
    Estado del orquestador + controles (se invoca dentro de ``with st.sidebar``). Es un fragmento con
    ``run_every``: se refresca solo, sin recargar la página, y los callbacks solo re-ejecutan el fragmento.
    """
    state = repo.control_state()
    st.subheader("Orquestador")
    if state.is_online():
        st.success(f"{MODE_LABEL.get(state.current_mode, state.current_mode)} · heartbeat {state.heartbeat_age_s:.1f}s")
    else:
        age = "nunca" if state.heartbeat_age_s is None else f"hace {state.heartbeat_age_s:,.0f}s"
        st.error(f"⚫ Sin heartbeat ({age}). Los comandos quedarán en cola.")
    if state.pending_commands:
        st.info(f"⏳ {state.pending_commands} comando(s) pendiente(s) de consumir")

    st.subheader("Control de régimen")
    authorized = operator_authorized()
    if not os.getenv("DASHBOARD_ADMIN_PASSWORD"):
        st.caption("⚠️ Controles sin autenticación (define DASHBOARD_ADMIN_PASSWORD en producción).")
    operator = st.session_state.get("operator_name", os.getenv("DASHBOARD_OPERATOR_NAME", "dashboard"))

    st.button("🟢 Volver a Modo Normal", key="btn_normal", width="stretch",
              disabled=not authorized or state.current_mode == "NORMAL",
              on_click=_send_command, args=(repo, SimulationCommand.SET_NORMAL, operator))
    confirm = st.checkbox("Confirmo el shock de mercado", key="confirm_bs", disabled=not authorized)
    st.button("🦢 Disparar Cisne Negro", key="btn_black_swan", type="primary", width="stretch",
              disabled=not authorized or not confirm or state.current_mode == "BLACK_SWAN",
              on_click=_send_command, args=(repo, SimulationCommand.TRIGGER_BLACK_SWAN, operator))

    st.subheader("Agentes")
    st.caption("Cierra sus posiciones y les devuelve el colateral inicial. El Insurance Fund no cambia.")
    confirm_reset = st.checkbox("Confirmo reiniciar las cuentas", key="confirm_reset", disabled=not authorized)
    st.button("♻️ Recapitalizar agentes", key="btn_reset_agents", width="stretch",
              disabled=not authorized or not confirm_reset,
              on_click=_send_command, args=(repo, SimulationCommand.RESET_AGENTS, operator))
    if msg := st.session_state.pop("last_command_msg", None):
        text, icon = msg
        st.toast(text, icon=icon)

    with st.expander("Últimos comandos"):
        cmds = repo.recent_commands()
        if cmds.empty:
            st.caption("Sin comandos")
        else:
            st.dataframe(cmds, hide_index=True, width="stretch",
                         column_config={"requested_at": st.column_config.DatetimeColumn("Pedido", format="HH:mm:ss"),
                                        "consumed_at": st.column_config.DatetimeColumn("Ejecutado", format="HH:mm:ss")})


# -----------------------------------------------------------------------------
# Gráficos (Altair): un eje, marcas finas, tooltip en todas las vistas
# -----------------------------------------------------------------------------
def fund_chart(ledger: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(ledger).encode(x=alt.X("created_at:T", title=None, axis=alt.Axis(format="%H:%M:%S")))
    line = base.mark_line(interpolate="step-after", strokeWidth=2, color=SERIES[0]).encode(
        y=alt.Y("balance_after:Q", title="Balance (USD)", axis=alt.Axis(format="$,.0f"),
                scale=alt.Scale(zero=False, nice=True)),
    )
    points = base.mark_circle(size=70, opacity=1, stroke="#fcfcfb", strokeWidth=2).encode(
        y="balance_after:Q",
        color=alt.Color("entry_type:N", scale=LEDGER_COLORS, title="Movimiento",
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("created_at:T", title="Fecha", format="%H:%M:%S"),
                 alt.Tooltip("entry_type:N", title="Tipo"),
                 alt.Tooltip("amount:Q", title="Importe", format="$,.2f"),
                 alt.Tooltip("balance_after:Q", title="Balance", format="$,.2f"),
                 alt.Tooltip("agent_name:N", title="Agente")],
    )
    return (line + points).properties(height=320)


def health_chart(table: pd.DataFrame) -> alt.Chart:
    data = table[table["n_positions"] > 0].copy()
    cap = 10.0
    data["health_plot"] = data["health_ratio"].clip(upper=cap).fillna(0.0)
    data["label"] = data["agent_name"].fillna(data["user_id"].astype(str)).astype(str)
    data["state"] = data["status"].str.split(" ", n=1).str[1]          # texto sin icono para la leyenda
    order = data.sort_values("health_plot")["label"].tolist()
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("label:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.35),
                axis=alt.Axis(labelLimit=200)),
        x=alt.X("health_plot:Q", title=f"Health ratio = Equity / MM (recortado a {cap:.0f})",
                scale=alt.Scale(domainMin=0)),
        color=alt.Color("state:N", scale=STATUS_COLORS, title="Estado", legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("label:N", title="Cuenta"), alt.Tooltip("status:N", title="Estado"),
                 alt.Tooltip("health_ratio:Q", title="Health ratio", format=".3f"),
                 alt.Tooltip("equity:Q", title="Equity", format="$,.2f"),
                 alt.Tooltip("maintenance_margin:Q", title="Margen mant.", format="$,.2f"),
                 alt.Tooltip("margin_buffer:Q", title="Colchón", format="$,.2f"),
                 alt.Tooltip("effective_leverage:Q", title="Apalanc. efectivo", format=".1f")],
    )
    rules = alt.Chart(pd.DataFrame({"x": [1.0, HEALTH_WARNING],
                                    "what": ["Umbral de liquidación (1.0)", f"Alerta ({HEALTH_WARNING})"]})).mark_rule(
        color=INK_MUTED, strokeDash=[4, 3], strokeWidth=1).encode(x="x:Q", tooltip=alt.Tooltip("what:N", title=""))
    return (bars + rules).properties(height=max(110, 44 * len(data)))


def deficit_chart(liq: pd.DataFrame) -> alt.Chart:
    data = liq.sort_values("liquidated_at")
    return alt.Chart(data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=14).encode(
        x=alt.X("liquidated_at:T", title=None, axis=alt.Axis(format="%H:%M:%S")),
        y=alt.Y("deficit:Q", title="Déficit absorbido (USD)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("regime:N", scale=REGIME_COLORS, title="Régimen", legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("liquidated_at:T", title="Fecha", format="%H:%M:%S"),
                 alt.Tooltip("agent_name:N", title="Agente"), alt.Tooltip("regime:N", title="Régimen"),
                 alt.Tooltip("deficit:Q", title="Déficit", format="$,.2f"),
                 alt.Tooltip("fee:Q", title="Fee", format="$,.2f"),
                 alt.Tooltip("max_slippage_bps:Q", title="Slippage máx (bps)", format=".1f")],
    ).properties(height=260)


def slippage_chart(liq: pd.DataFrame) -> alt.Chart:
    return alt.Chart(liq).mark_bar(color=SERIES[0], cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("wavg_slippage_bps:Q", bin=alt.Bin(maxbins=20), title="Slippage ponderado (bps)"),
        y=alt.Y("count():Q", title="Liquidaciones", axis=alt.Axis(format="d", tickMinStep=1)),
        tooltip=[alt.Tooltip("count():Q", title="Liquidaciones"),
                 alt.Tooltip("wavg_slippage_bps:Q", bin=True, title="Rango (bps)")],
    ).properties(height=260)


def adl_chart(adl: pd.DataFrame) -> alt.Chart:
    """Déficit de cada ADL partido en recuperado (ganadores) y residual (sin socializar): barras apiladas."""
    data = adl.melt(id_vars=["created_at", "adl_event_id", "phi", "status", "liquidated_agent"],
                    value_vars=["recovered", "residual_shortfall"], var_name="part", value_name="usd")
    data["part"] = data["part"].map({"recovered": "Recuperado", "residual_shortfall": "Residual"})
    data["label"] = "#" + data["adl_event_id"].astype(str)
    return alt.Chart(data).mark_bar(cornerRadiusEnd=4, size=16, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("label:N", title="Evento ADL", sort=alt.SortField("adl_event_id"), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("sum(usd):Q", title="Déficit (USD)", axis=alt.Axis(format="$,.0f"), stack="zero"),
        color=alt.Color("part:N", scale=ADL_STATUS_COLORS, title=None, legend=alt.Legend(orient="top")),
        order=alt.Order("part:N", sort="ascending"),
        tooltip=[alt.Tooltip("label:N", title="Evento"), alt.Tooltip("created_at:T", title="Fecha", format="%H:%M:%S"),
                 alt.Tooltip("part:N", title="Parte"), alt.Tooltip("usd:Q", title="Importe", format="$,.2f"),
                 alt.Tooltip("phi:Q", title="φ (fracción recortada)", format=".2%"),
                 alt.Tooltip("liquidated_agent:N", title="Liquidación origen"), alt.Tooltip("status:N", title="Estado")],
    ).properties(height=240)


# -----------------------------------------------------------------------------
# Paneles en vivo
# -----------------------------------------------------------------------------
def render_kpis(repo: DashboardRepository, ledger: pd.DataFrame, accounts: pd.DataFrame, risk: RiskOverview) -> None:
    fund = repo.fund_overview()
    at_risk = int(accounts["status"].isin(["🔴 Liquidable", "🟠 En riesgo"]).sum()) if not accounts.empty else 0
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    spark = ledger["balance_after"].tail(60).tolist() if len(ledger) > 1 else None
    c1.metric("Insurance Fund", fmt_short(fund.balance), delta=f"{fund.net_pnl:+,.0f} USD neto",
              delta_color="normal", border=True, chart_data=spark, chart_type="area",
              help="Delta = comisiones + recuperado por ADL − déficits absorbidos (excluye capitalizaciones)")
    cov = risk.coverage_ratio
    c2.metric("Cobertura IF / OI", "—" if cov is None else f"{cov:.2%}", border=True,
              delta=None if risk.open_interest <= 0 else f"OI {fmt_short(risk.open_interest)}",
              delta_color="off", delta_arrow="off",
              help="Insurance Fund / Open Interest (Σ|q|·mark de las cuentas activas). Los exchanges lo vigilan "
                   "como % del OI: por debajo, una liquidación grande con gap puede agotar el fondo.")
    if risk.var is not None and risk.es is not None:
        hz = f"{risk.horizon_ticks} ticks" if risk.horizon_ticks else "1 tick"
        cover = risk.fund_to_es
        c3.metric(f"VaR {risk.alpha:.0%} cartera", fmt_short(risk.var), delta=f"ES {fmt_short(risk.es)}",
                  delta_color="off", delta_arrow="off", border=True,
                  help=f"Cartera agregada de los agentes, horizonte {hz} (1 tick = 5 min simulados). Student-t "
                       "(ν=4) con covarianza EWMA (λ=0.94) de los marks del shadow market. ES = pérdida media en el "
                       f"1 % peor de los escenarios. El fondo cubre {cover:.2f}× el ES." if cover is not None else
                       f"Cartera agregada de los agentes, horizonte {hz}. Student-t (ν=4) + EWMA (λ=0.94).")
    else:
        c3.metric("VaR 99% cartera", "—", border=True, help="Disponible con el orquestador en marcha (≥ 30 observaciones).")
    c4.metric("Liquidaciones", f"{fund.liquidations:,}", border=True,
              delta=f"déficit {fmt_short(fund.deficits_absorbed)}" if fund.deficits_absorbed else None,
              delta_color="off", delta_arrow="off",
              help=f"Última: {fund.last_liquidation_at:%H:%M:%S}" if fund.last_liquidation_at else None)
    c5.metric("ADL ejecutados", f"{fund.adl_events:,}", border=True,
              delta=f"+{fmt_short(fund.adl_recovered)} al fondo" if fund.adl_recovered else None,
              delta_color="normal", help="Ejecuciones de Auto-Deleveraging y beneficio socializado de las posiciones ganadoras")
    c6.metric("Cuentas en riesgo", f"{at_risk}", border=True, help=f"Health ratio ≤ {HEALTH_WARNING} o liquidables")
    if fund.balance < 0:
        st.error(f"🚨 Insurance Fund en negativo ({fmt_usd(fund.balance)}): déficit pendiente de socializar. "
                 "El motor ejecuta ADL pro-rata sobre las posiciones ganadoras en cada ciclo (pestaña ⚖️ ADL).")


def render_fund_tab(ledger: pd.DataFrame) -> None:
    if ledger.empty:
        st.info("El Insurance Fund aún no tiene movimientos. Capitalízalo con `--insurance-fund` en el orquestador.")
        return
    flows = ledger.groupby("entry_type")["amount"].sum()
    a, b, c, d = st.columns(4)
    a.metric("Capital aportado", fmt_usd(float(flows.get("CAPITAL_INJECTION", 0.0)), compact=True))
    b.metric("Comisiones de liquidación", fmt_usd(float(flows.get("LIQUIDATION_FEE", 0.0)), compact=True))
    c.metric("Déficits absorbidos", fmt_usd(-float(flows.get("BANKRUPTCY_DEFICIT", 0.0)), compact=True))
    d.metric("Recuperado por ADL", fmt_usd(float(flows.get("ADL_RECOVERY", 0.0)), compact=True))
    st.altair_chart(fund_chart(ledger), width="stretch")
    st.dataframe(
        ledger.sort_values("entry_id", ascending=False), hide_index=True, width="stretch",
        column_order=["created_at", "entry_type", "amount", "balance_after", "agent_name", "liquidation_event_id", "note"],
        column_config={
            "created_at": st.column_config.DatetimeColumn("Fecha (UTC)", format="YYYY-MM-DD HH:mm:ss"),
            "entry_type": "Tipo", "agent_name": "Agente", "note": "Nota",
            "amount": st.column_config.NumberColumn("Importe", format="dollar"),
            "balance_after": st.column_config.NumberColumn("Balance", format="dollar"),
            "liquidation_event_id": st.column_config.NumberColumn("Evento", format="%d"),
        },
    )


def render_accounts_tab(health, risk: RiskOverview) -> None:  # noqa: ANN001 - AccountsHealth
    table = health.table.copy()
    if not table.empty:
        by = risk.var_by_account
        table["var"] = table["agent_name"].map(lambda n: (by.get(n) or {}).get("var"))
        table["es"] = table["agent_name"].map(lambda n: (by.get(n) or {}).get("es"))
        table["es_to_equity"] = table["es"] / table["equity"].where(table["equity"] > 0)
    if not health.marks_live:
        st.warning("Precios no publicados por el orquestador: health ratios calculados con el último "
                   "`last_mark_price` conocido." if health.marks else "Sin precios disponibles.")
    if table.empty:
        st.info("No hay cuentas activas.")
        return
    n_bars = int((table["n_positions"] > 0).sum())
    if n_bars:
        # altura explícita: con un eje y discreto (bandas) Streamlit no hereda la altura del spec
        st.altair_chart(health_chart(table), width="stretch", height=max(110, 44 * n_bars) + 90)
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_order=["status", "agent_name", "agent_profile", "positions", "health_ratio", "margin_ratio", "equity",
                      "maintenance_margin", "margin_buffer", "var", "es", "es_to_equity", "collateral", "unrealized_pnl",
                      "effective_leverage", "gross_notional"],
        column_config={
            "status": "Estado", "agent_name": "Agente", "agent_profile": "Perfil", "positions": "Posiciones",
            "health_ratio": st.column_config.NumberColumn("Health (Eq/MM)", format="%.3f"),
            "margin_ratio": st.column_config.NumberColumn("Margin ratio (MM/Eq)", format="%.3f"),
            "equity": st.column_config.NumberColumn("Equity", format="dollar"),
            "maintenance_margin": st.column_config.NumberColumn("Margen mant.", format="dollar"),
            "margin_buffer": st.column_config.NumberColumn("Colchón", format="dollar"),
            "collateral": st.column_config.NumberColumn("Colateral", format="dollar"),
            "unrealized_pnl": st.column_config.NumberColumn("uPnL", format="dollar"),
            "effective_leverage": st.column_config.NumberColumn("Apalanc.", format="%.1fx"),
            "gross_notional": st.column_config.NumberColumn("Nocional bruto", format="dollar"),
            "var": st.column_config.NumberColumn("VaR 99%", format="dollar", help="Student-t + EWMA, horizonte de 12 ticks"),
            "es": st.column_config.NumberColumn("ES 99%", format="dollar", help="Pérdida media en el 1 % peor"),
            "es_to_equity": st.column_config.NumberColumn("ES / Equity", format="percent",
                                                          help="> 100 %: un escenario de cola deja la cuenta en bancarrota"),
        },
    )


def render_liquidations_tab(liq: pd.DataFrame) -> None:
    if liq.empty:
        st.info("Sin liquidaciones registradas.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Con déficit", f"{int((liq['deficit'] > 0).sum())} / {len(liq)}")
    c2.metric("Déficit medio", fmt_usd(float(liq.loc[liq["deficit"] > 0, "deficit"].mean())
                                       if (liq["deficit"] > 0).any() else 0.0))
    c3.metric("Slippage ponderado medio", f"{liq['wavg_slippage_bps'].mean():.1f} bps")
    c4.metric("Slippage máximo", f"{liq['max_slippage_bps'].max():.1f} bps")
    left, right = st.columns([3, 2])
    left.altair_chart(deficit_chart(liq), width="stretch")
    right.altair_chart(slippage_chart(liq), width="stretch")
    st.dataframe(
        liq, hide_index=True, width="stretch",
        column_order=["liquidated_at", "agent_name", "regime", "symbols", "positions", "equity_before",
                      "maintenance_margin_before", "health_before", "realized_pnl", "fee", "deficit", "fund_delta",
                      "wavg_slippage_bps", "max_slippage_bps", "execution_source", "batch_id"],
        column_config={
            "liquidated_at": st.column_config.DatetimeColumn("Fecha (UTC)", format="YYYY-MM-DD HH:mm:ss"),
            "agent_name": "Agente", "regime": "Régimen", "symbols": "Activos", "positions": "Pos.",
            "equity_before": st.column_config.NumberColumn("Equity previa", format="dollar"),
            "maintenance_margin_before": st.column_config.NumberColumn("MM previo", format="dollar"),
            "health_before": st.column_config.NumberColumn("Health previo", format="%.3f"),
            "realized_pnl": st.column_config.NumberColumn("PnL realizado", format="dollar"),
            "fee": st.column_config.NumberColumn("Fee", format="dollar"),
            "deficit": st.column_config.NumberColumn("Déficit", format="dollar"),
            "fund_delta": st.column_config.NumberColumn("Δ Fondo", format="dollar"),
            "wavg_slippage_bps": st.column_config.NumberColumn("Slippage pond. (bps)", format="%.1f"),
            "max_slippage_bps": st.column_config.NumberColumn("Slippage máx (bps)", format="%.1f"),
            "execution_source": "Ejecución", "batch_id": "Batch",
        },
    )


def render_adl_tab(repo: DashboardRepository) -> None:
    adl = repo.adl_history()
    if adl.empty:
        st.info("Sin eventos de Auto-Deleveraging: el Insurance Fund ha cubierto todos los déficits.")
        st.caption("Si una liquidación deja el fondo en negativo, el motor recorta pro-rata las posiciones "
                   "ganadoras (misma fracción φ para todas, al precio de entrada) hasta reponerlo.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Déficit a socializar", fmt_usd(float(adl["shortfall"].sum()), compact=True))
    c2.metric("Recuperado", fmt_usd(float(adl["recovered"].sum()), compact=True))
    c3.metric("φ máxima", f"{adl['phi'].max():.2%}", help="Mayor fracción de las posiciones ganadoras cerrada en un ADL")
    c4.metric("Residual pendiente", fmt_usd(float(adl.iloc[0]["residual_shortfall"]), compact=True),
              help="Residual del último ADL (se reintenta en cada ciclo con las nuevas ganancias)")
    st.altair_chart(adl_chart(adl), width="stretch")
    st.dataframe(
        adl, hide_index=True, width="stretch",
        column_order=["created_at", "status", "liquidated_agent", "regime", "shortfall", "recovered",
                      "residual_shortfall", "phi", "counterparties", "positions", "fund_balance_before",
                      "fund_balance_after", "trigger_batch_id"],
        column_config={
            "created_at": st.column_config.DatetimeColumn("Fecha (UTC)", format="YYYY-MM-DD HH:mm:ss"),
            "status": "Estado", "liquidated_agent": "Liquidación origen", "regime": "Régimen",
            "shortfall": st.column_config.NumberColumn("Déficit", format="dollar"),
            "recovered": st.column_config.NumberColumn("Recuperado", format="dollar"),
            "residual_shortfall": st.column_config.NumberColumn("Residual", format="dollar"),
            "phi": st.column_config.NumberColumn("φ", format="percent"),
            "counterparties": st.column_config.NumberColumn("Cuentas", format="%d"),
            "positions": st.column_config.NumberColumn("Posiciones", format="%d"),
            "fund_balance_before": st.column_config.NumberColumn("Fondo antes", format="dollar"),
            "fund_balance_after": st.column_config.NumberColumn("Fondo después", format="dollar"),
            "trigger_batch_id": "Batch liquidación",
        },
    )
    with st.expander("Posiciones desapalancadas"):
        st.dataframe(
            repo.adl_fills(), hide_index=True, width="stretch",
            column_config={
                "adl_event_id": st.column_config.NumberColumn("ADL", format="%d"),
                "created_at": st.column_config.DatetimeColumn("Fecha (UTC)", format="HH:mm:ss"),
                "agent_name": "Agente", "symbol": "Activo",
                "quantity_before": st.column_config.NumberColumn("Cantidad previa", format="%.5f"),
                "quantity_closed": st.column_config.NumberColumn("Cerrada", format="%.5f"),
                "entry_price": st.column_config.NumberColumn("Entrada", format="dollar"),
                "mark_price": st.column_config.NumberColumn("Mark", format="dollar"),
                "unrealized_pnl_before": st.column_config.NumberColumn("uPnL previo", format="dollar"),
                "haircut": st.column_config.NumberColumn("Transferido al fondo", format="dollar"),
                "account_phi": st.column_config.NumberColumn("φ cuenta", format="percent"),
            },
        )


def render_market_tab(state) -> None:  # noqa: ANN001 - ControlState
    market = state.market_state or {}
    if not market:
        st.info("El orquestador no ha publicado estado de mercado.")
        return
    cols = st.columns(len(market))
    for col, (sym, m) in zip(cols, sorted(market.items())):
        with col.container(border=True):
            st.markdown(f"#### {sym}")
            a, b = st.columns(2)
            a.metric("Mark", fmt_usd(m.get("mark")))
            b.metric("Oracle", fmt_usd(m.get("oracle")))
            basis = m.get("basis_bps")
            liq = m.get("liquidity")
            spread = m.get("spread_bps")
            a.metric("Basis mark−oracle", "—" if basis is None else f"{basis:+.1f} bps")
            b.metric("Liquidez del libro", "—" if liq is None else f"{liq:.0%}")
            st.caption("Spread: " + ("—" if spread is None or not math.isfinite(spread) else f"{spread:.2f} bps"))
    stats = state.stats or {}
    if stats:
        st.caption(f"Fuente de precios: `{stats.get('source')}` · tiempo de simulación {stats.get('elapsed_s', 0):,.0f}s")
        cov = stats.get("stress_coverage") or {}
        if cov:
            ratio = cov.get("ratio")
            st.caption(f"Chequeo pre-flight de estrés (salto adverso {cov.get('jump', 0):.0%}): peor déficit "
                       f"{fmt_usd(cov.get('worst_deficit'), compact=True)} · cobertura del fondo "
                       f"{'∞' if ratio is None else f'{ratio:.2f}×'}")


def render_live(repo: DashboardRepository) -> None:
    state = repo.control_state()
    ledger = repo.fund_ledger()
    health = repo.accounts_health(state)
    liq = repo.liquidations()
    fund_balance = float(ledger["balance_after"].iloc[-1]) if not ledger.empty else repo.fund_overview().balance
    risk = risk_overview(state, health.table, fund_balance)
    render_kpis(repo, ledger, health.table, risk)
    t1, t2, t3, t4, t5 = st.tabs(["🏦 Insurance Fund", "👥 Cuentas", "💥 Liquidaciones", "⚖️ ADL", "📈 Mercado"])
    with t1:
        render_fund_tab(ledger)
    with t2:
        render_accounts_tab(health, risk)
    with t3:
        render_liquidations_tab(liq)
    with t4:
        render_adl_tab(repo)
    with t5:
        render_market_tab(state)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    try:
        repo = get_repository()
        repo.control_state()
    except Exception as exc:  # noqa: BLE001 - mensaje claro en lugar de un traceback
        st.error(f"No se puede conectar con PostgreSQL: {exc}")
        st.stop()
    with st.sidebar:
        st.title("🛡️ Risk Engine")
        refresh = st.slider("Refresco (s)", min_value=1, max_value=30,
                            value=int(os.getenv("DASHBOARD_REFRESH_S", "3")))
        st.fragment(run_every=refresh)(render_control_panel)(repo)
    st.title("AI-Driven Quant Risk & Liquidation Engine")
    st.caption("Monitorización en tiempo real del motor de margen cruzado, liquidaciones y Insurance Fund.")
    st.fragment(run_every=refresh)(render_live)(repo)


main()
