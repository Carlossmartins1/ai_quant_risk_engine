"""
dashboard/app.py
================

Terminal de riesgo en tiempo real del AI-Driven Quant Risk & Liquidation Engine (Plotly Dash).

Vistas
------
* **Cabecera**: modo del orquestador (Normal / Cisne Negro / detenido), heartbeat y fuente de precios.
* **Hyperliquid**: estado del feed real, mark/oracle de BTC y ETH, funding, desviación del precio
  simulado respecto al real (se dispara en el Cisne Negro) y latencia del feed.
* **KPIs**: Insurance Fund (alerta de déficit/ADL), cobertura sobre open interest, déficit absorbido,
  VaR/ES 99 % de la cartera agregada y liquidaciones.
* **Gráficos**: evolución del Insurance Fund y precio simulado vs. real.
* **Pestañas**: Cuentas y Posiciones · Mapa de liquidaciones · Liquidaciones · Insurance Fund y ADL ·
  Feed Hyperliquid.
* **Control de régimen** (lateral): Cisne Negro, Modo Normal y recapitalización de agentes con diálogo
  de confirmación, encolados en ``simulation_commands`` (plano de control en PostgreSQL).

Ejecución::

    python -m dashboard.app                                   # desarrollo (http://127.0.0.1:8501)
    gunicorn -w 1 --threads 8 -b 0.0.0.0:8501 dashboard.app:server   # producción (contenedor)

Un solo worker: la serie precio simulado vs. real se acumula en memoria del proceso (``MarketTape``).

Seguridad: si ``DASHBOARD_ADMIN_PASSWORD`` está definida, cada comando se valida en el servidor contra
la contraseña de operador (comparación en tiempo constante). Sin ella, los controles quedan abiertos
(solo desarrollo). Las lecturas son de solo lectura y sin locks: no interfieren con el motor.
"""

from __future__ import annotations

import hmac
import os
import pathlib
import sys
import threading
from dataclasses import dataclass, replace
from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
import structlog
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `python dashboard/app.py` desde la raíz del repo
    sys.path.insert(0, str(ROOT))

from ai_agents.control_plane import SimulationCommand  # noqa: E402
from dashboard import components as ui  # noqa: E402
from dashboard import figures  # noqa: E402
from dashboard.data import (  # noqa: E402
    DashboardRepository,
    MarketTape,
    feed_view,
    fmt_short,
    fmt_usd,
    liquidation_map,
    risk_overview,
)
from dashboard.theme import GRAPH_CONFIG  # noqa: E402
from database.connection import DatabaseManager, DBConfig  # noqa: E402

log = structlog.get_logger(__name__)

FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
TABS = ("accounts", "liqmap", "liquidations", "adl", "feed")
LIQMAP_BAND = 0.15
COMMANDS = {
    "confirm-normal": (SimulationCommand.SET_NORMAL, "Volver a Modo Normal"),
    "confirm-black-swan": (SimulationCommand.TRIGGER_BLACK_SWAN, "Cisne Negro disparado"),
    "confirm-reset": (SimulationCommand.RESET_AGENTS, "Agentes recapitalizados"),
}


# =============================================================================
# Repositorio (pool pequeño, creado de forma perezosa: la app arranca aunque la BD aún no esté lista)
# =============================================================================
class RepositoryProvider:
    def __init__(self, repo: DashboardRepository | None = None) -> None:
        self._repo = repo
        self._lock = threading.Lock()

    def get(self) -> DashboardRepository:
        if self._repo is None:
            with self._lock:
                if self._repo is None:
                    cfg = replace(DBConfig.from_env(), pool_min=1,
                                  pool_max=int(os.getenv("DASHBOARD_DB_POOL_MAX", "5")),
                                  application_name="risk_dashboard")
                    self._repo = DashboardRepository(DatabaseManager(cfg))
        return self._repo


# =============================================================================
# Autorización de operador
# =============================================================================
def operator_password() -> str:
    return os.getenv("DASHBOARD_ADMIN_PASSWORD") or ""


def is_authorized(supplied: str | None) -> bool:
    expected = operator_password()
    if not expected:
        return True
    return bool(supplied) and hmac.compare_digest(supplied.encode(), expected.encode())


def operator_name() -> str:
    return os.getenv("DASHBOARD_OPERATOR_NAME") or ("operator" if operator_password() else "dashboard")


# =============================================================================
# Vistas (funciones puras sobre el repositorio: los callbacks solo las enrutan)
# =============================================================================
@dataclass(frozen=True)
class MainView:
    header: list[Any]
    hl_brand: list[Any]
    tickers: list[Any]
    kpis: list[Any]
    fund_figure: Any
    price_figure: Any
    alert: Any


def main_view(repo: DashboardRepository, tape: MarketTape, logo_src: str) -> MainView:
    state = repo.control_state()
    tape.record(state)
    fund = repo.fund_overview()
    health = repo.accounts_health(state)
    at_risk = int(health.table["state"].isin(["Liquidable", "En riesgo"]).sum()) if not health.table.empty else 0
    risk = risk_overview(state, health.table, fund.balance)
    alert = None
    if fund.balance < 0:
        alert = html.Div(f"⚠ INSURANCE FUND EN DÉFICIT ({fmt_usd(fund.balance)}): déficit pendiente de socializar. "
                         "El motor ejecuta ADL pro-rata sobre las posiciones ganadoras en cada ciclo.",
                         className="alert-adl mb-3")
    return MainView(
        header=ui.header_status(state),
        hl_brand=ui.hyperliquid_brand(state, logo_src),
        tickers=ui.ticker_cells(feed_view(state)),
        kpis=ui.kpi_cards(fund, risk, at_risk),
        fund_figure=figures.fund_figure(repo.fund_ledger()),
        price_figure=figures.price_figure(tape.frame()),
        alert=alert,
    )


@dataclass(frozen=True)
class ControlsView:
    status: list[Any]
    normal_disabled: bool
    black_swan_disabled: bool
    reset_disabled: bool
    auth_hint: str
    commands: list[dict[str, Any]]


def controls_view(repo: DashboardRepository, password: str | None) -> ControlsView:
    state = repo.control_state()
    authorized = is_authorized(password)
    if not operator_password():
        hint = "⚠ Controles sin autenticación: define DASHBOARD_ADMIN_PASSWORD en producción."
    elif authorized:
        hint = f"Operador autenticado ({operator_name()})."
    else:
        hint = "Introduce la contraseña de operador para habilitar los controles."
    mode = state.current_mode
    return ControlsView(
        status=ui.orchestrator_status(state),
        normal_disabled=not authorized or mode == "NORMAL",
        black_swan_disabled=not authorized or mode == "BLACK_SWAN",
        reset_disabled=not authorized,
        auth_hint=hint,
        commands=ui.records(_command_status(repo.recent_commands()), ui.COMMAND_COLUMNS),
    )


COMMAND_SHORT = {"SET_NORMAL": "Modo Normal", "TRIGGER_BLACK_SWAN": "Cisne Negro", "RESET_AGENTS": "Recapitalizar"}


def _command_status(cmds: pd.DataFrame) -> pd.DataFrame:
    """Vista compacta para el lateral: nombre corto + hora de ejecución (o en cola)."""
    if cmds.empty:
        return cmds.assign(label=[], status=[])
    return cmds.assign(
        label=[f"#{i} {COMMAND_SHORT.get(c, c)}" for i, c in zip(cmds["command_id"], cmds["command"])],
        status=[f"✓ {c:%H:%M:%S}" if c is not None and not pd.isna(c) else "⏳ en cola" for c in cmds["consumed_at"]],
    )


def dispatch_command(repo: DashboardRepository, trigger: str, password: str | None) -> tuple[str, str]:
    """Encola el comando del botón confirmado. Devuelve ``(mensaje, color)`` para el aviso."""
    if trigger not in COMMANDS:
        raise ValueError(f"Disparador desconocido: {trigger}")
    if not is_authorized(password):
        log.warning("dashboard.command_rejected", trigger=trigger)
        return "Credencial de operador incorrecta: comando rechazado.", "danger"
    command, label = COMMANDS[trigger]
    cid = repo.request(command, operator_name())
    return f"Comando #{cid} encolado: {label}.", "success"


def tab_view(repo: DashboardRepository, tape: MarketTape, tab: str) -> dict[str, Any]:
    """Datos de la pestaña activa (las inactivas no consultan la BD)."""
    if tab == "accounts":
        state = repo.control_state()
        health = repo.accounts_health(state)
        table = health.table
        if not table.empty:
            by = risk_overview(state, table, 0.0).var_by_account
            table = table.assign(var=table["agent_name"].map(lambda n: (by.get(n) or {}).get("var")),
                                 es=table["agent_name"].map(lambda n: (by.get(n) or {}).get("es")))
            table["es_to_equity"] = pd.to_numeric(table["es"], errors="coerce") / table["equity"].where(table["equity"] > 0)
        note = "" if health.marks_live else (
            "Precios no publicados por el orquestador: health factors con el último last_mark_price conocido."
            if health.marks else "Sin precios disponibles.")
        return {"acc-health-graph": figures.health_figure(health.table),
                "acc-table": ui.records(table, ui.ACCOUNT_COLUMNS),
                "pos-table": ui.records(health.positions, ui.POSITION_COLUMNS),
                "acc-note": note}
    if tab == "liqmap":
        health = repo.accounts_health()
        lmap = liquidation_map(health, band=LIQMAP_BAND)
        summary = []
        for sym, grp in lmap.groupby("symbol"):
            for side in ("Largos", "Cortos"):
                g = grp[(grp["side"] == side) & (grp["notional"] > 0)]
                peak = g.loc[g["notional"].idxmax()] if len(g) else None
                summary.append(html.Div([
                    html.Span(f"{sym} {side.lower()}"),
                    html.Span("—" if peak is None else
                              f"{fmt_usd(g['notional'].sum(), compact=True)} · pico {peak['price_mid']:,.0f}"),
                ], className="kv"))
        if not summary:
            summary = [html.Div("Sin posiciones abiertas.", className="text-muted-x")]
        return {"liqmap-graph": figures.liquidation_heatmap(lmap, band=LIQMAP_BAND), "liqmap-summary": summary}
    if tab == "liquidations":
        liq = repo.liquidations()
        with_deficit = liq[liq["deficit"] > 0] if not liq.empty else liq
        stats = [("Con déficit", f"{len(with_deficit)} / {len(liq)}"),
                 ("Déficit medio", fmt_usd(float(with_deficit["deficit"].mean()) if len(with_deficit) else 0.0)),
                 ("Slippage pond. medio", ui.fmt_bps(liq["wavg_slippage_bps"].mean() if len(liq) else None, signed=False)),
                 ("Slippage máximo", ui.fmt_bps(liq["max_slippage_bps"].max() if len(liq) else None, signed=False))]
        return {"liq-stats": [html.Div([html.Span(k), html.Span(v)], className="kv") for k, v in stats],
                "liq-deficit-graph": figures.deficit_figure(liq),
                "liq-slippage-graph": figures.slippage_figure(liq),
                "liq-table": ui.records(liq, ui.LIQUIDATION_COLUMNS)}
    if tab == "adl":
        adl = repo.adl_history()
        fund = repo.fund_overview()
        if fund.balance < 0:
            callout = html.Div(f"Fondo en déficit: {fmt_usd(fund.balance)} pendientes de socializar. El ADL recorta "
                               "pro-rata las posiciones ganadoras (misma fracción φ) en cada ciclo.",
                               className="alert-adl")
        elif len(adl):
            callout = html.Div("Déficit socializado: el ADL repuso el Insurance Fund.", className="up mono")
        else:
            callout = html.Div("Sin eventos de Auto-Deleveraging: el Insurance Fund ha cubierto todos los déficits.",
                               className="up mono")
        cov = (repo.control_state().stats or {}).get("stress_coverage") or {}
        stats = [("Déficit a socializar", fmt_short(float(adl["shortfall"].sum()) if len(adl) else 0.0)),
                 ("Recuperado", fmt_usd(float(adl["recovered"].sum()) if len(adl) else 0.0)),
                 ("φ máxima", ui.fmt_pct(float(adl["phi"].max()) if len(adl) else None, 2)),
                 ("Residual pendiente", fmt_short(float(adl.iloc[0]["residual_shortfall"]) if len(adl) else 0.0))]
        if cov:
            ratio = cov.get("ratio")
            stats.append((f"Estrés pre-flight (salto {cov.get('jump', 0):.0%})",
                          f"peor déficit {fmt_short(cov.get('worst_deficit'))} · "
                          + ("cobertura ∞" if ratio is None else f"cobertura {ratio:.2f}×")))
        ledger = repo.fund_ledger().sort_values("entry_id", ascending=False)
        return {"adl-callout": callout,
                "adl-stats": [html.Div([html.Span(k), html.Span(v)], className="kv") for k, v in stats],
                "adl-graph": figures.adl_figure(adl), "adl-table": ui.records(adl, ui.ADL_COLUMNS),
                "adl-fills-table": ui.records(repo.adl_fills(), ui.ADL_FILL_COLUMNS),
                "ledger-table": ui.records(ledger, ui.LEDGER_COLUMNS)}
    if tab == "feed":
        state = repo.control_state()
        tape.record(state)
        feed = (state.stats or {}).get("feed") or {}
        meta = ("Feed real no configurado: arranca el orquestador con --live para comparar con Hyperliquid."
                if not feed.get("configured") else
                f"WebSocket {'conectado' if feed.get('connected') else 'DESCONECTADO'} · "
                f"{len(feed.get('symbols') or {})} activo(s) suscritos")
        return {"feed-table": ui.records(feed_view(state), ui.FEED_COLUMNS),
                "feed-deviation-graph": figures.deviation_figure(tape.frame()), "feed-meta": meta}
    raise ValueError(f"Pestaña desconocida: {tab}")


TAB_OUTPUTS = {
    "accounts": [("acc-health-graph", "figure"), ("acc-table", "data"), ("pos-table", "data"), ("acc-note", "children")],
    "liqmap": [("liqmap-graph", "figure"), ("liqmap-summary", "children")],
    "liquidations": [("liq-stats", "children"), ("liq-deficit-graph", "figure"), ("liq-slippage-graph", "figure"),
                     ("liq-table", "data")],
    "adl": [("adl-callout", "children"), ("adl-stats", "children"), ("adl-graph", "figure"), ("adl-table", "data"),
            ("adl-fills-table", "data"), ("ledger-table", "data")],
    "feed": [("feed-table", "data"), ("feed-deviation-graph", "figure"), ("feed-meta", "children")],
}


# =============================================================================
# Layout
# =============================================================================
def _graph(graph_id: str) -> dcc.Graph:
    return dcc.Graph(id=graph_id, figure=figures.empty_figure("Cargando…"), config=GRAPH_CONFIG)


def _panel(title: str, *children: Any, class_name: str = "") -> html.Div:
    return html.Div([html.Div(title, className="panel-title"), *children], className=f"panel {class_name}".strip())


def _confirm_button(trigger: str, button_id: str, label: str, color: str, message: str) -> dcc.ConfirmDialogProvider:
    return dcc.ConfirmDialogProvider(
        id=trigger, message=message,
        children=dbc.Button(label, id=button_id, color=color, outline=color != "danger", className="w-100 mb-2",
                            disabled=True),
    )


def sidebar(password_required: bool) -> html.Div:
    auth = [dbc.Input(id="operator-pwd", type="password", placeholder="Contraseña de operador", className="mb-2",
                      debounce=True, style={} if password_required else {"display": "none"})]
    return html.Div([
        html.Div("Orquestador", className="panel-title"),
        html.Div(id="orch-status", className="mb-4"),
        html.Div("Control de régimen", className="panel-title"),
        *auth,
        _confirm_button("confirm-black-swan", "btn-black-swan", "Disparar Cisne Negro", "danger",
                        "¿Disparar el CISNE NEGRO?\n\nEl motor pasará a la trayectoria de cola del Monte Carlo "
                        "(crash forzado) y se ejecutarán liquidaciones en régimen de estrés."),
        _confirm_button("confirm-normal", "btn-normal", "Volver a Modo Normal", "success",
                        "¿Volver a Modo Normal? Los precios vuelven al feed en vivo / Monte Carlo normal."),
        html.Div("Agentes", className="panel-title mt-3"),
        _confirm_button("confirm-reset", "btn-reset", "Recapitalizar agentes", "info",
                        "¿Recapitalizar a todos los agentes?\n\nSe cerrarán sus posiciones y recuperarán el "
                        "colateral inicial. El Insurance Fund no cambia."),
        html.Div(id="auth-hint", className="hint mb-3"),
        html.Div("Últimos comandos", className="panel-title mt-2"),
        ui.data_table("commands-table", ui.COMMAND_COLUMNS, page_size=8),
    ], className="sidebar")


def build_layout(app: Dash, refresh_ms: int) -> html.Div:
    logo = app.get_asset_url("hyperliquid.svg")
    tabs = dbc.Tabs(id="tabs", active_tab="accounts", className="mb-3", children=[
        dbc.Tab(label="Cuentas y posiciones", tab_id="accounts"),
        dbc.Tab(label="Mapa de liquidaciones", tab_id="liqmap"),
        dbc.Tab(label="Liquidaciones", tab_id="liquidations"),
        dbc.Tab(label="Insurance Fund · ADL", tab_id="adl"),
        dbc.Tab(label="Feed Hyperliquid", tab_id="feed"),
    ])
    panels = {
        "accounts": [
            html.Div(id="acc-note", className="warn mono mb-2"),
            _panel("Health factor por cuenta (Equity / Margen de mantenimiento)", _graph("acc-health-graph"),
                   class_name="mb-3"),
            _panel("Cuentas", ui.data_table("acc-table", ui.ACCOUNT_COLUMNS, style_data_conditional=ui.state_styles()
                                            + ui.signed_styles("unrealized_pnl", "margin_buffer")), class_name="mb-3"),
            _panel("Posiciones abiertas", ui.data_table("pos-table", ui.POSITION_COLUMNS,
                                                        style_data_conditional=ui.signed_styles("quantity",
                                                                                                "unrealized_pnl"))),
        ],
        "liqmap": [
            dbc.Row([
                dbc.Col(_panel(f"Nocional en liquidación por tramo de precio (±{LIQMAP_BAND:.0%} del mark)",
                               _graph("liqmap-graph")), lg=9, className="mb-3"),
                dbc.Col(_panel("Nocional a ±15 % · tramo pico", html.Div(id="liqmap-summary"),
                               html.Div("Precio de liquidación cross-margin de cada posición abierta (misma "
                                        "fórmula que el motor). Línea discontinua: mark actual.",
                                        className="hint text-muted-x mt-3", style={"fontSize": "11px"})),
                        lg=3, className="mb-3"),
            ]),
        ],
        "liquidations": [
            dbc.Row([
                dbc.Col(_panel("Resumen", html.Div(id="liq-stats")), lg=3, className="mb-3"),
                dbc.Col(_panel("Déficit absorbido por liquidación", _graph("liq-deficit-graph")), lg=5, className="mb-3"),
                dbc.Col(_panel("Distribución de slippage", _graph("liq-slippage-graph")), lg=4, className="mb-3"),
            ]),
            _panel("Historial de liquidaciones", ui.data_table("liq-table", ui.LIQUIDATION_COLUMNS,
                                                               style_data_conditional=ui.signed_styles("realized_pnl",
                                                                                                       "fund_delta"))),
        ],
        "adl": [
            html.Div(id="adl-callout", className="mb-3"),
            dbc.Row([
                dbc.Col(_panel("Resumen ADL", html.Div(id="adl-stats")), lg=4, className="mb-3"),
                dbc.Col(_panel("Déficit por evento: recuperado vs. residual", _graph("adl-graph")), lg=8,
                        className="mb-3"),
            ]),
            _panel("Eventos de Auto-Deleveraging", ui.data_table("adl-table", ui.ADL_COLUMNS), class_name="mb-3"),
            _panel("Posiciones desapalancadas (quién pagó el ADL)",
                   ui.data_table("adl-fills-table", ui.ADL_FILL_COLUMNS), class_name="mb-3"),
            _panel("Libro mayor del Insurance Fund", ui.data_table("ledger-table", ui.LEDGER_COLUMNS,
                                                                   style_data_conditional=ui.signed_styles("amount"))),
        ],
        "feed": [
            html.Div(id="feed-meta", className="mono text-muted-x mb-2"),
            _panel("Mercado sombra (motor) vs. Hyperliquid", ui.data_table("feed-table", ui.FEED_COLUMNS,
                                                                           style_data_conditional=ui.signed_styles(
                                                                               "deviation_bps", "basis_bps")),
                   class_name="mb-3"),
            _panel("Desviación del precio simulado respecto al real", _graph("feed-deviation-graph")),
        ],
    }
    main = html.Div([
        html.Div(id="global-alert"),
        dbc.Row(id="kpi-row", className="g-3"),
        dbc.Row([
            dbc.Col(_panel("Evolución del Insurance Fund", _graph("fund-graph")), lg=6, className="mb-3"),
            dbc.Col(_panel("Precio Hyperliquid vs. simulado", _graph("price-graph")), lg=6, className="mb-3"),
        ], className="g-3"),
        tabs,
        *[html.Div(children, id=f"panel-{key}", style={} if key == "accounts" else {"display": "none"})
          for key, children in panels.items()],
    ], className="p-3")
    return html.Div([
        html.Div([
            html.Span(["QUANT RISK TERMINAL", html.Small("AI-Driven Liquidation Engine")], className="brand"),
            html.Div(id="header-status", style={"display": "flex", "flexWrap": "wrap", "gap": "6px 14px",
                                                "alignItems": "center", "minWidth": 0}),
            html.Div(className="spacer"),
            html.Span("UTC · refresco 1 s", className="mono text-muted-x"),
        ], className="topbar"),
        html.Div([html.Div(ui.hyperliquid_brand(None, logo), id="hl-brand", className="hl-brand"),
                  html.Div(id="hl-tickers", style={"display": "flex"})], id="hl-strip", className="hl-strip"),
        dbc.Container(dbc.Row([dbc.Col(sidebar(bool(operator_password())), lg=3, xl=2, className="p-0"),
                               dbc.Col(main, lg=9, xl=10)], className="g-0"), fluid=True, className="p-0"),
        dbc.Toast(id="cmd-toast", header="Plano de control", is_open=False, dismissable=True, duration=5000,
                  icon="success", style={"position": "fixed", "top": 70, "right": 20, "zIndex": 50, "minWidth": 320}),
        dcc.Interval(id="tick", interval=refresh_ms, n_intervals=0),
    ])


# =============================================================================
# App + callbacks
# =============================================================================
def create_app(repo: DashboardRepository | None = None, *, refresh_ms: int | None = None) -> Dash:
    provider = RepositoryProvider(repo)
    tape = MarketTape()
    refresh_ms = refresh_ms or int(os.getenv("DASHBOARD_REFRESH_MS", "1000"))
    app = Dash(__name__, title="Quant Risk Terminal", update_title=None,
               external_stylesheets=[dbc.themes.CYBORG, FONTS], assets_folder=str(ROOT / "dashboard" / "assets"))
    app.layout = build_layout(app, refresh_ms)
    logo = app.get_asset_url("hyperliquid.svg")

    @app.server.get("/healthz")
    def healthz() -> tuple[str, int]:
        try:
            provider.get().control_state()
        except Exception:  # noqa: BLE001 - el healthcheck solo informa
            return "db unavailable", 503
        return "ok", 200

    @app.callback(
        Output("header-status", "children"), Output("hl-brand", "children"), Output("hl-tickers", "children"),
        Output("kpi-row", "children"), Output("fund-graph", "figure"), Output("price-graph", "figure"),
        Output("global-alert", "children"),
        Input("tick", "n_intervals"),
    )
    def refresh_main(_n: int):
        try:
            v = main_view(provider.get(), tape, logo)
        except Exception as exc:  # noqa: BLE001 - aviso en pantalla en lugar de un traceback
            log.error("dashboard.refresh_failed", error=str(exc))
            msg = html.Div(f"No se puede conectar con PostgreSQL: {exc}", className="alert-adl mb-3")
            return ui.header_status(None), no_update, no_update, no_update, no_update, no_update, msg
        return v.header, v.hl_brand, v.tickers, v.kpis, v.fund_figure, v.price_figure, v.alert

    @app.callback(
        Output("orch-status", "children"), Output("btn-normal", "disabled"), Output("btn-black-swan", "disabled"),
        Output("btn-reset", "disabled"), Output("auth-hint", "children"), Output("commands-table", "data"),
        Input("tick", "n_intervals"), Input("operator-pwd", "value"),
    )
    def refresh_controls(_n: int, password: str | None):
        try:
            v = controls_view(provider.get(), password)
        except Exception:  # noqa: BLE001
            return ui.orchestrator_status(None), True, True, True, "", []
        return v.status, v.normal_disabled, v.black_swan_disabled, v.reset_disabled, v.auth_hint, v.commands

    @app.callback(
        Output("cmd-toast", "children"), Output("cmd-toast", "icon"), Output("cmd-toast", "is_open"),
        Input("confirm-normal", "submit_n_clicks"), Input("confirm-black-swan", "submit_n_clicks"),
        Input("confirm-reset", "submit_n_clicks"),
        State("operator-pwd", "value"),
        prevent_initial_call=True,
    )
    def send_command(_normal: int | None, _bs: int | None, _reset: int | None, password: str | None):
        if ctx.triggered_id not in COMMANDS:
            raise PreventUpdate
        msg, color = dispatch_command(provider.get(), ctx.triggered_id, password)
        return msg, color, True

    @app.callback(*[Output(f"panel-{k}", "style") for k in TABS], Input("tabs", "active_tab"))
    def switch_tab(active: str):
        return [{} if k == active else {"display": "none"} for k in TABS]

    outputs = [(cid, prop) for tab in TABS for cid, prop in TAB_OUTPUTS[tab]]

    @app.callback(*[Output(cid, prop) for cid, prop in outputs], Input("tick", "n_intervals"),
                  Input("tabs", "active_tab"))
    def refresh_tab(_n: int, active: str):
        try:
            values = tab_view(provider.get(), tape, active)
        except Exception as exc:  # noqa: BLE001
            log.error("dashboard.tab_failed", tab=active, error=str(exc))
            raise PreventUpdate from exc
        return [values[cid] if cid in values else no_update for cid, _ in outputs]

    return app


app = create_app()
server = app.server  # WSGI (gunicorn dashboard.app:server)


def main() -> None:  # pragma: no cover - punto de entrada interactivo
    from common.logging_config import configure_logging

    configure_logging()
    app.run(host=os.getenv("DASHBOARD_HOST", "127.0.0.1"), port=int(os.getenv("DASHBOARD_PORT", "8501")),
            debug=os.getenv("DASHBOARD_DEBUG") == "1")


if __name__ == "__main__":  # pragma: no cover
    main()
