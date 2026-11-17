"""
dashboard/components.py
=======================

Piezas de layout del terminal (cabecera, tira de Hyperliquid, KPIs, tablas) como funciones puras:
reciben datos ya cargados y devuelven componentes Dash, de modo que se testean sin servidor.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, html
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme, Sign

from ai_agents.control_plane import ControlState
from dashboard.data import FundOverview, fmt_usd
from dashboard.theme import (
    BG,
    BORDER,
    CARD,
    CRITICAL,
    FONT_MONO,
    GOOD,
    MUTED,
    TEXT,
    WARN,
    fmt_bps,
    fmt_funding,
    fmt_ms,
    fmt_pct,
    fmt_price,
)

MODE_PILL = {"NORMAL": ("Modo Normal", "normal"), "BLACK_SWAN": ("Cisne Negro", "black-swan"),
             "OFFLINE": ("Orquestador detenido", "offline")}

MONEY = FormatTemplate.money(2)
NUM3 = Format(precision=3, scheme=Scheme.fixed)
NUM1 = Format(precision=1, scheme=Scheme.fixed)
QTY = Format(precision=4, scheme=Scheme.decimal_or_exponent, sign=Sign.positive)


# ============================================================================= cabecera
def mode_pill(state: ControlState | None) -> html.Span:
    if state is None:
        return html.Span([html.Span(className="dot off"), "Sin base de datos"], className="pill offline")
    mode = state.current_mode if state.is_online() else "OFFLINE"
    label, cls = MODE_PILL.get(mode, (mode, "offline"))
    return html.Span([html.Span(className="dot " + ("on" if mode == "NORMAL" else "off" if mode == "BLACK_SWAN"
                                                    else "idle")), label], className=f"pill {cls}")


def header_status(state: ControlState | None) -> list[Any]:
    if state is None:
        return [mode_pill(None)]
    hb = "sin heartbeat" if state.heartbeat_age_s is None else f"heartbeat {state.heartbeat_age_s:,.1f}s"
    source = (state.stats or {}).get("source") or "—"
    return [mode_pill(state), html.Span(hb, className="mono text-muted-x"),
            html.Span(f"fuente {source}", className="mono text-muted-x")]


# ======================================================================= Hyperliquid
def hyperliquid_brand(state: ControlState | None, logo_src: str) -> list[Any]:
    """Logotipo + estado de la conexión al feed real (hijos del contenedor ``#hl-brand``)."""
    feed = ((state.stats or {}).get("feed") or {}) if state else {}
    if not feed.get("configured"):
        dot, text = "idle", "feed real no configurado · precios sintéticos"
    elif feed.get("connected"):
        dot, text = "on", "conectado · testnet ws"
    else:
        dot, text = "off", "desconectado · reconectando"
    return [
        html.Img(src=logo_src, alt="Hyperliquid"),
        html.Div([html.Div("HYPERLIQUID", className="name"),
                  html.Div([html.Span(className=f"dot {dot}"), " ", text], className="status")]),
    ]


def _kv(label: str, value: str, cls: str = "") -> html.Div:
    return html.Div([html.Span(label), html.B(value, className=cls)], className="row-kv")


def ticker_cells(view: pd.DataFrame) -> list[html.Div]:
    """Una celda por activo: mark real (o simulado si no hay feed), oracle, funding, desviación y latencia."""
    cells = []
    for r in view.itertuples():
        has_live = r.live_mark is not None and not pd.isna(r.live_mark)
        px = r.live_mark if has_live else r.sim_mark
        dev = r.deviation_bps
        dev_cls = "" if dev is None or pd.isna(dev) else ("down" if abs(dev) >= 50 else "warn" if abs(dev) >= 10 else "up")
        funding = r.live_funding_hourly if has_live else r.sim_funding_hourly
        fund_cls = "" if funding is None or pd.isna(funding) else ("up" if funding >= 0 else "down")
        cells.append(html.Div([
            html.Div([html.Span(f"{r.symbol}-PERP", className="sym"),
                      html.Span(" MARK " + ("REAL" if has_live else "SIM"), className="mono text-muted-x",
                                style={"fontSize": "10px"})]),
            html.Div(fmt_price(px), className="px"),
            _kv("Oracle", fmt_price(r.live_oracle if has_live else r.sim_oracle)),
            _kv("Funding 1h", fmt_funding(funding), fund_cls),
            _kv("Sim vs real", fmt_bps(dev), dev_cls),
            _kv("Latencia feed", fmt_ms(r.latency_ms)),
        ], className="ticker"))
    if not cells:
        cells.append(html.Div(html.Span("Sin estado de mercado publicado", className="text-muted-x"), className="ticker"))
    return cells


# ============================================================================= KPIs
def _kpi(label: str, value: str, sub: str = "", *, state: str = "") -> dbc.Col:
    return dbc.Col(html.Div([html.Div(label, className="label"), html.Div(value, className="value"),
                             html.Div(sub, className="sub")], className=f"kpi {state}".strip()),
                   xs=6, md=4, xl=2, className="mb-3")


def kpi_cards(fund: FundOverview, open_interest: float, risk: dict[str, Any], at_risk: int) -> list[dbc.Col]:
    depleted = fund.balance < 0
    coverage = fund.balance / open_interest if open_interest > 0 else math.nan
    var, es = risk.get("var"), risk.get("es")
    alpha = risk.get("alpha", 0.99)
    var_sub = f"{risk.get('n_scenarios', 0):,} escenarios · 1 tick" if var is not None else "calentando ventana…"
    return [
        _kpi("Insurance Fund", fmt_usd(fund.balance, compact=True),
             "DÉFICIT · ADL REQUERIDO" if depleted else f"neto {fund.net_pnl:+,.2f} USD",
             state="alert" if depleted else ""),
        _kpi("Cobertura / OI", fmt_pct(coverage), f"OI {fmt_usd(open_interest, compact=True)}",
             state="alert" if depleted else "caution" if coverage < 0.05 else ""),
        _kpi("Déficit absorbido", fmt_usd(fund.deficits_absorbed), f"fees {fmt_usd(fund.fees_collected)}",
             state="caution" if fund.deficits_absorbed > 0 else ""),
        _kpi(f"VaR {alpha:.0%}", fmt_usd(var, compact=True), var_sub),
        _kpi(f"ES {alpha:.0%}", fmt_usd(es, compact=True), "expected shortfall · cartera agregada"),
        _kpi("Liquidaciones", f"{fund.liquidations:,}",
             f"{at_risk} cuenta(s) en riesgo" if at_risk else
             (f"última {fund.last_liquidation_at:%H:%M:%S}" if fund.last_liquidation_at else "sin eventos"),
             state="caution" if at_risk else ""),
    ]


# ============================================================================ control
def orchestrator_status(state: ControlState | None) -> list[html.Div]:
    if state is None:
        return [html.Div("PostgreSQL inaccesible", className="down mono")]
    online = state.is_online()
    stats = state.stats or {}
    rows = [
        ("Estado", "ONLINE" if online else "OFFLINE", "up" if online else "down"),
        ("Modo", state.current_mode, "down" if state.current_mode == "BLACK_SWAN" else ""),
        ("Heartbeat", "—" if state.heartbeat_age_s is None else f"{state.heartbeat_age_s:,.1f}s", ""),
        ("Instancia", (state.orchestrator_id or "—")[-18:], ""),
        ("T. simulación", f"{stats.get('elapsed_s') or 0:,.0f}s", ""),
        ("Cola pendiente", f"{state.pending_commands}", "warn" if state.pending_commands else ""),
    ]
    return [html.Div([html.Span(k), html.Span(v, className=c)], className="kv") for k, v, c in rows]


# ============================================================================= tablas
def data_table(table_id: str, columns: list[dict[str, Any]], *, page_size: int = 12, **kwargs: Any) -> dash_table.DataTable:
    """DataTable con el estilo del terminal (cifras monoespaciadas alineadas a la derecha)."""
    numeric = [c["id"] for c in columns if c.get("type") == "numeric"]
    return dash_table.DataTable(
        id=table_id, columns=columns, data=[], page_size=page_size, sort_action="native",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": BG, "color": MUTED, "fontWeight": "600", "fontSize": "10.5px",
                      "textTransform": "uppercase", "letterSpacing": ".06em", "border": "none",
                      "borderBottom": f"1px solid {BORDER}"},
        style_cell={"backgroundColor": CARD, "color": TEXT, "border": "none", "borderBottom": f"1px solid {BORDER}",
                    "fontFamily": FONT_MONO, "fontSize": "12px", "padding": "6px 10px", "textAlign": "left",
                    "maxWidth": "260px", "overflow": "hidden", "textOverflow": "ellipsis"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "right"} for c in numeric],
        style_data_conditional=kwargs.pop("style_data_conditional", []),
        **kwargs,
    )


def state_styles(column: str = "state") -> list[dict[str, Any]]:
    colors = {"Liquidable": CRITICAL, "En riesgo": WARN, "Sana": GOOD}
    return [{"if": {"filter_query": f'{{{column}}} = "{k}"', "column_id": column}, "color": v, "fontWeight": "600"}
            for k, v in colors.items()]


def signed_styles(*columns: str) -> list[dict[str, Any]]:
    out = []
    for c in columns:
        out.append({"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": CRITICAL})
        out.append({"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": GOOD})
    return out


def _cell(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "strftime"):                       # datetime / pd.Timestamp -> texto UTC
        return None if pd.isna(value) else value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, Decimal):
        value = float(value)
    if hasattr(value, "item"):                           # escalares numpy
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(df: pd.DataFrame, columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DataFrame -> filas JSON-safe (fechas en texto, NaN/inf -> None) solo con las columnas visibles."""
    if df.empty:
        return []
    ids = [c["id"] for c in columns if c["id"] in df.columns]
    return [{k: _cell(v) for k, v in row.items()} for row in df[ids].to_dict("records")]


def col(name: str, cid: str, fmt: Format | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {"name": name, "id": cid}
    if fmt is not None:
        c.update(type="numeric", format=fmt)
    return c


ACCOUNT_COLUMNS = [
    col("Estado", "state"), col("Agente", "agent_name"), col("Perfil", "agent_profile"), col("Posiciones", "positions"),
    col("Health", "health_ratio", NUM3), col("Equity", "equity", MONEY), col("Margen mant.", "maintenance_margin", MONEY),
    col("Colchón", "margin_buffer", MONEY), col("Colateral", "collateral", MONEY), col("uPnL", "unrealized_pnl", MONEY),
    col("Apalanc.", "effective_leverage", NUM1), col("Nocional bruto", "gross_notional", MONEY),
]
POSITION_COLUMNS = [
    col("Agente", "agent_name"), col("Activo", "symbol"), col("Cantidad", "quantity", QTY),
    col("Entrada", "entry_price", MONEY), col("Mark", "mark_price", MONEY), col("Nocional", "notional", MONEY),
    col("uPnL", "unrealized_pnl", MONEY), col("Margen mant.", "maintenance_margin", MONEY),
    col("Precio liq.", "liquidation_price", MONEY), col("Distancia liq.", "liq_distance_pct",
                                                        Format(precision=2, scheme=Scheme.percentage)),
]
LIQUIDATION_COLUMNS = [
    col("Fecha (UTC)", "liquidated_at"), col("Agente", "agent_name"), col("Régimen", "regime"), col("Activos", "symbols"),
    col("Pos.", "positions", Format(precision=0, scheme=Scheme.fixed)), col("Equity previa", "equity_before", MONEY),
    col("Health previo", "health_before", NUM3), col("PnL realizado", "realized_pnl", MONEY), col("Fee", "fee", MONEY),
    col("Déficit", "deficit", MONEY), col("Δ Fondo", "fund_delta", MONEY),
    col("Slippage pond.", "wavg_slippage_bps", NUM1), col("Slippage máx", "max_slippage_bps", NUM1),
    col("Ejecución", "execution_source"),
]
ADL_COLUMNS = [
    col("Fecha (UTC)", "created_at"), col("Agente", "agent_name"), col("Régimen", "regime"),
    col("Déficit", "deficit", MONEY), col("No cubierto (ADL)", "shortfall", MONEY),
    col("Fondo tras evento", "balance_after", MONEY), col("Nota", "note"),
]
LEDGER_COLUMNS = [
    col("Fecha (UTC)", "created_at"), col("Tipo", "entry_type"), col("Importe", "amount", MONEY),
    col("Balance", "balance_after", MONEY), col("Agente", "agent_name"), col("Nota", "note"),
]
FEED_COLUMNS = [
    col("Activo", "symbol"), col("Mark sim.", "sim_mark", MONEY), col("Mark real", "live_mark", MONEY),
    col("Sim vs real (bps)", "deviation_bps", NUM1), col("Oracle sim.", "sim_oracle", MONEY),
    col("Oracle real", "live_oracle", MONEY), col("Basis (bps)", "basis_bps", NUM1),
    col("Spread (bps)", "spread_bps", Format(precision=2, scheme=Scheme.fixed)),
    col("Liquidez", "liquidity", Format(precision=0, scheme=Scheme.percentage)),
    col("Funding 1h sim.", "sim_funding_hourly", Format(precision=4, scheme=Scheme.percentage, sign=Sign.positive)),
    col("Funding 1h real", "live_funding_hourly", Format(precision=4, scheme=Scheme.percentage, sign=Sign.positive)),
    col("Latencia (ms)", "latency_ms", Format(precision=0, scheme=Scheme.fixed)),
    col("Antigüedad (ms)", "age_ms", Format(precision=0, scheme=Scheme.fixed)),
]
COMMAND_COLUMNS = [col("Comando", "label"), col("Estado", "status")]
