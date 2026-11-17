"""
dashboard/figures.py
====================

Gráficos Plotly del terminal. Todos fijan ``uirevision``: al refrescarse cada segundo conservan el
zoom/pan y la leyenda que haya elegido el operador (sin parpadeos ni resets de vista).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.data import HEALTH_WARNING
from dashboard.theme import (
    BG,
    BORDER,
    CARD,
    CRITICAL,
    FONT_MONO,
    LEDGER_COLORS,
    LIVE,
    MUTED,
    REGIME_COLORS,
    SIM,
    STATE_COLORS,
    TEMPLATE,
    WARN,
)

HEALTH_CAP = 10.0
TEXT_LINE = "#E6EAF0"


def _base(fig: go.Figure, key: str, *, height: int) -> go.Figure:
    fig.update_layout(template=TEMPLATE, uirevision=key, height=height)
    return fig


def empty_figure(message: str, *, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=13), xref="paper", yref="paper",
                       x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, f"empty:{message}", height=height)


def _padded_range(values: pd.Series, *, include_zero: bool = False, rel: float = 0.02) -> list[float]:
    """Rango Y con margen proporcional; evita ejes degenerados (serie plana o un solo punto)."""
    lo, hi = float(values.min()), float(values.max())
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = max((hi - lo) * 0.08, max(abs(lo), abs(hi)) * rel, 1.0)
    return [lo - pad, hi + pad]


def fund_figure(ledger: pd.DataFrame) -> go.Figure:
    """Balance del Insurance Fund (escalonado) + cada movimiento del libro mayor coloreado por tipo."""
    if ledger.empty:
        return empty_figure("Sin movimientos en el Insurance Fund")
    fig = go.Figure(go.Scatter(
        x=ledger["created_at"], y=ledger["balance_after"], mode="lines", line=dict(color=SIM, width=2, shape="hv"),
        name="Balance", hovertemplate="%{y:$,.2f}<extra>Balance</extra>",
    ))
    for entry_type, grp in ledger.groupby("entry_type", sort=False):
        agent = grp["agent_name"].fillna("—")
        fig.add_trace(go.Scatter(
            x=grp["created_at"], y=grp["balance_after"], mode="markers", name=entry_type,
            marker=dict(size=8, color=LEDGER_COLORS.get(entry_type, MUTED), line=dict(color=BORDER, width=1)),
            customdata=np.column_stack([grp["amount"], agent]),
            hovertemplate="%{customdata[0]:+$,.2f} · %{customdata[1]}<extra>" + entry_type + "</extra>",
        ))
    if ledger["balance_after"].min() < 0:
        fig.add_hline(y=0, line=dict(color=CRITICAL, dash="dash", width=1),
                      annotation=dict(text="ADL", font=dict(color=CRITICAL, family=FONT_MONO)))
    # Escala dinámica: ticks abreviados (−$38.6M, $100k) para que convivan valores extremos sin cortar
    # etiquetas; el hover conserva el importe exacto. Selector de ventana temporal para hacer zoom rápido.
    fig.update_yaxes(range=_padded_range(ledger["balance_after"], include_zero=ledger["balance_after"].min() < 0),
                     tickformat="$.3~s" if ledger["balance_after"].abs().max() >= 1e6 else "$,.0f",
                     automargin=True, nticks=6)
    t0, t1 = ledger["created_at"].min(), ledger["created_at"].max()
    if (t1 - t0).total_seconds() < 60:                     # serie casi puntual: ventana mínima de 1 minuto
        fig.update_xaxes(range=[t0 - pd.Timedelta(seconds=30), t1 + pd.Timedelta(seconds=30)])
    fig.update_xaxes(rangeselector=dict(
        buttons=[dict(count=5, label="5m", step="minute", stepmode="backward"),
                 dict(count=1, label="1h", step="hour", stepmode="backward"),
                 dict(count=1, label="24h", step="day", stepmode="backward"),
                 dict(step="all", label="Todo")],
        bgcolor=BG, activecolor=BORDER, bordercolor=BORDER, borderwidth=1,
        font=dict(color=MUTED, family=FONT_MONO, size=10), x=1, xanchor="right", y=1.02, yanchor="bottom",
    ))
    fig.update_layout(legend=dict(font=dict(size=10), itemwidth=30, y=1.12), margin=dict(t=56))
    return _base(fig, "fund", height=320)


def price_figure(tape: pd.DataFrame) -> go.Figure:
    """Mark simulado (motor) vs mark real de Hyperliquid, un panel por activo."""
    if tape.empty:
        return empty_figure("Esperando heartbeats del orquestador…")
    symbols = sorted(tape["symbol"].unique())
    fig = make_subplots(rows=len(symbols), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=symbols)
    for i, sym in enumerate(symbols, start=1):
        d = tape[tape["symbol"] == sym]
        fig.add_trace(go.Scatter(x=d["t"], y=d["sim_mark"], name="Simulado (motor)", legendgroup="sim",
                                 showlegend=i == 1, line=dict(color=SIM, width=2),
                                 hovertemplate=sym + " sim %{y:,.2f}<extra></extra>"), row=i, col=1)
        if d["live_mark"].notna().any():
            fig.add_trace(go.Scatter(x=d["t"], y=d["live_mark"], name="Hyperliquid (real)", legendgroup="live",
                                     showlegend=i == 1, line=dict(color=LIVE, width=1.5, dash="dot"),
                                     hovertemplate=sym + " real %{y:,.2f}<extra></extra>"), row=i, col=1)
        prices = pd.concat([d["sim_mark"], d["live_mark"]]).dropna()
        fig.update_yaxes(tickformat=",.0f", nticks=5, row=i, col=1,
                         range=_padded_range(prices, rel=0.001) if len(prices) else None)
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=MUTED, size=11, family=FONT_MONO), x=0, xanchor="left")
    return _base(fig, "prices", height=300)


def deviation_figure(tape: pd.DataFrame) -> go.Figure:
    """Desviación del precio simulado respecto al real (bps): se dispara durante el Cisne Negro."""
    d = tape.dropna(subset=["deviation_bps"]) if not tape.empty else tape
    if d.empty:
        return empty_figure("Sin referencia real (arranca el orquestador con --live)", height=260)
    fig = go.Figure()
    for sym, grp in d.groupby("symbol"):
        fig.add_trace(go.Scatter(x=grp["t"], y=grp["deviation_bps"], name=sym, mode="lines",
                                 hovertemplate=sym + " %{y:+.1f} bps<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig.update_yaxes(title="Sim − real (bps)", ticksuffix="")
    return _base(fig, "deviation", height=260)


def health_figure(table: pd.DataFrame) -> go.Figure:
    """Health factor (Equity / MM) por cuenta con los umbrales de liquidación y alerta."""
    data = table[table["n_positions"] > 0].copy() if not table.empty else table
    if data.empty:
        return empty_figure("Ninguna cuenta con posiciones abiertas", height=220)
    data["plot"] = data["health_ratio"].clip(upper=HEALTH_CAP).fillna(0.0)
    data["label"] = data["agent_name"].fillna(data["user_id"].astype(str)).astype(str)
    data = data.sort_values("plot", ascending=False)
    fig = go.Figure(go.Bar(
        y=data["label"], x=data["plot"], orientation="h",
        marker=dict(color=[STATE_COLORS.get(s, MUTED) for s in data["state"]]),
        customdata=np.column_stack([data["health_ratio"], data["equity"], data["maintenance_margin"],
                                    data["effective_leverage"], data["state"]]),
        hovertemplate=("<b>%{y}</b> · %{customdata[4]}<br>Health %{customdata[0]:.3f}"
                       "<br>Equity %{customdata[1]:$,.2f} · MM %{customdata[2]:$,.2f}"
                       "<br>Apalancamiento %{customdata[3]:.1f}x<extra></extra>"),
    ))
    fig.add_vline(x=1.0, line=dict(color=CRITICAL, dash="dash", width=1),
                  annotation=dict(text="liq 1.0", font=dict(color=CRITICAL, family=FONT_MONO, size=10)))
    fig.add_vline(x=HEALTH_WARNING, line=dict(color=WARN, dash="dot", width=1),
                  annotation=dict(text=f"alerta {HEALTH_WARNING}", yshift=-14,
                                  font=dict(color=WARN, family=FONT_MONO, size=10)))
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_yaxes(automargin=True, tickfont=dict(family=FONT_MONO, size=11))
    fig.update_xaxes(range=[0, HEALTH_CAP * 1.02], title=f"Health factor (recortado a {HEALTH_CAP:.0f})")
    return _base(fig, "health", height=max(220, 46 * len(data) + 80))


LIQ_COLORSCALE = [[0.0, CARD], [0.02, "#12343A"], [0.35, "#2E8C84"], [0.7, LIVE], [1.0, WARN]]


def liquidation_heatmap(liq_map: pd.DataFrame, *, band: float = 0.15) -> go.Figure:
    """
    Mapa de calor de liquidaciones: nocional que se liquidaría en cada tramo de precio (±band del mark),
    un panel por activo con largos/cortos en filas y el mark actual como línea vertical discontinua.
    """
    if liq_map.empty or liq_map["notional"].sum() <= 0:
        return empty_figure(f"Ninguna posición abierta con precio de liquidación a ±{band:.0%} del mark", height=320)
    symbols = sorted(liq_map["symbol"].unique())
    fig = make_subplots(rows=len(symbols), cols=1, vertical_spacing=0.18,
                        subplot_titles=[f"{s}-PERP · mark {liq_map.loc[liq_map['symbol'] == s, 'mark'].iloc[0]:,.2f}"
                                        for s in symbols])
    zmax = float(liq_map["notional"].max())
    for i, sym in enumerate(symbols, start=1):
        d = liq_map[liq_map["symbol"] == sym]
        mark = float(d["mark"].iloc[0])
        sides = ["Cortos", "Largos"]
        piv = {s: d[d["side"] == s].sort_values("price_mid") for s in sides}
        x = piv["Largos"]["price_mid"].to_numpy()
        z = [piv[s]["notional"].to_numpy() for s in sides]
        custom = [np.column_stack([piv[s]["price_lo"], piv[s]["price_hi"], piv[s]["positions"],
                                   (piv[s]["price_mid"] / mark - 1) * 100]) for s in sides]
        fig.add_trace(go.Heatmap(
            x=x, y=sides, z=z, customdata=custom, coloraxis="coloraxis", xgap=1, ygap=2,
            hovertemplate=("<b>%{y}</b> " + sym + "<br>Tramo %{customdata[0]:,.2f} – %{customdata[1]:,.2f}"
                           " (%{customdata[3]:+.1f}%)<br>Nocional %{z:$,.0f} · %{customdata[2]} pos.<extra></extra>"),
        ), row=i, col=1)
        fig.add_vline(x=mark, row=i, col=1, line=dict(color=TEXT_LINE, dash="dash", width=1.5))
        fig.update_xaxes(range=[mark * (1 - band), mark * (1 + band)], tickformat=",.0f", row=i, col=1)
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=MUTED, size=11, family=FONT_MONO), x=0, xanchor="left")
    fig.update_layout(hovermode="closest", coloraxis=dict(
        colorscale=LIQ_COLORSCALE, cmin=0, cmax=zmax,
        colorbar=dict(title=dict(text="Nocional", font=dict(color=MUTED, size=10)), tickformat="$.2~s",
                      tickfont=dict(family=FONT_MONO, color=MUTED, size=10), thickness=10, outlinewidth=0)))
    return _base(fig, "liqmap", height=170 * len(symbols) + 80)


def deficit_figure(liq: pd.DataFrame) -> go.Figure:
    if liq.empty:
        return empty_figure("Sin liquidaciones registradas", height=260)
    fig = go.Figure()
    for regime, grp in liq.sort_values("liquidated_at").groupby("regime"):
        fig.add_trace(go.Bar(
            x=grp["liquidated_at"], y=grp["deficit"], name=regime, marker_color=REGIME_COLORS.get(regime, MUTED),
            customdata=np.column_stack([grp["agent_name"], grp["fee"], grp["max_slippage_bps"]]),
            hovertemplate=("%{customdata[0]}<br>Déficit %{y:$,.2f} · fee %{customdata[1]:$,.2f}"
                           "<br>Slippage máx %{customdata[2]:.1f} bps<extra>" + regime + "</extra>"),
        ))
    fig.update_layout(hovermode="closest", barmode="stack")
    fig.update_yaxes(title="Déficit absorbido", tickformat="$,.0f")
    return _base(fig, "deficit", height=260)


def slippage_figure(liq: pd.DataFrame) -> go.Figure:
    data = liq["wavg_slippage_bps"].dropna() if not liq.empty else pd.Series(dtype=float)
    if data.empty:
        return empty_figure("Sin datos de slippage", height=260)
    fig = go.Figure(go.Histogram(x=data, nbinsx=20, marker_color=SIM,
                                 hovertemplate="%{x} bps · %{y} liquidaciones<extra></extra>"))
    fig.update_layout(hovermode="closest", bargap=0.08)
    fig.update_xaxes(title="Slippage ponderado (bps)")
    fig.update_yaxes(title="Liquidaciones", dtick=1)
    return _base(fig, "slippage", height=260)
