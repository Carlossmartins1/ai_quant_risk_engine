"""
dashboard/theme.py
==================

Identidad visual del terminal de riesgo: paleta oscura institucional, plantilla Plotly y formateadores.

Reglas de color: los estados (verde/ámbar/rojo) se reservan para salud y alertas; las series usan
azul (simulado / motor) y menta Hyperliquid (referencia real), de modo que nunca compiten entre sí.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import plotly.io as pio

BG = "#0B0E11"
CARD = "#151A21"
BORDER = "#262D3A"
GRID = "#1E2530"
TEXT = "#E6EAF0"
MUTED = "#8A94A6"

SIM = "#4C8DFF"            # mercado sombra / motor
LIVE = "#97FCE4"           # Hyperliquid (referencia real)
GOOD = "#16C784"
WARN = "#F0B90B"
CRITICAL = "#EA3943"
NEUTRAL = "#5D6778"
ADL = "#E87BA4"            # recuperado por Auto-Deleveraging

LEDGER_COLORS = {"CAPITAL_INJECTION": SIM, "LIQUIDATION_FEE": GOOD, "BANKRUPTCY_DEFICIT": CRITICAL,
                 "ADJUSTMENT": NEUTRAL, "ADL_RECOVERY": ADL}
REGIME_COLORS = {"LIVE_TESTNET": SIM, "STRESS_MONTE_CARLO": WARN}
STATE_COLORS = {"Sana": GOOD, "En riesgo": WARN, "Liquidable": CRITICAL, "Sin precio": NEUTRAL,
                "Sin posiciones": NEUTRAL}

FONT_UI = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace"

TEMPLATE = "risk_terminal"
pio.templates[TEMPLATE] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT_UI, color=TEXT, size=12),
        colorway=[SIM, LIVE, WARN, NEUTRAL],
        margin=dict(l=16, r=16, t=36, b=32),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG, bordercolor=BORDER, font=dict(family=FONT_MONO, color=TEXT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=MUTED)),
        xaxis=dict(gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER, automargin=True,
                   tickfont=dict(family=FONT_MONO, color=MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER, automargin=True,
                   tickfont=dict(family=FONT_MONO, color=MUTED)),
    )
)

GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def _finite(value: float | None) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def fmt_price(value: float | None) -> str:
    if not _finite(value):
        return "—"
    return f"{value:,.2f}" if abs(value) >= 100 else f"{value:,.4f}"


def fmt_bps(value: float | None, *, signed: bool = True) -> str:
    if not _finite(value):
        return "—"
    return f"{value:+.1f} bps" if signed else f"{value:.1f} bps"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "—" if not _finite(value) else f"{value:.{digits}%}"


def fmt_funding(hourly: float | None) -> str:
    """Funding horario en % con su anualización (convención de Hyperliquid: pago cada hora)."""
    if not _finite(hourly):
        return "—"
    return f"{hourly * 100:+.4f}% · {hourly * 24 * 365:+.1%} APR"


def fmt_ms(value: float | None) -> str:
    return "—" if not _finite(value) else f"{value:,.0f} ms"
