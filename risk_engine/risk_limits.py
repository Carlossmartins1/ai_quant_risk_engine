"""
risk_engine/risk_limits.py
==========================

Límites de exposición por cuenta y chequeo de cobertura del Insurance Fund.

Dos capas (defensa en profundidad)
----------------------------------
* **Cliente** — los guardarraíles del agente (``TradingAgent._guard_*``) recortan sus propias
  decisiones con los límites de su ``AgentPersona``. Un LLM puede ignorar instrucciones, así que
  esta capa es una cortesía, no un control.
* **Servidor** — ``OrderGateway`` aplica los límites persistidos en ``users_collateral`` bajo el lock
  de la cuenta, en la misma transacción que la orden. Es el control autoritativo.

Cobertura de estrés (pre-flight)
--------------------------------
Peor déficit de una cuenta ante un salto adverso instantáneo ``J`` (sin tiempo para liquidar):

    cap_i      = min(max_leverage_i · collateral_i,  max_gross_notional_i)
    deficit_i  = max(cap_i · J − collateral_i, 0)
    coverage   = insurance_fund / Σ deficit_i          (≥ 1 => el fondo absorbe el escenario sin ADL)

Con la calibración del incidente (manipulador 5x sobre $5M, sin tope) y ``J = 30 %`` la cobertura
era 0.04x: el chequeo lo habría detectado antes de arrancar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

__all__ = ["RiskLimits", "LimitBreach", "StressCoverage", "stress_deficit", "stress_coverage"]


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_leverage: float
    max_gross_notional_usd: float | None = None

    def __post_init__(self) -> None:
        if not self.max_leverage > 0:
            raise ValueError("max_leverage debe ser > 0")
        if self.max_gross_notional_usd is not None and not self.max_gross_notional_usd > 0:
            raise ValueError("max_gross_notional_usd debe ser > 0 o None")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "RiskLimits":
        cap = row.get("max_gross_notional_usd")
        return cls(float(row["max_leverage"]), None if cap is None else float(cap))

    def gross_cap(self, equity: float) -> float:
        """Nocional bruto máximo con esta equity (0 si la equity no es positiva)."""
        by_leverage = self.max_leverage * max(equity, 0.0)
        return by_leverage if self.max_gross_notional_usd is None else min(by_leverage, self.max_gross_notional_usd)

    def headroom(self, gross: float, equity: float) -> float:
        """Nocional adicional que aún cabe (≥ 0)."""
        return max(self.gross_cap(equity) - gross, 0.0)

    def breach(self, gross_after: float, equity_after: float, *, tol: float = 1e-6) -> "LimitBreach | None":
        """Motivo de rechazo si la exposición post-trade excede algún límite; ``None`` si cumple."""
        if self.max_gross_notional_usd is not None and gross_after > self.max_gross_notional_usd * (1 + tol):
            return LimitBreach("MAX_GROSS_NOTIONAL", gross_after, self.max_gross_notional_usd)
        if equity_after <= 0 or gross_after > self.max_leverage * equity_after * (1 + tol):
            lev = math.inf if equity_after <= 0 else gross_after / equity_after
            return LimitBreach("MAX_LEVERAGE", lev, self.max_leverage)
        return None


@dataclass(frozen=True, slots=True)
class LimitBreach:
    limit: str           # MAX_GROSS_NOTIONAL | MAX_LEVERAGE
    value: float
    threshold: float

    def __str__(self) -> str:
        if self.limit == "MAX_GROSS_NOTIONAL":
            return f"límite de nocional bruto excedido ({self.value:,.0f} > {self.threshold:,.0f} USD)"
        return f"límite de apalancamiento excedido ({self.value:.2f}x > {self.threshold:.2f}x)"


def stress_deficit(collateral: float, limits: RiskLimits, jump: float) -> float:
    """Peor bad debt de una cuenta al máximo de sus límites ante un salto adverso ``jump`` (fracción)."""
    return max(limits.gross_cap(collateral) * jump - collateral, 0.0)


@dataclass(frozen=True, slots=True)
class StressCoverage:
    jump: float
    insurance_fund: float
    worst_deficit: float
    by_account: Mapping[str, float]

    @property
    def ratio(self) -> float:
        return math.inf if self.worst_deficit <= 0 else self.insurance_fund / self.worst_deficit

    @property
    def adequate(self) -> bool:
        return self.ratio >= 1.0

    def as_dict(self) -> dict[str, Any]:
        return {"jump": self.jump, "insurance_fund": round(self.insurance_fund, 2),
                "worst_deficit": round(self.worst_deficit, 2),
                "ratio": None if math.isinf(self.ratio) else round(self.ratio, 4),
                "by_account": {k: round(v, 2) for k, v in self.by_account.items()}}


def stress_coverage(accounts: Iterable[tuple[str, float, RiskLimits]], insurance_fund: float,
                    jump: float = 0.30) -> StressCoverage:
    """``accounts``: (nombre, colateral, límites). Ver docstring del módulo."""
    if not 0 < jump < 1:
        raise ValueError("jump debe estar en (0, 1)")
    by_account = {name: stress_deficit(c, lim, jump) for name, c, lim in accounts}
    return StressCoverage(jump, float(insurance_fund), sum(by_account.values()), by_account)
