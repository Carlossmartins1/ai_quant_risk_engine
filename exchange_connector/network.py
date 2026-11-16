"""
exchange_connector/network.py
=============================

Endpoints de Hyperliquid y guardarraíl anti-mainnet.

El proyecto opera EXCLUSIVAMENTE en Testnet con fondos ficticios. Apuntar a Mainnet
exige una doble confirmación explícita (``HL_NETWORK=mainnet`` + ``HL_ALLOW_MAINNET=true``)
para que un error de configuración nunca pueda mover fondos reales.
"""

from __future__ import annotations

import os
from enum import Enum

from hyperliquid.utils import constants as hl_constants

__all__ = ["Network", "MainnetNotAllowedError", "resolve_network"]


class MainnetNotAllowedError(RuntimeError):
    """Se intentó usar Mainnet sin la confirmación explícita ``HL_ALLOW_MAINNET=true``."""


class Network(str, Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet"

    @property
    def api_url(self) -> str:
        return hl_constants.TESTNET_API_URL if self is Network.TESTNET else hl_constants.MAINNET_API_URL

    @property
    def ws_url(self) -> str:
        return self.api_url.replace("https://", "wss://", 1) + "/ws"


def resolve_network(value: str | Network | None = None, *, allow_mainnet: bool | None = None) -> Network:
    """Resuelve la red (por defecto ``HL_NETWORK`` o testnet) aplicando el guardarraíl."""
    raw = value if value is not None else os.getenv("HL_NETWORK", Network.TESTNET.value)
    try:
        network = Network(str(raw.value if isinstance(raw, Network) else raw).strip().lower())
    except ValueError as exc:
        raise ValueError(f"HL_NETWORK inválido: {raw!r} (usa 'testnet' o 'mainnet')") from exc

    if allow_mainnet is None:
        allow_mainnet = os.getenv("HL_ALLOW_MAINNET", "false").strip().lower() == "true"
    if network is Network.MAINNET and not allow_mainnet:
        raise MainnetNotAllowedError(
            "Mainnet bloqueado. Este motor está diseñado para Testnet; "
            "define HL_ALLOW_MAINNET=true solo si sabes exactamente lo que haces."
        )
    return network
