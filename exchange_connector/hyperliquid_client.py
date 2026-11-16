"""
exchange_connector/hyperliquid_client.py
========================================

Cliente REST de Hyperliquid (Testnet) sobre el SDK oficial ``hyperliquid-python-sdk``.

Responsabilidades
-----------------
* **Inicialización segura** desde variables de entorno: la clave privada se valida,
  nunca aparece en ``repr``/logs, y Mainnet está bloqueado por defecto.
* **Ejecución de órdenes** limit y market (LONG/SHORT) con:
    - redondeo conforme a las reglas del exchange (tick/lot size, 5 cifras significativas),
      siempre en la dirección conservadora (nunca más agresivo de lo pedido);
    - controles pre-trade (nocional mínimo/máximo, apalancamiento, activo permitido);
    - **idempotencia**: cada orden lleva un ``cloid`` (client order id). Ante un fallo de
      red con resultado ambiguo, se reconcilia consultando el exchange por ``cloid``
      ANTES de reintentar -> imposible duplicar una orden por un timeout.
* **Lectura de estado**: equity, márgenes (inicial y de mantenimiento cross), posiciones,
  órdenes abiertas y contexto de mercado (mark/oracle/funding) como fallback REST del WebSocket.
* **Resiliencia**: reintentos con backoff exponencial + jitter (red, 5xx, 429), token
  bucket de rate-limit, y serialización de acciones firmadas para evitar colisiones de nonce.

Variables de entorno
--------------------
``HL_NETWORK`` (testnet) · ``HL_SECRET_KEY`` (clave de la API wallet; opcional => modo solo-lectura)
``HL_ACCOUNT_ADDRESS`` (cuenta principal; obligatoria si se usa una API/agent wallet o en solo-lectura)
``HL_VAULT_ADDRESS`` · ``HL_ALLOW_MAINNET`` · ``HL_REQUEST_TIMEOUT_S`` · ``HL_DEFAULT_SLIPPAGE``
``HL_MAX_ORDER_NOTIONAL_USD`` · ``HL_ALLOWED_COINS`` (p.ej. ``BTC,ETH``) · ``HL_RATE_LIMIT_PER_MIN``

Buenas prácticas de seguridad: usa una **API wallet** (Hyperliquid -> API) en lugar de la
clave de tu wallet principal; una API wallet puede operar pero no retirar fondos.

Ejemplo
-------
>>> client = HyperliquidClient()                                   # lee el entorno
>>> client.set_leverage("BTC", 20, cross=True)
>>> r = client.place_limit_order("BTC", Side.LONG, size="0.001", price="60000")
>>> r = client.place_market_order("ETH", Side.SHORT, size="0.05", slippage=0.01)
>>> state = client.get_account_state()
>>> state.equity, state.maintenance_margin, state.is_liquidatable
"""

from __future__ import annotations

import os
import random
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP, Decimal, InvalidOperation
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, TypeVar

import requests
import structlog
from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.error import ClientError, ServerError
from hyperliquid.utils.types import Cloid

from exchange_connector.network import Network, resolve_network

__all__ = [
    "HyperliquidClient",
    "HyperliquidConfig",
    "Side",
    "TimeInForce",
    "OrderKind",
    "OrderStatus",
    "OrderResult",
    "AssetMeta",
    "ExchangePosition",
    "OpenOrder",
    "MarginSummary",
    "AccountState",
    "MarketContext",
    "ExchangeError",
    "ConfigurationError",
    "UnknownAssetError",
    "OrderValidationError",
    "ExchangeRequestError",
    "round_price",
    "round_size",
]

log = structlog.get_logger(__name__)
T = TypeVar("T")

# Reglas de precisión de Hyperliquid para perpetuos
MAX_PERP_DECIMALS = 6
MAX_SIG_FIGS = 5

_HEX_KEY_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_TRANSIENT_ERRORS = (requests.ConnectionError, requests.Timeout, ServerError)


# =============================================================================
# Excepciones
# =============================================================================
class ExchangeError(Exception):
    """Raíz de los errores del conector."""


class ConfigurationError(ExchangeError):
    """Configuración inválida o incompleta (clave, direcciones, parámetros)."""


class UnknownAssetError(ExchangeError):
    """El activo no existe en el universo del exchange o no está permitido."""


class OrderValidationError(ExchangeError):
    """La orden no supera los controles pre-trade (nunca llega al exchange)."""


class ExchangeRequestError(ExchangeError):
    """La petición falló tras agotar reintentos o fue rechazada por la API."""


# =============================================================================
# Enums y modelos de dominio (inmutables, Decimal para importes)
# =============================================================================
class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def is_buy(self) -> bool:
        return self is Side.LONG

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class TimeInForce(str, Enum):
    GTC = "Gtc"  # Good-til-cancelled: reposa en el libro
    ALO = "Alo"  # Add-liquidity-only (post-only): se rechaza si cruzaría
    IOC = "Ioc"  # Immediate-or-cancel: base de las órdenes "market"


class OrderKind(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    RESTING = "RESTING"                    # aceptada y en el libro
    FILLED = "FILLED"                      # ejecutada por completo
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # IOC con ejecución parcial (el resto se cancela)
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"                  # rechazada por el exchange (margen, nocional mínimo, IOC sin match...)
    UNKNOWN = "UNKNOWN"                    # resultado no verificable: requiere reconciliación manual


def _dec(value: Any) -> Decimal | None:
    """Convierte a Decimal vía ``str`` (evita artefactos binarios de float). ``None``/'' -> ``None``."""
    if value is None or value == "":
        return None
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Valor numérico inválido: {value!r}") from exc


def _dec0(value: Any) -> Decimal:
    result = _dec(value)
    return result if result is not None else Decimal(0)


@dataclass(frozen=True, slots=True)
class AssetMeta:
    name: str
    index: int
    sz_decimals: int
    max_leverage: int
    only_isolated: bool = False


@dataclass(frozen=True, slots=True)
class OrderResult:
    coin: str
    side: Side
    kind: OrderKind
    requested_size: Decimal
    limit_price: Decimal
    status: OrderStatus
    cloid: str
    reduce_only: bool = False
    oid: int | None = None
    filled_size: Decimal = Decimal(0)
    avg_fill_price: Decimal | None = None
    error: str | None = None
    latency_ms: float | None = None
    raw: Any = field(default=None, repr=False, compare=False)

    @property
    def is_success(self) -> bool:
        return self.status in (OrderStatus.RESTING, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)

    @property
    def signed_filled_size(self) -> Decimal:
        """Cantidad ejecutada con signo (+long / -short): convención del esquema ``open_positions``."""
        return self.filled_size if self.side.is_buy else -self.filled_size


@dataclass(frozen=True, slots=True)
class ExchangePosition:
    coin: str
    size: Decimal                      # con signo: +long / -short
    entry_price: Decimal | None
    position_value: Decimal            # nocional a mark price
    unrealized_pnl: Decimal
    return_on_equity: Decimal
    liquidation_price: Decimal | None  # None si la cuenta no es liquidable en ningún precio
    margin_used: Decimal
    leverage_type: str                 # 'cross' | 'isolated'
    leverage: Decimal
    max_leverage: int
    cum_funding_since_open: Decimal

    @property
    def side(self) -> Side:
        return Side.LONG if self.size > 0 else Side.SHORT

    @property
    def mark_price(self) -> Decimal | None:
        """Mark implícito: positionValue / |size|."""
        return (self.position_value / abs(self.size)) if self.size else None

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "ExchangePosition":
        lev = payload.get("leverage") or {}
        cum = payload.get("cumFunding") or {}
        return cls(
            coin=payload["coin"],
            size=_dec0(payload["szi"]),
            entry_price=_dec(payload.get("entryPx")),
            position_value=_dec0(payload.get("positionValue")),
            unrealized_pnl=_dec0(payload.get("unrealizedPnl")),
            return_on_equity=_dec0(payload.get("returnOnEquity")),
            liquidation_price=_dec(payload.get("liquidationPx")),
            margin_used=_dec0(payload.get("marginUsed")),
            leverage_type=str(lev.get("type", "cross")),
            leverage=_dec0(lev.get("value", 1)),
            max_leverage=int(payload.get("maxLeverage", 0)),
            cum_funding_since_open=_dec0(cum.get("sinceOpen")),
        )


@dataclass(frozen=True, slots=True)
class OpenOrder:
    coin: str
    oid: int
    side: Side
    limit_price: Decimal
    size: Decimal
    original_size: Decimal
    timestamp_ms: int
    cloid: str | None = None

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "OpenOrder":
        return cls(
            coin=payload["coin"],
            oid=int(payload["oid"]),
            side=Side.LONG if payload["side"] == "B" else Side.SHORT,
            limit_price=_dec0(payload["limitPx"]),
            size=_dec0(payload["sz"]),
            original_size=_dec0(payload.get("origSz", payload["sz"])),
            timestamp_ms=int(payload.get("timestamp", 0)),
            cloid=payload.get("cloid"),
        )


@dataclass(frozen=True, slots=True)
class MarginSummary:
    account_value: Decimal
    total_margin_used: Decimal
    total_notional_position: Decimal
    total_raw_usd: Decimal

    @classmethod
    def from_api(cls, payload: Mapping[str, Any] | None) -> "MarginSummary":
        payload = payload or {}
        return cls(
            account_value=_dec0(payload.get("accountValue")),
            total_margin_used=_dec0(payload.get("totalMarginUsed")),
            total_notional_position=_dec0(payload.get("totalNtlPos")),
            total_raw_usd=_dec0(payload.get("totalRawUsd")),
        )


@dataclass(frozen=True, slots=True)
class AccountState:
    """
    Foto del estado de margen de la cuenta en el exchange.

    Relación con el modelo del Risk Engine (Fase 1)::

        equity               = crossMarginSummary.accountValue  (colateral + uPnL)
        maintenance_margin   = crossMaintenanceMarginUsed
        liquidable  <=>  equity <= maintenance_margin
    """

    address: str
    timestamp_ms: int
    margin_summary: MarginSummary
    cross_margin_summary: MarginSummary
    cross_maintenance_margin_used: Decimal
    withdrawable: Decimal
    positions: tuple[ExchangePosition, ...]

    @property
    def equity(self) -> Decimal:
        return self.cross_margin_summary.account_value

    @property
    def maintenance_margin(self) -> Decimal:
        return self.cross_maintenance_margin_used

    @property
    def initial_margin_used(self) -> Decimal:
        return self.margin_summary.total_margin_used

    @property
    def unrealized_pnl(self) -> Decimal:
        return sum((p.unrealized_pnl for p in self.positions), Decimal(0))

    @property
    def maintenance_buffer(self) -> Decimal:
        """Colchón hasta liquidación (USD). <= 0 => liquidable."""
        return self.equity - self.maintenance_margin

    @property
    def margin_ratio(self) -> Decimal | None:
        """MM / equity (convención Hyperliquid). >= 1 => liquidable. ``None`` si equity <= 0."""
        return (self.maintenance_margin / self.equity) if self.equity > 0 else None

    @property
    def is_liquidatable(self) -> bool:
        return bool(self.positions) and self.equity <= self.maintenance_margin

    @classmethod
    def from_api(cls, address: str, payload: Mapping[str, Any]) -> "AccountState":
        positions = tuple(
            ExchangePosition.from_api(item["position"])
            for item in payload.get("assetPositions", [])
            if item.get("position") and _dec0(item["position"].get("szi")) != 0
        )
        return cls(
            address=address,
            timestamp_ms=int(payload.get("time", 0)),
            margin_summary=MarginSummary.from_api(payload.get("marginSummary")),
            cross_margin_summary=MarginSummary.from_api(payload.get("crossMarginSummary")),
            cross_maintenance_margin_used=_dec0(payload.get("crossMaintenanceMarginUsed")),
            withdrawable=_dec0(payload.get("withdrawable")),
            positions=positions,
        )


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Contexto de mercado vía REST (fallback cuando el WebSocket está caído)."""

    coin: str
    mark_price: Decimal
    oracle_price: Decimal
    mid_price: Decimal | None
    funding_rate_hourly: Decimal
    premium: Decimal | None
    open_interest: Decimal

    @property
    def funding_rate_annualized(self) -> Decimal:
        return self.funding_rate_hourly * 24 * 365


# =============================================================================
# Reglas de precisión del exchange (funciones puras, testeables)
# =============================================================================
def round_size(size: Decimal, sz_decimals: int) -> Decimal:
    """Trunca la cantidad al lot size del activo. Siempre hacia abajo: nunca más riesgo del pedido."""
    return size.quantize(Decimal(1).scaleb(-sz_decimals), rounding=ROUND_DOWN)


def round_price(price: Decimal, sz_decimals: int, *, rounding: str = ROUND_HALF_EVEN) -> Decimal:
    """
    Ajusta un precio a las reglas de Hyperliquid para perpetuos:

    * máximo 5 cifras significativas;
    * máximo ``6 - szDecimals`` decimales;
    * los precios enteros son siempre válidos (aunque superen 5 cifras significativas).
    """
    if price <= 0:
        raise OrderValidationError(f"Precio no positivo: {price}")
    if price >= Decimal(10) ** MAX_SIG_FIGS:
        return price.quantize(Decimal(1), rounding=rounding)
    exponent = max(price.adjusted() - (MAX_SIG_FIGS - 1), -(MAX_PERP_DECIMALS - sz_decimals))
    return price.quantize(Decimal(1).scaleb(exponent), rounding=rounding)


# =============================================================================
# Configuración
# =============================================================================
def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


@dataclass(frozen=True, slots=True)
class HyperliquidConfig:
    network: Network = Network.TESTNET
    secret_key: str | None = field(default=None, repr=False)   # NUNCA se loguea
    account_address: str | None = None
    vault_address: str | None = None
    allow_mainnet: bool = False
    request_timeout_s: float = 10.0
    max_retries: int = 3
    retry_backoff_s: float = 0.25
    default_slippage: float = 0.01                 # 1% (el SDK usa 5%: demasiado laxo para un motor de riesgo)
    min_order_notional_usd: Decimal = Decimal("10")      # mínimo del exchange
    max_order_notional_usd: Decimal = Decimal("250000")  # guardarraíl fat-finger
    allowed_coins: tuple[str, ...] | None = ("BTC", "ETH")
    rate_limit_per_min: int = 600                  # límite REST de HL: 1200 de peso/min por IP

    def __post_init__(self) -> None:
        network = resolve_network(self.network, allow_mainnet=self.allow_mainnet)
        object.__setattr__(self, "network", network)

        if self.secret_key is not None:
            if not _HEX_KEY_RE.match(self.secret_key):
                raise ConfigurationError("HL_SECRET_KEY no es una clave privada hex de 32 bytes")
            if not self.secret_key.startswith("0x"):
                object.__setattr__(self, "secret_key", "0x" + self.secret_key)
        for name in ("account_address", "vault_address"):
            addr = getattr(self, name)
            if addr is not None and not _ADDRESS_RE.match(addr):
                raise ConfigurationError(f"{name} no es una dirección EVM válida: {addr!r}")
        if self.secret_key is None and self.account_address is None:
            raise ConfigurationError("Define HL_SECRET_KEY (trading) o HL_ACCOUNT_ADDRESS (solo lectura)")
        if not 0 < self.default_slippage < 0.2:
            raise ConfigurationError("default_slippage debe estar en (0, 0.2)")
        if self.min_order_notional_usd >= self.max_order_notional_usd:
            raise ConfigurationError("min_order_notional_usd debe ser < max_order_notional_usd")
        if self.allowed_coins is not None:
            object.__setattr__(self, "allowed_coins", tuple(c.upper() for c in self.allowed_coins))

    @property
    def trading_enabled(self) -> bool:
        return self.secret_key is not None

    @classmethod
    def from_env(cls, *, load_dotenv: bool = True) -> "HyperliquidConfig":
        if load_dotenv:
            try:
                from dotenv import load_dotenv as _load

                _load(override=False)
            except ImportError:
                pass
        coins = _env("HL_ALLOWED_COINS")
        defaults = cls.__dataclass_fields__
        return cls(
            network=_env("HL_NETWORK") or Network.TESTNET.value,
            secret_key=_env("HL_SECRET_KEY"),
            account_address=_env("HL_ACCOUNT_ADDRESS"),
            vault_address=_env("HL_VAULT_ADDRESS"),
            allow_mainnet=(_env("HL_ALLOW_MAINNET") or "false").lower() == "true",
            request_timeout_s=float(_env("HL_REQUEST_TIMEOUT_S") or defaults["request_timeout_s"].default),
            default_slippage=float(_env("HL_DEFAULT_SLIPPAGE") or defaults["default_slippage"].default),
            max_order_notional_usd=Decimal(
                _env("HL_MAX_ORDER_NOTIONAL_USD") or defaults["max_order_notional_usd"].default
            ),
            allowed_coins=tuple(c.strip() for c in coins.split(",")) if coins else ("BTC", "ETH"),
            rate_limit_per_min=int(_env("HL_RATE_LIMIT_PER_MIN") or defaults["rate_limit_per_min"].default),
        )


# =============================================================================
# Rate limiter (token bucket thread-safe)
# =============================================================================
class _TokenBucket:
    def __init__(self, rate_per_min: int, burst: int | None = None) -> None:
        self._rate = rate_per_min / 60.0
        self._capacity = float(burst or max(5, rate_per_min // 6))
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, weight: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                if self._tokens >= weight:
                    self._tokens -= weight
                    return
                wait = (weight - self._tokens) / self._rate
            time.sleep(wait)  # fuera del lock


def _mask(address: str | None) -> str | None:
    return f"{address[:6]}…{address[-4:]}" if address else None


def _backoff(attempt: int, base: float, cap: float = 5.0) -> float:
    return random.uniform(0, min(cap, base * 2 ** (attempt - 1)))  # full jitter


# =============================================================================
# Cliente
# =============================================================================
class HyperliquidClient:
    """
    Cliente thread-safe de Hyperliquid.

    * Lecturas (``Info``): concurrentes, sujetas al rate limiter.
    * Acciones firmadas (``Exchange``): serializadas con un lock y separadas >= 1 ms
      para garantizar nonces (timestamp en ms) estrictamente crecientes.
    """

    def __init__(
        self,
        config: HyperliquidConfig | None = None,
        *,
        info: Info | None = None,
        exchange: Exchange | None = None,
    ) -> None:
        self._cfg = config or HyperliquidConfig.from_env()
        self._bucket = _TokenBucket(self._cfg.rate_limit_per_min)
        self._action_lock = threading.Lock()
        self._last_action_ms = 0

        self._wallet: LocalAccount | None = (
            Account.from_key(self._cfg.secret_key) if self._cfg.secret_key else None
        )
        # Dirección cuyo estado se consulta: vault > cuenta principal > wallet firmante
        self._address: str = (
            self._cfg.vault_address
            or self._cfg.account_address
            or (self._wallet.address if self._wallet else "")
        )
        agent_mode = bool(
            self._wallet and self._cfg.account_address
            and self._cfg.account_address.lower() != self._wallet.address.lower()
        )

        base_url = self._cfg.network.api_url
        timeout = self._cfg.request_timeout_s
        self._info: Info = info or self._call(
            "info.init", lambda: Info(base_url, skip_ws=True, timeout=timeout)
        )
        meta = self._call("info.meta", self._info.meta)
        self._assets: dict[str, AssetMeta] = {
            u["name"]: AssetMeta(
                name=u["name"],
                index=i,
                sz_decimals=int(u["szDecimals"]),
                max_leverage=int(u["maxLeverage"]),
                only_isolated=bool(u.get("onlyIsolated", False)),
            )
            for i, u in enumerate(meta["universe"])
            if not u.get("isDelisted", False)
        }

        self._exchange: Exchange | None = None
        if self._wallet is not None:
            self._exchange = exchange or self._call(
                "exchange.init",
                lambda: Exchange(
                    self._wallet,
                    base_url,
                    meta=meta,
                    account_address=self._cfg.account_address,
                    vault_address=self._cfg.vault_address,
                    timeout=timeout,
                ),
            )

        log.info(
            "hyperliquid.client.initialized",
            network=self._cfg.network.value,
            address=_mask(self._address),
            signer=_mask(self._wallet.address) if self._wallet else None,
            agent_mode=agent_mode,
            trading_enabled=self._cfg.trading_enabled,
            assets_loaded=len(self._assets),
        )

    # ------------------------------------------------------------- properties
    @property
    def address(self) -> str:
        return self._address

    @property
    def network(self) -> Network:
        return self._cfg.network

    @property
    def config(self) -> HyperliquidConfig:
        return self._cfg

    def asset(self, coin: str) -> AssetMeta:
        """Metadatos del activo, validando que existe y está en la lista permitida."""
        coin = coin.upper()
        if self._cfg.allowed_coins is not None and coin not in self._cfg.allowed_coins:
            raise UnknownAssetError(f"{coin} no está en HL_ALLOWED_COINS={self._cfg.allowed_coins}")
        try:
            return self._assets[coin]
        except KeyError:
            raise UnknownAssetError(f"{coin} no existe en el universo de {self._cfg.network.value}") from None

    # ---------------------------------------------------------- infra / retry
    def _call(self, op: str, fn: Callable[[], T], *, weight: float = 1.0) -> T:
        """Ejecuta una llamada IDEMPOTENTE con reintentos (red, 5xx, 429)."""
        attempts = self._cfg.max_retries + 1
        for attempt in range(1, attempts + 1):
            self._bucket.acquire(weight)
            try:
                return fn()
            except ClientError as exc:
                if exc.status_code != 429 or attempt == attempts:
                    raise ExchangeRequestError(
                        f"{op}: HTTP {exc.status_code} {exc.error_message}"
                    ) from exc
                reason = "rate_limited"
            except _TRANSIENT_ERRORS as exc:
                if attempt == attempts:
                    raise ExchangeRequestError(f"{op}: {type(exc).__name__} tras {attempts} intentos") from exc
                reason = type(exc).__name__
            delay = _backoff(attempt, self._cfg.retry_backoff_s)
            log.warning("hyperliquid.request.retry", op=op, attempt=attempt, reason=reason, delay_ms=round(delay * 1e3))
            time.sleep(delay)
        raise AssertionError("inalcanzable")  # pragma: no cover

    def _signed(self, fn: Callable[[], T]) -> T:
        """Serializa acciones firmadas y garantiza nonces (ms) estrictamente crecientes."""
        with self._action_lock:
            # El SDK usa el timestamp (ms) como nonce y lo toma DENTRO de fn: esperamos a que el
            # reloj supere el instante en que terminó la acción anterior (cota superior de su nonce).
            while time.time_ns() // 1_000_000 <= self._last_action_ms:
                time.sleep(0.0002)
            try:
                return fn()
            finally:
                self._last_action_ms = time.time_ns() // 1_000_000

    def _require_trading(self) -> Exchange:
        if self._exchange is None:
            raise ConfigurationError("Cliente en modo solo-lectura: define HL_SECRET_KEY para operar")
        return self._exchange

    # ----------------------------------------------------------- pre-trade
    def _validate_order(self, asset: AssetMeta, size: Decimal, ref_price: Decimal, reduce_only: bool) -> None:
        if size <= 0:
            raise OrderValidationError(
                f"Cantidad nula tras redondear a {asset.sz_decimals} decimales (lot size de {asset.name})"
            )
        notional = size * ref_price
        if not reduce_only and notional < self._cfg.min_order_notional_usd:
            raise OrderValidationError(
                f"Nocional {notional:.2f} USD < mínimo {self._cfg.min_order_notional_usd} USD"
            )
        if notional > self._cfg.max_order_notional_usd:
            raise OrderValidationError(
                f"Nocional {notional:.2f} USD > máximo permitido {self._cfg.max_order_notional_usd} USD (fat-finger)"
            )

    # -------------------------------------------------------------- órdenes
    def place_limit_order(
        self,
        coin: str,
        side: Side | str,
        size: Decimal | float | str,
        price: Decimal | float | str,
        *,
        tif: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        cloid: str | None = None,
    ) -> OrderResult:
        """
        Orden limit. El precio se redondea en dirección PASIVA (compra hacia abajo,
        venta hacia arriba): la orden nunca resulta más agresiva que lo solicitado.
        """
        asset = self.asset(coin)
        side = Side(side)
        sz = round_size(_dec0(size), asset.sz_decimals)
        px = round_price(_dec0(price), asset.sz_decimals, rounding=ROUND_DOWN if side.is_buy else ROUND_UP)
        self._validate_order(asset, sz, px, reduce_only)
        return self._submit(asset, side, OrderKind.LIMIT, sz, px, tif, reduce_only, cloid)

    def place_market_order(
        self,
        coin: str,
        side: Side | str,
        size: Decimal | float | str,
        *,
        slippage: float | None = None,
        reference_price: Decimal | float | str | None = None,
        reduce_only: bool = False,
        cloid: str | None = None,
    ) -> OrderResult:
        """
        Orden "market" = limit IOC agresiva con slippage acotado (así funciona en Hyperliquid).

        ``reference_price``: pásale el mid/mark del ``MarketDataStore`` del WebSocket para
        ahorrar una llamada REST; si se omite se consulta ``allMids``. El precio límite
        resultante nunca excede ``reference * (1 ± slippage)``.
        """
        asset = self.asset(coin)
        side = Side(side)
        slip = Decimal(str(slippage if slippage is not None else self._cfg.default_slippage))
        if not Decimal(0) < slip < Decimal("0.2"):
            raise OrderValidationError(f"Slippage fuera de rango (0, 20%): {slip}")
        ref = _dec(reference_price) if reference_price is not None else self.get_mid_price(asset.name)
        if ref is None or ref <= 0:
            raise OrderValidationError(f"Precio de referencia inválido: {ref}")

        worst = ref * (1 + slip) if side.is_buy else ref * (1 - slip)
        px = round_price(worst, asset.sz_decimals, rounding=ROUND_DOWN if side.is_buy else ROUND_UP)
        sz = round_size(_dec0(size), asset.sz_decimals)
        self._validate_order(asset, sz, ref, reduce_only)
        return self._submit(asset, side, OrderKind.MARKET, sz, px, TimeInForce.IOC, reduce_only, cloid)

    def close_position(
        self,
        coin: str,
        *,
        slippage: float | None = None,
        reference_price: Decimal | float | str | None = None,
    ) -> OrderResult | None:
        """Cierra la posición completa con una orden market reduce-only. ``None`` si no hay posición."""
        asset = self.asset(coin)
        position = next((p for p in self.get_open_positions() if p.coin == asset.name), None)
        if position is None:
            log.info("hyperliquid.close_position.no_position", coin=asset.name)
            return None
        return self.place_market_order(
            asset.name,
            position.side.opposite,
            abs(position.size),
            slippage=slippage,
            reference_price=reference_price,
            reduce_only=True,
        )

    def _submit(
        self,
        asset: AssetMeta,
        side: Side,
        kind: OrderKind,
        size: Decimal,
        price: Decimal,
        tif: TimeInForce,
        reduce_only: bool,
        cloid: str | None,
    ) -> OrderResult:
        exchange = self._require_trading()
        order_cloid = Cloid.from_str(cloid) if cloid else Cloid.from_str("0x" + secrets.token_hex(16))
        olog = log.bind(
            coin=asset.name, side=side.value, kind=kind.value, size=str(size), limit_px=str(price),
            tif=tif.value, reduce_only=reduce_only, cloid=str(order_cloid),
        )
        base = dict(
            coin=asset.name, side=side, kind=kind, requested_size=size, limit_price=price,
            cloid=str(order_cloid), reduce_only=reduce_only,
        )

        def send() -> Any:
            return self._signed(
                lambda: exchange.order(
                    asset.name, side.is_buy, float(size), float(price),
                    {"limit": {"tif": tif.value}}, reduce_only=reduce_only, cloid=order_cloid,
                )
            )

        olog.info("order.submit")
        started = time.perf_counter()
        attempts = self._cfg.max_retries + 1
        for attempt in range(1, attempts + 1):
            self._bucket.acquire()
            try:
                raw = send()
                break
            except ClientError as exc:
                if exc.status_code == 429 and attempt < attempts:
                    time.sleep(_backoff(attempt, self._cfg.retry_backoff_s))
                    continue
                result = OrderResult(
                    **base, status=OrderStatus.REJECTED, error=f"HTTP {exc.status_code}: {exc.error_message}",
                    latency_ms=_ms_since(started),
                )
                olog.error("order.rejected_http", status_code=exc.status_code, error=exc.error_message)
                return result
            except _TRANSIENT_ERRORS as exc:
                # Resultado AMBIGUO: la orden pudo llegar al matching engine. Nunca reenviar a ciegas.
                olog.warning("order.ambiguous_outcome", attempt=attempt, error=type(exc).__name__)
                reconciled = self._reconcile(order_cloid, base, started)
                if reconciled is not None:
                    olog.info("order.reconciled", status=reconciled.status.value, oid=reconciled.oid)
                    return reconciled
                if attempt == attempts:
                    olog.error("order.unknown_outcome", attempts=attempts)
                    return OrderResult(
                        **base, status=OrderStatus.UNKNOWN, latency_ms=_ms_since(started),
                        error=f"Sin respuesta tras {attempts} intentos; reconciliar por cloid",
                    )
                time.sleep(_backoff(attempt, self._cfg.retry_backoff_s))
        else:  # pragma: no cover
            raise AssertionError("inalcanzable")

        result = self._parse_order_response(raw, base, started)
        log_fn = olog.info if result.is_success else olog.warning
        log_fn(
            "order.result", status=result.status.value, oid=result.oid, filled=str(result.filled_size),
            avg_px=str(result.avg_fill_price) if result.avg_fill_price else None,
            error=result.error, latency_ms=result.latency_ms,
        )
        return result

    def _reconcile(self, cloid: Cloid, base: dict[str, Any], started: float) -> OrderResult | None:
        """
        Consulta el estado por ``cloid`` tras un fallo ambiguo.

        * Orden encontrada -> devuelve su estado real (NO se reenvía).
        * ``unknownOid``   -> el exchange confirma que no existe -> ``None`` (reintento seguro).
        * Consulta fallida -> ``UNKNOWN`` (reenviar podría duplicar la orden).
        """
        try:
            resp = self._call("info.orderStatus", lambda: self._info.query_order_by_cloid(self._address, cloid))
        except ExchangeError as exc:
            return OrderResult(
                **base, status=OrderStatus.UNKNOWN, latency_ms=_ms_since(started),
                error=f"Reconciliación imposible ({exc}); verificar cloid manualmente",
            )
        if not isinstance(resp, Mapping) or resp.get("status") != "order":
            return None  # unknownOid: la orden nunca llegó

        wrapper = resp.get("order", {})
        order = wrapper.get("order", {})
        state = str(wrapper.get("status", "")).lower()
        orig = _dec0(order.get("origSz"))
        remaining = _dec0(order.get("sz"))
        filled = max(orig - remaining, Decimal(0))
        status = {
            "open": OrderStatus.RESTING,
            "filled": OrderStatus.FILLED,
            "triggered": OrderStatus.RESTING,
        }.get(state)
        if status is None:
            if "cancel" in state:
                status = OrderStatus.PARTIALLY_FILLED if filled > 0 else OrderStatus.CANCELED
            elif "reject" in state:
                status = OrderStatus.REJECTED
            else:
                status = OrderStatus.UNKNOWN
        return OrderResult(
            **base, status=status, oid=order.get("oid"), filled_size=filled,
            latency_ms=_ms_since(started), raw=resp,
            error=None if status not in (OrderStatus.REJECTED, OrderStatus.UNKNOWN) else state,
        )

    @staticmethod
    def _parse_order_response(raw: Any, base: dict[str, Any], started: float) -> OrderResult:
        """
        Formatos de respuesta de ``/exchange``::

            {"status":"ok","response":{"type":"order","data":{"statuses":[{"resting":{"oid":1}}]}}}
            {"status":"ok", ... "statuses":[{"filled":{"totalSz":"0.01","avgPx":"95000.0","oid":2}}]}
            {"status":"ok", ... "statuses":[{"error":"Insufficient margin to place order."}]}
            {"status":"err","response":"User or API Wallet 0x... does not exist."}
        """
        latency = _ms_since(started)
        if not isinstance(raw, Mapping) or raw.get("status") != "ok":
            error = raw.get("response") if isinstance(raw, Mapping) else repr(raw)
            return OrderResult(**base, status=OrderStatus.REJECTED, error=str(error), latency_ms=latency, raw=raw)
        try:
            entry = raw["response"]["data"]["statuses"][0]
        except (KeyError, IndexError, TypeError):
            return OrderResult(
                **base, status=OrderStatus.UNKNOWN, error="Respuesta sin 'statuses'", latency_ms=latency, raw=raw
            )

        if isinstance(entry, Mapping) and "resting" in entry:
            return OrderResult(
                **base, status=OrderStatus.RESTING, oid=entry["resting"].get("oid"), latency_ms=latency, raw=raw
            )
        if isinstance(entry, Mapping) and "filled" in entry:
            fill = entry["filled"]
            filled = _dec0(fill.get("totalSz"))
            status = OrderStatus.FILLED if filled >= base["requested_size"] else OrderStatus.PARTIALLY_FILLED
            return OrderResult(
                **base, status=status, oid=fill.get("oid"), filled_size=filled,
                avg_fill_price=_dec(fill.get("avgPx")), latency_ms=latency, raw=raw,
            )
        if isinstance(entry, Mapping) and "error" in entry:
            return OrderResult(**base, status=OrderStatus.REJECTED, error=str(entry["error"]), latency_ms=latency, raw=raw)
        if entry in ("waitingForFill", "waitingForTrigger"):
            return OrderResult(**base, status=OrderStatus.RESTING, latency_ms=latency, raw=raw)
        return OrderResult(
            **base, status=OrderStatus.UNKNOWN, error=f"Estado no reconocido: {entry!r}", latency_ms=latency, raw=raw
        )

    # ----------------------------------------------------- gestión órdenes
    def cancel_order(self, coin: str, oid: int) -> bool:
        exchange = self._require_trading()
        asset = self.asset(coin)
        raw = self._call("exchange.cancel", lambda: self._signed(lambda: exchange.cancel(asset.name, int(oid))))
        ok = _first_status(raw) == "success"
        (log.info if ok else log.warning)("order.cancel", coin=asset.name, oid=oid, success=ok, raw=None if ok else raw)
        return ok

    def cancel_all_orders(self, coin: str | None = None) -> int:
        """Cancela todas las órdenes abiertas (opcionalmente de un activo). Devuelve cuántas se cancelaron."""
        exchange = self._require_trading()
        target = self.asset(coin).name if coin else None
        orders = [o for o in self.get_open_orders() if target is None or o.coin == target]
        if not orders:
            return 0
        requests_ = [{"coin": o.coin, "oid": o.oid} for o in orders]
        raw = self._call("exchange.bulk_cancel", lambda: self._signed(lambda: exchange.bulk_cancel(requests_)))
        statuses = _statuses(raw)
        cancelled = sum(1 for s in statuses if s == "success")
        log.info("order.cancel_all", coin=target, requested=len(orders), cancelled=cancelled)
        return cancelled

    def set_leverage(self, coin: str, leverage: int, *, cross: bool = True) -> bool:
        """Fija el apalancamiento del activo (cross o isolated), validado contra ``maxLeverage``."""
        exchange = self._require_trading()
        asset = self.asset(coin)
        if not 1 <= int(leverage) <= asset.max_leverage:
            raise OrderValidationError(f"Apalancamiento {leverage}x fuera de [1, {asset.max_leverage}x] para {asset.name}")
        if cross and asset.only_isolated:
            raise OrderValidationError(f"{asset.name} solo admite margen aislado")
        raw = self._call(
            "exchange.update_leverage",
            lambda: self._signed(lambda: exchange.update_leverage(int(leverage), asset.name, is_cross=cross)),
        )
        ok = isinstance(raw, Mapping) and raw.get("status") == "ok"
        (log.info if ok else log.error)(
            "account.leverage_updated", coin=asset.name, leverage=int(leverage), cross=cross, success=ok,
            raw=None if ok else raw,
        )
        return ok

    # ---------------------------------------------------------- lecturas
    def get_account_state(self) -> AccountState:
        """Equity, márgenes (inicial y mantenimiento cross), retirable y posiciones."""
        raw = self._call("info.user_state", lambda: self._info.user_state(self._address), weight=2)
        state = AccountState.from_api(self._address, raw)
        log.debug(
            "account.state", equity=str(state.equity), maintenance_margin=str(state.maintenance_margin),
            margin_ratio=str(state.margin_ratio), positions=len(state.positions),
        )
        return state

    def get_open_positions(self) -> tuple[ExchangePosition, ...]:
        return self.get_account_state().positions

    def get_open_orders(self) -> tuple[OpenOrder, ...]:
        raw = self._call("info.open_orders", lambda: self._info.open_orders(self._address))
        return tuple(OpenOrder.from_api(o) for o in raw or [])

    def get_mid_price(self, coin: str) -> Decimal:
        asset = self.asset(coin)
        mids = self._call("info.all_mids", self._info.all_mids, weight=2)
        price = _dec(mids.get(asset.name))
        if price is None:
            raise ExchangeRequestError(f"Sin mid price para {asset.name}")
        return price

    def get_market_contexts(self, coins: Iterable[str] | None = None) -> dict[str, MarketContext]:
        """Mark, oracle, funding (horario), premium y open interest vía REST."""
        wanted = {self.asset(c).name for c in coins} if coins else set(self._cfg.allowed_coins or self._assets)
        meta, ctxs = self._call("info.meta_and_asset_ctxs", self._info.meta_and_asset_ctxs, weight=2)
        result: dict[str, MarketContext] = {}
        for universe_entry, ctx in zip(meta["universe"], ctxs):
            name = universe_entry["name"]
            if name not in wanted or ctx.get("markPx") is None:
                continue
            result[name] = MarketContext(
                coin=name,
                mark_price=_dec0(ctx["markPx"]),
                oracle_price=_dec0(ctx.get("oraclePx")),
                mid_price=_dec(ctx.get("midPx")),
                funding_rate_hourly=_dec0(ctx.get("funding")),
                premium=_dec(ctx.get("premium")),
                open_interest=_dec0(ctx.get("openInterest")),
            )
        return result

    def health_check(self) -> dict[str, Any]:
        started = time.perf_counter()
        self._call("info.all_mids", self._info.all_mids, weight=2)
        return {
            "network": self._cfg.network.value,
            "address": _mask(self._address),
            "trading_enabled": self._cfg.trading_enabled,
            "latency_ms": _ms_since(started),
        }


# =============================================================================
# Utilidades de parseo
# =============================================================================
def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1e3, 3)


def _statuses(raw: Any) -> list[Any]:
    try:
        return list(raw["response"]["data"]["statuses"]) if raw.get("status") == "ok" else []
    except (KeyError, TypeError, AttributeError):
        return []


def _first_status(raw: Any) -> Any:
    statuses = _statuses(raw)
    return statuses[0] if statuses else None


# =============================================================================
# Demo:  python -m exchange_connector.hyperliquid_client
# =============================================================================
if __name__ == "__main__":  # pragma: no cover
    from common.logging_config import configure_logging

    configure_logging()
    client = HyperliquidClient()
    log.info("health", **client.health_check())
    for coin, ctx in client.get_market_contexts().items():
        log.info(
            "market.context", coin=coin, mark=str(ctx.mark_price), oracle=str(ctx.oracle_price),
            funding_1h=str(ctx.funding_rate_hourly), funding_apr=f"{ctx.funding_rate_annualized:.2%}",
        )
    st = client.get_account_state()
    log.info(
        "account", equity=str(st.equity), maintenance_margin=str(st.maintenance_margin),
        margin_ratio=str(st.margin_ratio), liquidatable=st.is_liquidatable,
        positions=[(p.coin, str(p.size), str(p.unrealized_pnl)) for p in st.positions],
    )
