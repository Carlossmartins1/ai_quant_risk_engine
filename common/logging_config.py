"""
common/logging_config.py
========================

Logging estructurado unificado (``structlog``) para todo el Risk Engine.

* **Un solo pipeline**: los logs de ``structlog`` y los de la librería estándar
  (``logging``: psycopg2, websockets, el SDK de Hyperliquid, ``database/connection.py``)
  se renderizan con los mismos procesadores -> mismo formato, mismos campos.
* **JSON en producción** (``LOG_FORMAT=json``): una línea por evento, lista para
  CloudWatch / Datadog / Loki. **Consola coloreada** en desarrollo.
* **Redacción de secretos**: cualquier clave cuyo nombre sugiera un secreto
  (``secret``, ``private_key``, ``password``, ``token``...) se enmascara antes de emitirse.
* **Fichero rotado opcional** (``LOG_FILE``): además de stdout, JSON a fichero con rotación por
  tamaño; ``deployment/scripts/upload_logs_s3.py`` sube a S3 los ficheros ya rotados.
* **Contexto por tarea** vía ``structlog.contextvars`` (p.ej. ``bind_contextvars(agent="whale")``)
  que se propaga entre corrutinas asyncio.

Uso::

    from common.logging_config import configure_logging, get_logger
    configure_logging()                 # una vez, al arrancar el proceso
    log = get_logger(__name__)
    log.info("order.submitted", coin="BTC", size="0.01")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from typing import Any, MutableMapping

import structlog

__all__ = ["configure_logging", "get_logger"]

_SENSITIVE_MARKERS = ("secret", "private", "password", "passwd", "token", "api_key", "apikey", "mnemonic", "seed")
_NOISY_LOGGERS = ("websockets", "urllib3", "asyncio", "hpack", "httpx", "openai")
_configured = False
_config_lock = threading.Lock()


def _redact_secrets(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Procesador: enmascara valores cuyo nombre de campo sugiere un secreto."""
    for key in list(event_dict):
        if any(marker in key.lower() for marker in _SENSITIVE_MARKERS):
            event_dict[key] = "***REDACTED***"
    return event_dict


def configure_logging(
    level: str | int | None = None,
    *,
    json_logs: bool | None = None,
    service: str = "risk_engine",
    force: bool = False,
) -> None:
    """
    Configura structlog + logging estándar. Idempotente (salvo ``force=True``).

    Args:
        level: nivel (por defecto ``LOG_LEVEL`` o ``INFO``).
        json_logs: ``True`` -> JSON; ``False`` -> consola. Por defecto ``LOG_FORMAT=json|console``
            y, si no está definido, JSON cuando stdout no es una TTY (contenedor/EC2).
        service: nombre de servicio añadido a cada evento.
    """
    global _configured
    with _config_lock:
        if _configured and not force:
            return

        level = level or os.getenv("LOG_LEVEL", "INFO")
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
        if json_logs is None:
            fmt = os.getenv("LOG_FORMAT", "").lower()
            json_logs = fmt == "json" if fmt else not sys.stdout.isatty()

        shared: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.CallsiteParameterAdder(
                {structlog.processors.CallsiteParameter.THREAD_NAME}
            ),
            _redact_secrets,
            structlog.stdlib.ExtraAdder(),
            structlog.processors.StackInfoRenderer(),
        ]

        renderer: Any
        if json_logs:
            # show_locals=False: las variables locales de un traceback podrían contener secretos
            shared.append(
                structlog.processors.ExceptionRenderer(
                    structlog.tracebacks.ExceptionDictTransformer(show_locals=False)
                )
            )
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

        structlog.configure(
            processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                lambda _, __, ed: {**ed, "service": service},
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)

        log_file = os.getenv("LOG_FILE")
        if log_file:  # siempre JSON en fichero: es lo que se archiva en S3
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=int(os.getenv("LOG_FILE_MAX_BYTES", str(20 * 1024 * 1024))),
                backupCount=int(os.getenv("LOG_FILE_BACKUPS", "50")),
                encoding="utf-8",
            )
            file_handler.setFormatter(
                structlog.stdlib.ProcessorFormatter(
                    foreign_pre_chain=shared,
                    processors=[
                        lambda _, __, ed: {**ed, "service": service},
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.format_exc_info,
                        structlog.processors.JSONRenderer(),
                    ],
                )
            )
            root.addHandler(file_handler)
        root.setLevel(level)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(max(level, logging.WARNING))

        _configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Logger estructurado con contexto inicial opcional."""
    return structlog.get_logger(name, **initial_context)
