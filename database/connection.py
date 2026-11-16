"""
database/connection.py
======================

Capa de acceso transaccional a PostgreSQL para el Core Risk Engine.

Garantías que ofrece este módulo
--------------------------------
1. **Atomicidad ACID**: ``DatabaseManager.transaction()`` hace COMMIT solo si el
   bloque termina sin excepción; ante cualquier error (incluido ``KeyboardInterrupt``)
   hace ROLLBACK y devuelve la conexión limpia al pool.
2. **Bloqueo pesimista de filas**: helpers ``select_for_update`` / ``lock_account`` /
   ``lock_open_positions`` / ``lock_insurance_fund`` construyen ``SELECT ... FOR UPDATE``
   con SQL compuesto de forma segura (``psycopg2.sql``, sin interpolación de strings).
3. **Protocolo anti-deadlock**: orden de bloqueo canónico
   ``users_collateral -> open_positions (ORDER BY position_id) -> insurance_fund``.
4. **Timeouts por transacción** (``SET LOCAL``): un liquidador nunca espera
   indefinidamente un lock ni deja una transacción colgada reteniendo filas.
5. **Reintentos con backoff exponencial + jitter** ante fallos transitorios
   (``40001 serialization_failure``, ``40P01 deadlock_detected``).
6. **Pool thread-safe con espera acotada**: ``ThreadedConnectionPool`` de psycopg2
   lanza error inmediato si se agota; aquí se envuelve con un semáforo para que
   los workers esperen hasta ``pool_timeout_s`` antes de fallar.

Ejemplo (esqueleto de una liquidación atómica)
----------------------------------------------
>>> db = get_db()
>>> @retry_on_transient()
... def liquidate(user_id: int) -> None:
...     with db.transaction(lock_timeout_ms=1_000) as cur:
...         account   = lock_account(cur, user_id)            # 1º cuenta
...         positions = lock_open_positions(cur, user_id)     # 2º posiciones
...         fund      = lock_insurance_fund(cur)              # 3º fondo (fila caliente, lo último)
...         ...  # calcular equity, bankruptcy price, fee, déficit; UPDATE/INSERT
...     # COMMIT automático al salir del bloque
"""

from __future__ import annotations

import atexit
import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

import psycopg2
import psycopg2.extensions as pg_ext
import psycopg2.extras
from psycopg2 import errors as pg_errors
from psycopg2 import sql
from psycopg2.pool import ThreadedConnectionPool

__all__ = [
    "DBConfig",
    "DatabaseManager",
    "IsolationLevel",
    "LockStrength",
    "WaitPolicy",
    "DatabaseError",
    "PoolExhaustedError",
    "LockAcquisitionError",
    "TransientTransactionError",
    "StatementTimeoutError",
    "RecordNotFoundError",
    "retry_on_transient",
    "select_for_update",
    "lock_account",
    "lock_open_positions",
    "lock_insurance_fund",
    "get_db",
]

logger = logging.getLogger("risk_engine.database")

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Excepciones de dominio
# =============================================================================
class DatabaseError(Exception):
    """Raíz de los errores de la capa de datos."""


class PoolExhaustedError(DatabaseError):
    """No hubo conexión libre dentro de ``pool_timeout_s``."""


class LockAcquisitionError(DatabaseError):
    """No se pudo adquirir un lock de fila (NOWAIT o ``lock_timeout`` agotado). SQLSTATE 55P03."""


class TransientTransactionError(DatabaseError):
    """Fallo transitorio reintentable: serialization failure (40001) o deadlock (40P01)."""


class StatementTimeoutError(DatabaseError):
    """La sentencia superó ``statement_timeout``. SQLSTATE 57014."""


class RecordNotFoundError(DatabaseError):
    """La fila a bloquear no existe."""


def _translate_exception(exc: BaseException) -> BaseException:
    """Mapea errores de psycopg2 a excepciones de dominio (el original queda en ``__cause__``)."""
    if isinstance(exc, pg_errors.LockNotAvailable):
        return LockAcquisitionError(str(exc).strip())
    if isinstance(exc, (pg_errors.SerializationFailure, pg_errors.DeadlockDetected)):
        return TransientTransactionError(f"[{exc.pgcode}] {str(exc).strip()}")
    if isinstance(exc, pg_errors.QueryCanceled):
        return StatementTimeoutError(str(exc).strip())
    return exc


# =============================================================================
# Enums de control transaccional
# =============================================================================
class IsolationLevel(Enum):
    """
    READ COMMITTED + ``FOR UPDATE`` explícito es el patrón recomendado para el motor
    de liquidación: los locks de fila serializan el acceso a cada cuenta sin el coste
    de abortos masivos de SERIALIZABLE bajo alta contención.
    """

    READ_COMMITTED = pg_ext.ISOLATION_LEVEL_READ_COMMITTED
    REPEATABLE_READ = pg_ext.ISOLATION_LEVEL_REPEATABLE_READ
    SERIALIZABLE = pg_ext.ISOLATION_LEVEL_SERIALIZABLE


class LockStrength(str, Enum):
    UPDATE = "FOR UPDATE"                # exclusivo: vamos a modificar/borrar la fila
    NO_KEY_UPDATE = "FOR NO KEY UPDATE"  # exclusivo sin bloquear FKs entrantes (más concurrencia)
    SHARE = "FOR SHARE"                  # lectura estable, impide UPDATE de terceros
    KEY_SHARE = "FOR KEY SHARE"


class WaitPolicy(str, Enum):
    WAIT = ""                  # espera hasta lock_timeout
    NOWAIT = "NOWAIT"          # falla inmediatamente si la fila está bloqueada
    SKIP_LOCKED = "SKIP LOCKED"  # omite filas bloqueadas (colas de trabajo / workers paralelos)


# =============================================================================
# Configuración
# =============================================================================
def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True, slots=True)
class DBConfig:
    host: str = "127.0.0.1"
    port: int = 5432
    dbname: str = "risk_engine"
    user: str = "risk_admin"
    password: str = field(default="", repr=False)  # nunca aparece en logs/repr
    sslmode: str = "prefer"
    application_name: str = "risk_engine"
    pool_min: int = 2
    pool_max: int = 20
    pool_timeout_s: float = 5.0
    connect_timeout_s: int = 5
    statement_timeout_ms: int = 5_000
    lock_timeout_ms: int = 2_000
    idle_in_tx_timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        if not 1 <= self.pool_min <= self.pool_max:
            raise ValueError(f"Pool inválido: min={self.pool_min}, max={self.pool_max}")
        if self.pool_timeout_s <= 0:
            raise ValueError("pool_timeout_s debe ser > 0")

    @classmethod
    def from_env(cls, *, load_dotenv: bool = True) -> "DBConfig":
        """Construye la configuración desde variables de entorno (y ``.env`` si existe)."""
        if load_dotenv:
            try:
                from dotenv import load_dotenv as _load

                _load(override=False)
            except ImportError:  # python-dotenv es opcional en runtime
                pass

        d = cls()  # valores por defecto (con slots=True no son atributos de clase)
        return cls(
            host=_env("POSTGRES_HOST", d.host),
            port=int(_env("POSTGRES_PORT", str(d.port))),
            dbname=_env("POSTGRES_DB", d.dbname),
            user=_env("POSTGRES_USER", d.user),
            password=_env("POSTGRES_PASSWORD", ""),
            sslmode=_env("POSTGRES_SSLMODE", d.sslmode),
            application_name=_env("DB_APPLICATION_NAME", d.application_name),
            pool_min=int(_env("DB_POOL_MIN", str(d.pool_min))),
            pool_max=int(_env("DB_POOL_MAX", str(d.pool_max))),
            pool_timeout_s=float(_env("DB_POOL_TIMEOUT_S", str(d.pool_timeout_s))),
            connect_timeout_s=int(_env("DB_CONNECT_TIMEOUT_S", str(d.connect_timeout_s))),
            statement_timeout_ms=int(_env("DB_STATEMENT_TIMEOUT_MS", str(d.statement_timeout_ms))),
            lock_timeout_ms=int(_env("DB_LOCK_TIMEOUT_MS", str(d.lock_timeout_ms))),
            idle_in_tx_timeout_ms=int(_env("DB_IDLE_IN_TX_TIMEOUT_MS", str(d.idle_in_tx_timeout_ms))),
        )

    def connect_kwargs(self) -> dict[str, Any]:
        """Parámetros libpq. Los timeouts de sesión viajan en ``options`` (aplican desde el handshake)."""
        options = " ".join(
            (
                "-c timezone=UTC",
                f"-c statement_timeout={self.statement_timeout_ms}",
                f"-c lock_timeout={self.lock_timeout_ms}",
                f"-c idle_in_transaction_session_timeout={self.idle_in_tx_timeout_ms}",
            )
        )
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "sslmode": self.sslmode,
            "application_name": self.application_name,
            "connect_timeout": self.connect_timeout_s,
            "options": options,
            # TCP keepalives: detecta conexiones muertas (NAT/ELB de AWS cortan conexiones ociosas)
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }


# =============================================================================
# Gestor de conexiones y transacciones
# =============================================================================
class DatabaseManager:
    """Pool de conexiones thread-safe + gestor de contexto transaccional."""

    def __init__(self, config: DBConfig | None = None) -> None:
        self._config = config or DBConfig.from_env()
        self._pool: ThreadedConnectionPool | None = None
        self._init_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self._config.pool_max)

    # ------------------------------------------------------------------ pool
    @property
    def config(self) -> DBConfig:
        return self._config

    def _get_pool(self) -> ThreadedConnectionPool:
        if self._pool is None:
            with self._init_lock:  # double-checked locking: inicialización perezosa y única
                if self._pool is None:
                    self._pool = ThreadedConnectionPool(
                        self._config.pool_min,
                        self._config.pool_max,
                        **self._config.connect_kwargs(),
                    )
                    logger.info(
                        "Pool PostgreSQL inicializado host=%s db=%s min=%d max=%d",
                        self._config.host, self._config.dbname,
                        self._config.pool_min, self._config.pool_max,
                    )
        return self._pool

    def close(self) -> None:
        with self._init_lock:
            if self._pool is not None and not self._pool.closed:
                self._pool.closeall()
                logger.info("Pool PostgreSQL cerrado")
            self._pool = None

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ conexiones
    @contextmanager
    def connection(self) -> Iterator[pg_ext.connection]:
        """
        Presta una conexión cruda del pool (bajo nivel). Al devolverla garantiza que
        no queda ninguna transacción abierta y descarta conexiones rotas.
        Para lógica de negocio usa ``transaction()``.
        """
        if not self._slots.acquire(timeout=self._config.pool_timeout_s):
            raise PoolExhaustedError(
                f"Sin conexiones libres tras {self._config.pool_timeout_s}s "
                f"(pool_max={self._config.pool_max})"
            )
        pool: ThreadedConnectionPool | None = None
        conn: pg_ext.connection | None = None
        discard = False
        try:
            pool = self._get_pool()
            conn = pool.getconn()
            yield conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            discard = True  # conexión potencialmente corrupta: no se recicla
            raise
        finally:
            try:
                if pool is not None and conn is not None:
                    if conn.closed:
                        discard = True
                    elif not discard and conn.info.transaction_status != pg_ext.TRANSACTION_STATUS_IDLE:
                        # Red de seguridad: nunca devolver al pool una conexión "sucia"
                        try:
                            conn.rollback()
                        except psycopg2.Error:
                            discard = True
                    pool.putconn(conn, close=discard)
            finally:
                self._slots.release()

    @contextmanager
    def transaction(
        self,
        *,
        isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
        readonly: bool = False,
        lock_timeout_ms: int | None = None,
        statement_timeout_ms: int | None = None,
        cursor_factory: type[pg_ext.cursor] = psycopg2.extras.RealDictCursor,
    ) -> Iterator[pg_ext.cursor]:
        """
        Unidad de trabajo ACID. COMMIT al salir sin error; ROLLBACK ante cualquier excepción.

        Los timeouts se aplican con ``SET LOCAL`` (vía ``set_config(..., true)``), por lo que
        mueren con la transacción y no contaminan la conexión al volver al pool.
        """
        with self.connection() as conn:
            conn.set_session(isolation_level=isolation.value, readonly=readonly, autocommit=False)
            cur = conn.cursor(cursor_factory=cursor_factory)
            started = time.perf_counter()
            try:
                if lock_timeout_ms is not None:
                    cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{int(lock_timeout_ms)}ms",))
                if statement_timeout_ms is not None:
                    cur.execute(
                        "SELECT set_config('statement_timeout', %s, true)", (f"{int(statement_timeout_ms)}ms",)
                    )
                yield cur
                conn.commit()  # bajo SERIALIZABLE el COMMIT también puede fallar con 40001
                logger.debug("COMMIT en %.2f ms", (time.perf_counter() - started) * 1e3)
            except BaseException as exc:
                if not conn.closed:
                    try:
                        conn.rollback()
                    except psycopg2.Error:
                        logger.exception("ROLLBACK fallido; la conexión se descartará")
                logger.debug("ROLLBACK por %s: %s", type(exc).__name__, exc)
                translated = _translate_exception(exc)
                if translated is exc:
                    raise
                raise translated from exc
            finally:
                if not cur.closed:
                    cur.close()

    # ---------------------------------------------------------------- utils
    def ping(self) -> dict[str, Any]:
        """Health-check: versión del servidor y latencia de ida y vuelta."""
        t0 = time.perf_counter()
        with self.transaction(readonly=True) as cur:
            cur.execute("SELECT version() AS version, now() AS server_time, current_database() AS db")
            row = dict(cur.fetchone())
        row["latency_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        return row


# =============================================================================
# Reintentos ante fallos transitorios
# =============================================================================
def retry_on_transient(
    max_attempts: int = 5,
    base_delay_s: float = 0.05,
    max_delay_s: float = 1.0,
    *,
    retry_on_lock_timeout: bool = False,
) -> Callable[[F], F]:
    """
    Reintenta la función COMPLETA (la transacción entera, no una sentencia suelta)
    con backoff exponencial y "full jitter" para evitar reintentos sincronizados.

    La función decorada debe ser idempotente a nivel de transacción: como el ROLLBACK
    deshace todo, re-ejecutarla desde el principio es seguro.
    """
    retriable: tuple[type[BaseException], ...] = (TransientTransactionError,)
    if retry_on_lock_timeout:
        retriable += (LockAcquisitionError,)

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retriable as exc:
                    if attempt == max_attempts:
                        logger.error("%s agotó %d intentos: %s", fn.__qualname__, max_attempts, exc)
                        raise
                    delay = random.uniform(0, min(max_delay_s, base_delay_s * 2 ** (attempt - 1)))
                    logger.warning(
                        "%s intento %d/%d fallido (%s). Reintento en %.0f ms",
                        fn.__qualname__, attempt, max_attempts, type(exc).__name__, delay * 1e3,
                    )
                    time.sleep(delay)
            raise AssertionError("inalcanzable")  # pragma: no cover

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Helpers de bloqueo de filas (SELECT ... FOR UPDATE)
# =============================================================================
def _locking_clause(strength: LockStrength, wait: WaitPolicy, of_tables: Sequence[str] | None) -> sql.Composable:
    parts: list[sql.Composable] = [sql.SQL(strength.value)]
    if of_tables:
        parts.append(sql.SQL("OF ") + sql.SQL(", ").join(map(sql.Identifier, of_tables)))
    if wait is not WaitPolicy.WAIT:
        parts.append(sql.SQL(wait.value))
    return sql.SQL(" ").join(parts)


def select_for_update(
    cur: pg_ext.cursor,
    table: str,
    *,
    where: Mapping[str, Any],
    columns: Sequence[str] | None = None,
    order_by: Sequence[str] | None = None,
    limit: int | None = None,
    strength: LockStrength = LockStrength.UPDATE,
    wait: WaitPolicy = WaitPolicy.WAIT,
) -> list[dict[str, Any]]:
    """
    ``SELECT <cols> FROM <table> WHERE ... [ORDER BY ...] [LIMIT n] FOR UPDATE [NOWAIT|SKIP LOCKED]``

    * Identificadores escapados con ``sql.Identifier`` y valores con placeholders: inmune a SQL injection.
    * Un valor list/tuple genera ``col = ANY(%s)``.
    * ``where`` vacío está prohibido: impide bloquear una tabla entera por accidente.
    * Usa ``order_by`` por PK al bloquear varias filas para mantener un orden determinista (anti-deadlock).
    """
    if cur.connection.autocommit:
        raise DatabaseError("FOR UPDATE fuera de una transacción no retiene el lock: usa db.transaction()")
    if not where:
        raise ValueError("select_for_update requiere al menos una condición en 'where'")

    params: dict[str, Any] = {}
    conditions: list[sql.Composable] = []
    for i, (col, value) in enumerate(where.items()):
        key = f"w{i}"
        if isinstance(value, (list, tuple)):
            conditions.append(sql.SQL("{} = ANY({})").format(sql.Identifier(col), sql.Placeholder(key)))
            params[key] = list(value)
        elif value is None:
            conditions.append(sql.SQL("{} IS NULL").format(sql.Identifier(col)))
        else:
            conditions.append(sql.SQL("{} = {}").format(sql.Identifier(col), sql.Placeholder(key)))
            params[key] = value

    query = sql.SQL("SELECT {cols} FROM {table} WHERE {conds}").format(
        cols=sql.SQL(", ").join(map(sql.Identifier, columns)) if columns else sql.SQL("*"),
        table=sql.Identifier(table),
        conds=sql.SQL(" AND ").join(conditions),
    )
    if order_by:
        query += sql.SQL(" ORDER BY ") + sql.SQL(", ").join(map(sql.Identifier, order_by))
    if limit is not None:
        query += sql.SQL(" LIMIT {}").format(sql.Literal(int(limit)))
    query += sql.SQL(" ") + _locking_clause(strength, wait, None)

    cur.execute(query, params)
    return [dict(r) for r in cur.fetchall()]


def lock_account(
    cur: pg_ext.cursor, user_id: int, *, wait: WaitPolicy = WaitPolicy.WAIT
) -> dict[str, Any]:
    """Paso 1 del protocolo: bloquea la cuenta de margen. Serializa toda operación sobre el usuario."""
    rows = select_for_update(cur, "users_collateral", where={"user_id": user_id}, wait=wait)
    if not rows:
        raise RecordNotFoundError(f"users_collateral.user_id={user_id} no existe")
    return rows[0]


def lock_open_positions(
    cur: pg_ext.cursor, user_id: int, *, wait: WaitPolicy = WaitPolicy.WAIT
) -> list[dict[str, Any]]:
    """Paso 2: bloquea TODAS las posiciones abiertas del usuario en orden de PK (usa idx_user_risk)."""
    return select_for_update(
        cur,
        "open_positions",
        where={"user_id": user_id, "status": "OPEN"},
        order_by=("position_id",),
        wait=wait,
    )


def lock_insurance_fund(cur: pg_ext.cursor, *, wait: WaitPolicy = WaitPolicy.WAIT) -> dict[str, Any]:
    """Paso 3 (último): la fila del fondo es el punto de máxima contención; se bloquea lo más tarde posible."""
    rows = select_for_update(cur, "insurance_fund", where={"fund_id": 1}, wait=wait)
    if not rows:
        raise RecordNotFoundError("insurance_fund no inicializado (¿se ejecutó schema.sql?)")
    return rows[0]


# =============================================================================
# Singleton del proceso
# =============================================================================
_default_db: DatabaseManager | None = None
_default_db_lock = threading.Lock()


def get_db(config: DBConfig | None = None) -> DatabaseManager:
    """Devuelve el ``DatabaseManager`` compartido del proceso (creado una sola vez)."""
    global _default_db
    if _default_db is None:
        with _default_db_lock:
            if _default_db is None:
                _default_db = DatabaseManager(config)
                atexit.register(_default_db.close)
    return _default_db


# =============================================================================
# Smoke test:  python -m database.connection
# =============================================================================
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    db = get_db()
    info = db.ping()
    logger.info("Conectado a %s | latencia %.2f ms", info["db"], info["latency_ms"])
    with db.transaction(readonly=True) as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        logger.info("Tablas: %s", [r["tablename"] for r in cur.fetchall()])
        cur.execute("SELECT balance FROM insurance_fund WHERE fund_id = 1")
        logger.info("Insurance Fund balance: %s", cur.fetchone()["balance"])
