"""
Fixtures compartidas.

Los tests de integración necesitan PostgreSQL (``docker compose up -d`` basta). La sesión crea una
base de datos efímera ``risk_engine_test`` con ``database.migrate`` (mismo camino que producción)
y la destruye al final.
Conexión (por orden de prioridad): ``TEST_POSTGRES_*`` y si no ``POSTGRES_*`` (.env).
Si PostgreSQL no está disponible, los tests marcados ``integration`` se omiten (skip), no fallan.
"""

from __future__ import annotations

import os
from typing import Iterable

import psycopg2
import pytest

from database.connection import DatabaseManager, DBConfig

TEST_DB = "risk_engine_test"


def _env(key: str, default: str) -> str:
    return os.getenv(f"TEST_{key}") or os.getenv(key) or default


def _conn_kwargs() -> dict:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass
    return {
        "host": _env("POSTGRES_HOST", "127.0.0.1"),
        "port": int(_env("POSTGRES_PORT", "5432")),
        "user": _env("POSTGRES_USER", "risk_admin"),
        "password": _env("POSTGRES_PASSWORD", ""),
    }


@pytest.fixture(scope="session")
def db() -> Iterable[DatabaseManager]:
    kw = _conn_kwargs()
    try:
        admin = psycopg2.connect(dbname="postgres", connect_timeout=3, **kw)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"PostgreSQL no disponible para tests de integración: {exc}")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        cur.execute(f"CREATE DATABASE {TEST_DB}")
    from database.migrate import migrate

    migrate(DBConfig(host=kw["host"], port=kw["port"], user=kw["user"], password=kw["password"], dbname=TEST_DB))

    manager = DatabaseManager(
        DBConfig(host=kw["host"], port=kw["port"], user=kw["user"], password=kw["password"],
                 dbname=TEST_DB, pool_min=1, pool_max=12, pool_timeout_s=10)
    )
    yield manager
    manager.close()
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.close()


@pytest.fixture
def clean_db(db: DatabaseManager) -> DatabaseManager:
    """Estado vacío por test. TRUNCATE no dispara los triggers append-only (son de fila)."""
    with db.transaction() as cur:
        cur.execute(
            "TRUNCATE adl_fills, adl_events, insurance_fund_ledger, liquidation_events, open_positions, "
            "users_collateral, simulation_commands RESTART IDENTITY CASCADE"
        )
        cur.execute("UPDATE insurance_fund SET balance = 0 WHERE fund_id = 1")
        cur.execute("UPDATE simulation_control SET current_mode = 'OFFLINE', heartbeat_at = NULL, "
                    "orchestrator_id = NULL, market_state = '{}', stats = '{}' WHERE control_id = 1")
    return db
