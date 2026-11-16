"""
database/migrate.py
===================

Runner de migraciones SQL idempotentes, pensado para el contenedor ``migrate`` de producción.

* Aplica ``database/schema.sql`` (base, idempotente) y después ``database/migrations/NNN_*.sql``
  en orden lexicográfico.
* Registra cada migración en ``schema_migrations`` con su checksum SHA-256: si un fichero ya
  aplicado cambia, aborta (las migraciones publicadas son inmutables; se corrige con una nueva).
* Un ``pg_advisory_lock`` impide que dos réplicas migren a la vez.

Uso::

    python -m database.migrate            # aplica pendientes
    python -m database.migrate --status   # lista aplicadas / pendientes
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import psycopg2
import structlog

from database.connection import DBConfig

log = structlog.get_logger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent
BASE_SCHEMA = ROOT / "schema.sql"
MIGRATIONS_DIR = ROOT / "migrations"
ADVISORY_LOCK_KEY = 74_210_001  # constante arbitraria del proyecto


class MigrationError(RuntimeError):
    pass


def _checksum(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migration_files() -> list[pathlib.Path]:
    return sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.name[:3].isdigit())


def migrate(config: DBConfig | None = None, *, status_only: bool = False) -> list[str]:
    """Aplica migraciones pendientes. Devuelve las versiones aplicadas en esta ejecución."""
    cfg = config or DBConfig.from_env()
    kwargs = cfg.connect_kwargs()
    kwargs.pop("options", None)  # sin statement_timeout: una migración puede tardar
    conn = psycopg2.connect(**kwargs)
    conn.autocommit = True
    applied_now: list[str] = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
            try:
                if not status_only:
                    cur.execute(BASE_SCHEMA.read_text())  # idempotente, trae su propio BEGIN/COMMIT
                cur.execute("SELECT to_regclass('schema_migrations') IS NOT NULL")
                has_table = cur.fetchone()[0]
                done: dict[str, str] = {}
                if has_table:
                    cur.execute("SELECT version, checksum FROM schema_migrations")
                    done = dict(cur.fetchall())

                for path in migration_files():
                    version, digest = path.stem, _checksum(path)
                    if version in done:
                        if done[version] != digest:
                            raise MigrationError(f"{path.name} ya aplicada pero su contenido ha cambiado")
                        continue
                    if status_only:
                        print(f"PENDIENTE  {version}")
                        continue
                    log.info("migration.apply", version=version)
                    cur.execute("BEGIN")
                    try:
                        cur.execute(path.read_text())
                        cur.execute("INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                                    (version, digest))
                        cur.execute("COMMIT")
                    except Exception:
                        cur.execute("ROLLBACK")
                        raise
                    applied_now.append(version)
                if status_only:
                    for version in sorted(done):
                        print(f"APLICADA   {version}")
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
    finally:
        conn.close()
    log.info("migration.done", applied=applied_now)
    return applied_now


def main() -> None:  # pragma: no cover
    from common.logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Migraciones del Risk Engine")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        migrate(status_only=args.status)
    except MigrationError as exc:
        log.error("migration.failed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
