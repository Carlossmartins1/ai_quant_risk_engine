#!/usr/bin/env python3
"""
Healthcheck del contenedor ``engine`` (Docker HEALTHCHECK / ECS / supervisor).

* ``orchestrator``: sano si su heartbeat en ``simulation_control`` tiene menos de
  ``HEALTH_MAX_HEARTBEAT_AGE_S`` segundos (detecta procesos colgados aunque el PID siga vivo).
* ``shipper`` / ``migrate``: sano si PostgreSQL responde.

Código de salida 0 = sano, 1 = no sano (contrato de Docker).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/app")


def main() -> int:
    role = os.getenv("SERVICE_ROLE", "orchestrator")
    max_age = float(os.getenv("HEALTH_MAX_HEARTBEAT_AGE_S", "30"))
    try:
        from dataclasses import replace

        from database.connection import DatabaseManager, DBConfig

        cfg = replace(DBConfig.from_env(load_dotenv=False), pool_min=1, pool_max=1, connect_timeout_s=3,
                      application_name="healthcheck")
        db = DatabaseManager(cfg)
        try:
            if role != "orchestrator":
                db.ping()
                return 0
            with db.transaction(readonly=True) as cur:
                cur.execute("SELECT EXTRACT(EPOCH FROM now() - heartbeat_at) AS age FROM simulation_control WHERE control_id = 1")
                row = cur.fetchone()
            age = None if row is None or row["age"] is None else float(row["age"])
            if age is None or age > max_age:
                print(f"unhealthy: heartbeat age={age}", file=sys.stderr)
                return 1
            return 0
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 - cualquier fallo => no sano
        print(f"unhealthy: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
