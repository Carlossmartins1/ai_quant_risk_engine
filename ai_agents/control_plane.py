"""
ai_agents/control_plane.py
==========================

Plano de control entre procesos (dashboard <-> orquestador) sobre PostgreSQL.

* El **dashboard** encola comandos (``enqueue``) en ``simulation_commands``.
* El **orquestador** los consume con ``SELECT ... FOR UPDATE SKIP LOCKED`` (cada comando lo procesa
  exactamente una réplica, sin bloquear a las demás) y publica su estado en ``simulation_control``:
  modo, heartbeat y snapshot de mercado.

Sin brokers adicionales (Redis/SQS): la base de datos ya es la fuente de verdad transaccional, y la
cola queda auditada (quién pidió qué y cuándo se ejecutó).
"""

from __future__ import annotations

import math
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

import psycopg2.extras
import structlog

from database.connection import DatabaseManager, WaitPolicy, select_for_update

__all__ = ["SimulationCommand", "ControlState", "ControlPlane"]

log = structlog.get_logger(__name__)


class SimulationCommand(str, Enum):  # = ENUM simulation_command
    SET_NORMAL = "SET_NORMAL"
    TRIGGER_BLACK_SWAN = "TRIGGER_BLACK_SWAN"


@dataclass(frozen=True)
class ControlState:
    current_mode: str
    orchestrator_id: str | None
    heartbeat_at: datetime | None
    heartbeat_age_s: float | None
    market_state: Mapping[str, Any]
    stats: Mapping[str, Any]
    pending_commands: int

    def is_online(self, max_age_s: float = 10.0) -> bool:
        return self.heartbeat_age_s is not None and self.heartbeat_age_s <= max_age_s


def _json_safe(value: Any) -> Any:
    """JSONB no admite NaN/Infinity: se sustituyen por null de forma recursiva."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class ControlPlane:
    def __init__(self, db: DatabaseManager, *, instance_id: str | None = None) -> None:
        self.db = db
        self.instance_id = instance_id or f"{socket.gethostname()}:{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------ productor
    def enqueue(self, command: SimulationCommand | str, *, requested_by: str, note: str | None = None) -> int:
        command = SimulationCommand(command)
        with self.db.transaction() as cur:
            cur.execute(
                """INSERT INTO simulation_commands (command, requested_by, note)
                   VALUES (%s, %s, %s) RETURNING command_id""",
                (command.value, requested_by[:120], note),
            )
            command_id = int(cur.fetchone()["command_id"])
        log.info("control.command_enqueued", command=command.value, command_id=command_id, requested_by=requested_by)
        return command_id

    # ------------------------------------------------------------ consumidor
    def consume(self, *, limit: int = 10) -> list[SimulationCommand]:
        """Reclama y sella los comandos pendientes (exactly-once entre réplicas)."""
        with self.db.transaction(lock_timeout_ms=500) as cur:
            rows = select_for_update(
                cur, "simulation_commands", where={"consumed_at": None},
                columns=("command_id", "command"), order_by=("command_id",), limit=limit,
                wait=WaitPolicy.SKIP_LOCKED,
            )
            if not rows:
                return []
            cur.execute(
                "UPDATE simulation_commands SET consumed_at = now(), consumed_by = %s WHERE command_id = ANY(%s)",
                (self.instance_id, [r["command_id"] for r in rows]),
            )
        commands = [SimulationCommand(r["command"]) for r in rows]
        log.info("control.commands_consumed", commands=[c.value for c in commands], consumer=self.instance_id)
        return commands

    def publish(self, mode: str, market_state: Mapping[str, Any], stats: Mapping[str, Any] | None = None) -> None:
        with self.db.transaction(lock_timeout_ms=500) as cur:
            cur.execute(
                """UPDATE simulation_control
                      SET current_mode = %s, orchestrator_id = %s, heartbeat_at = now(),
                          market_state = %s, stats = %s
                    WHERE control_id = 1""",
                (mode, self.instance_id, psycopg2.extras.Json(_json_safe(market_state)),
                 psycopg2.extras.Json(_json_safe(stats or {}))),
            )

    def mark_offline(self) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE simulation_control SET current_mode = 'OFFLINE' WHERE control_id = 1 AND orchestrator_id = %s",
                (self.instance_id,),
            )

    # --------------------------------------------------------------- lectura
    def state(self) -> ControlState:
        with self.db.transaction(readonly=True) as cur:
            cur.execute(
                """SELECT current_mode, orchestrator_id, heartbeat_at, market_state, stats,
                          EXTRACT(EPOCH FROM (now() - heartbeat_at)) AS age,
                          (SELECT count(*) FROM simulation_commands WHERE consumed_at IS NULL) AS pending
                     FROM simulation_control WHERE control_id = 1"""
            )
            r = cur.fetchone()
        return ControlState(
            current_mode=r["current_mode"], orchestrator_id=r["orchestrator_id"], heartbeat_at=r["heartbeat_at"],
            heartbeat_age_s=None if r["age"] is None else float(r["age"]),
            market_state=r["market_state"] or {}, stats=r["stats"] or {}, pending_commands=int(r["pending"]),
        )
