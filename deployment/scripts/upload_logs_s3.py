#!/usr/bin/env python3
"""
deployment/scripts/upload_logs_s3.py
====================================

Archivado en S3 de logs y registros de auditoría del Risk Engine.

1. **Logs rotados** (``LOG_DIR``, p.ej. ``orchestrator.log.1``): cada fichero se *reclama* moviéndolo de
   forma atómica a ``LOG_DIR/.spool/`` (``os.replace`` en el mismo filesystem: sin carreras con
   ``RotatingFileHandler``), se comprime con gzip y se sube. Solo se borra en local tras verificar con
   ``HeadObject`` que el tamaño remoto coincide. Si algo falla, queda en el spool y se reintenta.
2. **Auditoría** (``--export-audit``): exporta incrementalmente ``liquidation_events`` e
   ``insurance_fund_ledger`` (tablas append-only) a NDJSON.gz. La marca de agua vive en S3
   (``_state/audit_watermark.json``): el proceso es *stateless* y re-ejecutable.

Particionado Hive (consultable directamente con Athena / Glue)::

    s3://<bucket>/<prefix>/logs/service=<svc>/dt=YYYY-MM-DD/host=<host>/<fichero>.ndjson.gz
    s3://<bucket>/<prefix>/audit/table=<tabla>/dt=YYYY-MM-DD/<tabla>-<desde>-<hasta>.ndjson.gz

Cifrado en reposo: SSE-KMS si ``LOG_ARCHIVE_KMS_KEY_ID`` está definido; si no, SSE-S3 (AES256).

Uso::

    python upload_logs_s3.py                    # una pasada
    python upload_logs_s3.py --loop --interval 300 --export-audit
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import logging
import os
import pathlib
import shutil
import signal
import socket
import sys
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

log = logging.getLogger("log_shipper")

AUDIT_TABLES = {"liquidation_events": "event_id", "insurance_fund_ledger": "entry_id"}
WATERMARK_KEY = "_state/audit_watermark.json"


@dataclass(frozen=True)
class ShipperConfig:
    bucket: str
    prefix: str = "risk-engine"
    log_dir: pathlib.Path = pathlib.Path("/var/log/risk-engine")
    service: str = "risk-engine"
    host: str = field(default_factory=socket.gethostname)
    kms_key_id: str | None = None
    region: str | None = None
    endpoint_url: str | None = None     # MinIO / LocalStack
    audit_batch_rows: int = 50_000

    @classmethod
    def from_env(cls) -> "ShipperConfig":
        return cls(
            bucket=os.environ["LOG_ARCHIVE_BUCKET"],
            prefix=os.getenv("LOG_ARCHIVE_PREFIX", "risk-engine").strip("/"),
            log_dir=pathlib.Path(os.getenv("LOG_DIR", "/var/log/risk-engine")),
            service=os.getenv("LOG_SERVICE_NAME", "risk-engine"),
            host=os.getenv("LOG_HOST_NAME", socket.gethostname()),
            kms_key_id=os.getenv("LOG_ARCHIVE_KMS_KEY_ID") or None,
            region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
        )

    def key(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in (self.prefix, *parts) if p)


def s3_client(cfg: ShipperConfig) -> Any:
    """Cliente con reintentos adaptativos (throttling/5xx). Credenciales: rol IAM de la instancia EC2."""
    return boto3.client(
        "s3", region_name=cfg.region, endpoint_url=cfg.endpoint_url,
        config=BotoConfig(retries={"max_attempts": 10, "mode": "adaptive"}, connect_timeout=5, read_timeout=60),
    )


def _encryption_args(cfg: ShipperConfig) -> dict[str, str]:
    if cfg.kms_key_id:
        return {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": cfg.kms_key_id}
    return {"ServerSideEncryption": "AES256"}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def put_verified(client: Any, cfg: ShipperConfig, path: pathlib.Path, key: str, *, content_type: str) -> None:
    """Sube el fichero y verifica tamaño remoto + checksum en metadatos. Lanza si no coincide."""
    sha = _sha256(path)
    with path.open("rb") as fh:
        client.put_object(
            Bucket=cfg.bucket, Key=key, Body=fh, ContentType=content_type, ContentEncoding="gzip",
            Metadata={"sha256": sha, "host": cfg.host, "service": cfg.service}, **_encryption_args(cfg),
        )
    head = client.head_object(Bucket=cfg.bucket, Key=key)
    if head["ContentLength"] != path.stat().st_size or head.get("Metadata", {}).get("sha256") != sha:
        raise IOError(f"Verificación fallida para s3://{cfg.bucket}/{key}")


# =============================================================================
# Logs rotados
# =============================================================================
def rotated_logs(log_dir: pathlib.Path) -> list[pathlib.Path]:
    """Ficheros ya rotados (``x.log.1``, ``x.log.2``...). El fichero activo nunca se toca."""
    if not log_dir.is_dir():
        return []
    return sorted(p for p in log_dir.iterdir()
                  if p.is_file() and ".log." in p.name and p.name.rsplit(".", 1)[-1].isdigit())


def claim_to_spool(paths: Iterable[pathlib.Path], spool: pathlib.Path) -> list[pathlib.Path]:
    spool.mkdir(parents=True, exist_ok=True)
    claimed = []
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for p in paths:
        target = spool / f"{p.name.split('.log.')[0]}-{stamp}-{p.name.rsplit('.', 1)[-1]}.ndjson"
        try:
            os.replace(p, target)  # atómico: el handler de rotación ya no lo verá
            claimed.append(target)
        except FileNotFoundError:
            continue  # rotado/movido entre el listado y el claim
    return claimed


def gzip_file(src: pathlib.Path) -> pathlib.Path:
    dst = src.with_suffix(src.suffix + ".gz")
    with src.open("rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    src.unlink()
    return dst


def ship_logs(client: Any, cfg: ShipperConfig) -> list[str]:
    """Reclama, comprime y sube los logs rotados (y los pendientes del spool). Devuelve claves S3."""
    spool = cfg.log_dir / ".spool"
    claim_to_spool(rotated_logs(cfg.log_dir), spool)
    for raw in sorted(spool.glob("*.ndjson")):
        gzip_file(raw)
    uploaded = []
    for gz in sorted(spool.glob("*.ndjson.gz")):
        day = dt.datetime.fromtimestamp(gz.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%d")
        key = cfg.key("logs", f"service={cfg.service}", f"dt={day}", f"host={cfg.host}", gz.name)
        try:
            put_verified(client, cfg, gz, key, content_type="application/x-ndjson")
        except (BotoCoreError, ClientError, IOError) as exc:
            log.warning("upload fallido %s: %s (se reintentará)", gz.name, exc)
            continue
        gz.unlink()
        uploaded.append(key)
        log.info("subido s3://%s/%s", cfg.bucket, key)
    return uploaded


# =============================================================================
# Exportación de auditoría (incremental, marca de agua en S3)
# =============================================================================
def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)  # sin pérdida de precisión (NUMERIC 36,18)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return str(value)


def load_watermark(client: Any, cfg: ShipperConfig) -> dict[str, int]:
    try:
        body = client.get_object(Bucket=cfg.bucket, Key=cfg.key(WATERMARK_KEY))["Body"].read()
        return {k: int(v) for k, v in json.loads(body).items()}
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return {}
        raise


def save_watermark(client: Any, cfg: ShipperConfig, marks: dict[str, int]) -> None:
    client.put_object(Bucket=cfg.bucket, Key=cfg.key(WATERMARK_KEY), Body=json.dumps(marks).encode(),
                      ContentType="application/json", **_encryption_args(cfg))


def export_audit(client: Any, cfg: ShipperConfig, db: Any, *, workdir: pathlib.Path) -> list[str]:
    """Exporta filas nuevas de las tablas append-only. La marca de agua avanza solo tras subir y verificar."""
    marks = load_watermark(client, cfg)
    uploaded: list[str] = []
    workdir.mkdir(parents=True, exist_ok=True)
    for table, pk in AUDIT_TABLES.items():
        last = marks.get(table, 0)
        with db.transaction(readonly=True) as cur:
            cur.execute(f"SELECT * FROM {table} WHERE {pk} > %s ORDER BY {pk} LIMIT %s",  # noqa: S608 (identificadores fijos)
                        (last, cfg.audit_batch_rows))
            rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            continue
        first_id, last_id = rows[0][pk], rows[-1][pk]
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        path = workdir / f"{table}-{first_id:012d}-{last_id:012d}.ndjson.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=_json_default, ensure_ascii=False) + "\n")
        key = cfg.key("audit", f"table={table}", f"dt={day}", path.name)
        put_verified(client, cfg, path, key, content_type="application/x-ndjson")
        path.unlink()
        marks[table] = int(last_id)
        save_watermark(client, cfg, marks)
        uploaded.append(key)
        log.info("auditoría %s: filas %s..%s -> s3://%s/%s", table, first_id, last_id, cfg.bucket, key)
    return uploaded


# =============================================================================
# CLI
# =============================================================================
def run_once(client: Any, cfg: ShipperConfig, *, export: bool, db: Any | None) -> None:
    try:
        ship_logs(client, cfg)
    except Exception:  # noqa: BLE001
        log.exception("fallo en el envío de logs")
    if export and db is not None:
        try:
            export_audit(client, cfg, db, workdir=cfg.log_dir / ".audit")
        except Exception:  # noqa: BLE001
            log.exception("fallo en la exportación de auditoría")


def main() -> None:  # pragma: no cover - punto de entrada
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
    parser = argparse.ArgumentParser(description="Archivado de logs y auditoría en S3")
    parser.add_argument("--loop", action="store_true", help="ejecución continua")
    parser.add_argument("--interval", type=float, default=float(os.getenv("LOG_SHIP_INTERVAL_S", "300")))
    parser.add_argument("--export-audit", action="store_true",
                        default=os.getenv("AUDIT_EXPORT_ENABLED", "false").lower() == "true")
    args = parser.parse_args()

    cfg = ShipperConfig.from_env()
    client = s3_client(cfg)
    db = None
    if args.export_audit:
        sys.path.insert(0, "/app")
        from database.connection import DatabaseManager, DBConfig

        db = DatabaseManager(DBConfig.from_env(load_dotenv=False))

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    log.info("shipper iniciado bucket=%s prefix=%s loop=%s", cfg.bucket, cfg.prefix, args.loop)
    run_once(client, cfg, export=args.export_audit, db=db)
    while args.loop and not stop.wait(args.interval):
        run_once(client, cfg, export=args.export_audit, db=db)
    if args.loop:
        run_once(client, cfg, export=args.export_audit, db=db)  # vaciado final en parada ordenada
    log.info("shipper detenido")


if __name__ == "__main__":  # pragma: no cover
    main()
