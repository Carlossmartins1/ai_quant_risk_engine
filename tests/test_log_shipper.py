"""
Tests del archivado en S3 (``deployment/scripts/upload_logs_s3.py``) contra S3 simulado (moto):
claim atómico de logs rotados, compresión, particionado Hive, cifrado, verificación, reintentos y
exportación incremental de auditoría con marca de agua.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import pathlib
import sys

import boto3
import pytest

moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("upload_logs_s3", ROOT / "deployment" / "scripts" / "upload_logs_s3.py")
shipper = importlib.util.module_from_spec(_spec)
sys.modules["upload_logs_s3"] = shipper  # requerido por @dataclass al cargar por ruta
_spec.loader.exec_module(shipper)

BUCKET = "risk-archive-test"


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-1")
        client.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        yield client


def cfg(tmp_path, **kw):
    return shipper.ShipperConfig(bucket=BUCKET, prefix="risk/prod", log_dir=tmp_path, service="orchestrator",
                                 host="ip-10-0-0-1", region="eu-west-1", **kw)


def test_ships_only_rotated_logs_with_verification(s3, tmp_path):
    (tmp_path / "orchestrator.log").write_text('{"event":"activo"}\n')
    payloads = {}
    for i in (1, 2):
        p = tmp_path / f"orchestrator.log.{i}"
        p.write_text("".join(json.dumps({"event": "x", "i": i, "n": n}) + "\n" for n in range(500)))
        payloads[i] = p.read_text()

    keys = shipper.ship_logs(s3, cfg(tmp_path))
    assert len(keys) == 2
    assert all(k.startswith("risk/prod/logs/service=orchestrator/dt=") and "/host=ip-10-0-0-1/" in k for k in keys)
    assert all(k.endswith(".ndjson.gz") for k in keys)
    contents = sorted(gzip.decompress(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()).decode() for k in keys)
    assert contents == sorted(payloads.values())                   # sin pérdida de datos
    head = s3.head_object(Bucket=BUCKET, Key=keys[0])
    assert head["ServerSideEncryption"] == "AES256" and len(head["Metadata"]["sha256"]) == 64
    assert head["ContentEncoding"] == "gzip"
    assert (tmp_path / "orchestrator.log").exists()                # el fichero activo nunca se toca
    assert not list((tmp_path / ".spool").iterdir())               # spool vacío tras verificar
    assert shipper.ship_logs(s3, cfg(tmp_path)) == []              # idempotente


def test_failed_upload_stays_in_spool_and_retries(s3, tmp_path, monkeypatch):
    (tmp_path / "orchestrator.log.1").write_text('{"event":"y"}\n')
    real_put = s3.put_object
    monkeypatch.setattr(s3, "put_object", lambda **kw: (_ for _ in ()).throw(
        shipper.ClientError({"Error": {"Code": "SlowDown"}}, "PutObject")))
    assert shipper.ship_logs(s3, cfg(tmp_path)) == []
    assert len(list((tmp_path / ".spool").glob("*.gz"))) == 1     # conservado para reintento
    monkeypatch.setattr(s3, "put_object", real_put)
    assert len(shipper.ship_logs(s3, cfg(tmp_path))) == 1


def test_kms_encryption_arguments(tmp_path):
    c = cfg(tmp_path, kms_key_id="arn:aws:kms:eu-west-1:123:alias/logs")
    assert shipper._encryption_args(c) == {"ServerSideEncryption": "aws:kms",
                                           "SSEKMSKeyId": "arn:aws:kms:eu-west-1:123:alias/logs"}


@pytest.mark.integration
def test_incremental_audit_export_with_watermark(s3, tmp_path, clean_db):
    from risk_engine.liquidation_executor import LiquidationExecutor
    from tests.helpers import make_store, seed_account, seed_insurance_fund

    seed_insurance_fund(clean_db, 50_000)
    store = make_store({"BTC": {"mark": 89_000, "bids": [(88_900, 5.0)]}})
    a = seed_account(clean_db, "w1", "WHALE_AGGRESSIVE", 4_750.0, [("BTC", 1.0, 95_000.0, 20, 0.025)])
    LiquidationExecutor(clean_db, store).liquidate_account(a)

    c = cfg(tmp_path)
    keys = shipper.export_audit(s3, c, clean_db, workdir=tmp_path / ".audit")
    assert {k.split("/")[3] for k in keys} == {"table=liquidation_events", "table=insurance_fund_ledger"}
    ev_key = next(k for k in keys if "liquidation_events" in k)
    rows = [json.loads(l) for l in gzip.decompress(s3.get_object(Bucket=BUCKET, Key=ev_key)["Body"].read()).splitlines()]
    assert rows[0]["deficit"] == "1350.000000000000000000"         # Decimal exportado sin pérdida
    marks = json.loads(s3.get_object(Bucket=BUCKET, Key="risk/prod/_state/audit_watermark.json")["Body"].read())
    assert marks == {"liquidation_events": 1, "insurance_fund_ledger": 2}

    assert shipper.export_audit(s3, c, clean_db, workdir=tmp_path / ".audit") == []   # nada nuevo
    b = seed_account(clean_db, "w2", "WHALE_AGGRESSIVE", 4_750.0, [("BTC", 1.0, 95_000.0, 20, 0.025)])
    LiquidationExecutor(clean_db, store).liquidate_account(b)
    new = shipper.export_audit(s3, c, clean_db, workdir=tmp_path / ".audit")
    ev2 = next(k for k in new if "liquidation_events" in k)
    assert ev2.endswith("liquidation_events-000000000002-000000000002.ndjson.gz")
