# Despliegue — AI-Driven Quant Risk & Liquidation Engine

## Arquitectura en producción (una instancia EC2)

```
                      ┌──────────────── red Docker "backend" ────────────────┐
 SSM port-forward ──► │ dashboard :8501 (127.0.0.1)   ──┐                    │
                      │ orchestrator (agentes+riesgo) ──┼──► postgres (vol)  │
                      │ migrate (one-shot)            ──┘        ▲           │
                      │ log-shipper ── logs vol ── S3 (SSE-KMS)  │ auditoría │
                      └──────────────────────────────────────────┴───────────┘
```

| Servicio | Imagen (target) | Función |
|---|---|---|
| `postgres` | `postgres:15-alpine` | Ledger ACID (Fase 1) |
| `migrate` | `engine` | `python -m database.migrate`: esquema + migraciones con checksum y advisory lock |
| `orchestrator` | `engine` | Agentes, mercado sombra, `LiquidationExecutor`, plano de control |
| `dashboard` | `dashboard` | Streamlit (sin LangChain/SDKs: imagen ligera) |
| `log-shipper` | `engine` | Logs rotados + export incremental de auditoría a S3 |

## Pasos

1. **Secretos en SSM** (SecureString), p. ej.:
   ```bash
   aws ssm put-parameter --type SecureString --name /risk-engine/prod/postgres_password --value '...'
   aws ssm put-parameter --type SecureString --name /risk-engine/prod/dashboard_admin_password --value '...'
   aws ssm put-parameter --type SecureString --name /risk-engine/prod/openai_api_key --value '...'
   ```
2. **Bucket S3** con cifrado KMS, bloqueo de acceso público y ciclo de vida:
   ```bash
   aws s3api put-bucket-lifecycle-configuration --bucket my-risk-engine-archive \
     --lifecycle-configuration file://aws/s3_lifecycle.json
   ```
3. **Rol IAM** de la instancia con `aws/iam_policy.json` (mínimo privilegio: solo el prefijo del bucket,
   solo los parámetros `/risk-engine/prod*`, pull de ECR).
4. **Lanzar EC2** (Amazon Linux 2023, ≥ t3.large) con `aws/user_data.sh` como user-data. Sin puertos de
   entrada abiertos: el acceso es por SSM.
5. **Dashboard**:
   ```bash
   aws ssm start-session --target <instance-id> --document-name AWS-StartPortForwardingSession \
     --parameters portNumber=8501,localPortNumber=8501
   # http://localhost:8501
   ```

## Build local de imágenes

```bash
docker build -f deployment/Dockerfile --target engine    -t risk-engine:dev .
docker build -f deployment/Dockerfile --target dashboard -t risk-dashboard:dev .
cd deployment && cp .env.production.example .env.production   # rellenar valores
docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build
```

## Operación

* **Cisne Negro**: botón del dashboard → `simulation_commands` → el orquestador lo consume
  (`FOR UPDATE SKIP LOCKED`) en ≤ 1 s. Si el orquestador está caído, el comando queda en cola.
* **Health**: `docker compose ps` muestra el estado de los healthchecks (heartbeat del orquestador < 30 s;
  `/_stcore/health` del dashboard).
* **Logs**: JSON en stdout (rotado por Docker) + fichero rotado que el shipper archiva en
  `s3://…/logs/service=…/dt=…/host=…/` (consultable con Athena).
* **Auditoría**: `liquidation_events` e `insurance_fund_ledger` se exportan de forma incremental a
  `s3://…/audit/table=…/dt=…/` con marca de agua en `_state/audit_watermark.json` (retención 7 años).
* **Actualizar**: `git pull && docker compose … up -d --build` (las migraciones se aplican solas).
