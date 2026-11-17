#!/usr/bin/env bash
# =============================================================================
#  Punto de entrada por rol del contenedor (PID 1 = tini, que reenvía SIGTERM).
#
#    orchestrator [args]   agentes + mercado + motor de liquidación (servicio continuo)
#    dashboard             Plotly Dash (gunicorn) en :8501
#    migrate               aplica migraciones SQL pendientes y termina
#    shipper [args]        sube a S3 logs rotados y exportaciones de auditoría (bucle)
#    shell                 depuración
# =============================================================================
set -Eeuo pipefail

ROLE="${1:-${SERVICE_ROLE:-orchestrator}}"
shift || true
export SERVICE_ROLE="${ROLE}"

log() { printf '{"level":"info","event":"entrypoint","role":"%s","msg":"%s"}\n' "${ROLE}" "$*"; }

require_env() {
  local missing=()
  for var in "$@"; do [[ -n "${!var:-}" ]] || missing+=("${var}"); done
  if (( ${#missing[@]} )); then
    printf '{"level":"error","event":"entrypoint","role":"%s","missing_env":"%s"}\n' "${ROLE}" "${missing[*]}" >&2
    exit 64
  fi
}

# Rechaza valores de plantilla (change_me, <SSM:...>) que nunca deben llegar a un contenedor.
reject_placeholders() {
  local bad=()
  for var in "$@"; do
    local value="${!var:-}"
    [[ -z "${value}" ]] && continue
    if [[ "${value}" == "change_me" || "${value}" == "<SSM:"* || "${value}" == "changeme" ]]; then
      bad+=("${var}")
    fi
  done
  if (( ${#bad[@]} )); then
    printf '{"level":"error","event":"entrypoint","role":"%s","placeholder_secrets":"%s"}\n' "${ROLE}" "${bad[*]}" >&2
    exit 64
  fi
}

wait_for_db() {
  local retries="${DB_WAIT_RETRIES:-30}"
  for ((i = 1; i <= retries; i++)); do
    if python - <<'PY' >/dev/null 2>&1
from database.connection import DatabaseManager, DBConfig
db = DatabaseManager(DBConfig.from_env(load_dotenv=False)); db.ping(); db.close()
PY
    then
      log "PostgreSQL disponible"
      return 0
    fi
    sleep 2
  done
  log "PostgreSQL no disponible tras ${retries} intentos"
  exit 69
}

require_env POSTGRES_HOST POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD
reject_placeholders POSTGRES_PASSWORD DASHBOARD_ADMIN_PASSWORD OPENAI_API_KEY HL_SECRET_KEY
cd /app

case "${ROLE}" in
  orchestrator)
    wait_for_db
    export LOG_FILE="${LOG_FILE:-/var/log/risk-engine/orchestrator.log}"
    log "arrancando orquestador ${APP_GIT_SHA:-}"
    exec python -m ai_agents.orchestrator ${ORCHESTRATOR_ARGS:-} "$@"
    ;;
  dashboard)
    # En contenedor los controles (Cisne Negro, recapitalización) exigen contraseña de operador,
    # salvo opt-out explícito para demos locales.
    if [[ "${DASHBOARD_ALLOW_ANONYMOUS:-false}" != "true" ]]; then
      require_env DASHBOARD_ADMIN_PASSWORD
    fi
    wait_for_db
    log "arrancando dashboard"
    # 1 worker: la serie precio simulado vs. real vive en memoria del proceso; hilos para los callbacks
    exec gunicorn dashboard.app:server \
      --bind="0.0.0.0:${DASHBOARD_PORT:-8501}" \
      --workers=1 --threads="${DASHBOARD_THREADS:-8}" \
      --timeout=60 --access-logfile=- "$@"
    ;;
  migrate)
    wait_for_db
    exec python -m database.migrate "$@"
    ;;
  shipper)
    require_env LOG_ARCHIVE_BUCKET
    exec python /app/deployment/scripts/upload_logs_s3.py --loop "$@"
    ;;
  shell)
    exec bash
    ;;
  *)
    printf 'Rol desconocido: %s (orchestrator|dashboard|migrate|shipper|shell)\n' "${ROLE}" >&2
    exit 64
    ;;
esac
