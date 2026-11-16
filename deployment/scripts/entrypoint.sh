#!/usr/bin/env bash
# =============================================================================
#  Punto de entrada por rol del contenedor (PID 1 = tini, que reenvía SIGTERM).
#
#    orchestrator [args]   agentes + mercado + motor de liquidación (servicio continuo)
#    dashboard             Streamlit en :8501
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
cd /app

case "${ROLE}" in
  orchestrator)
    wait_for_db
    export LOG_FILE="${LOG_FILE:-/var/log/risk-engine/orchestrator.log}"
    log "arrancando orquestador ${APP_GIT_SHA:-}"
    exec python -m ai_agents.orchestrator ${ORCHESTRATOR_ARGS:-} "$@"
    ;;
  dashboard)
    wait_for_db
    log "arrancando dashboard"
    exec streamlit run dashboard/app.py \
      --server.port="${DASHBOARD_PORT:-8501}" \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false "$@"
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
