#!/usr/bin/env bash
# =============================================================================
#  EC2 user-data (Amazon Linux 2023): instala Docker, materializa .env.production desde
#  SSM Parameter Store y levanta el stack. Requiere el instance profile de aws/iam_policy.json.
#
#  Variables a sustituir al lanzar la instancia (o vía plantilla de Launch Template):
#    REPO_URL, GIT_REF, SSM_PREFIX, AWS_REGION, REGISTRY
#
#  Acceso al dashboard sin abrir puertos públicos (recomendado):
#    aws ssm start-session --target <instance-id> \
#      --document-name AWS-StartPortForwardingSession --parameters portNumber=8501,localPortNumber=8501
# =============================================================================
set -Eeuo pipefail
exec > >(tee -a /var/log/risk-engine-bootstrap.log) 2>&1

REPO_URL="${REPO_URL:-https://github.com/<usuario>/ai-quant-risk-engine.git}"
GIT_REF="${GIT_REF:-main}"
SSM_PREFIX="${SSM_PREFIX:-/risk-engine/prod}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
REGISTRY="${REGISTRY:-}"
APP_DIR=/opt/risk-engine

echo ">> Paquetes base"
dnf -y update
dnf -y install docker git jq
systemctl enable --now docker
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

echo ">> Código"
if [[ -d "${APP_DIR}/.git" ]]; then
  git -C "${APP_DIR}" fetch --depth 1 origin "${GIT_REF}" && git -C "${APP_DIR}" checkout -f FETCH_HEAD
else
  git clone --depth 1 --branch "${GIT_REF}" "${REPO_URL}" "${APP_DIR}"
fi
cd "${APP_DIR}/deployment"

echo ">> Secretos desde SSM (${SSM_PREFIX}) -> .env.production (0600, root)"
umask 077
cp .env.production.example .env.production
while IFS=$'\t' read -r name value; do
  key="$(basename "${name}" | tr '[:lower:]' '[:upper:]')"
  # Sustituye la línea KEY=... existente o la añade
  if grep -q "^${key}=" .env.production; then
    python3 - "$key" "$value" <<'PY'
import pathlib, sys
key, value = sys.argv[1], sys.argv[2]
p = pathlib.Path(".env.production")
lines = [f"{key}={value}" if l.startswith(f"{key}=") else l for l in p.read_text().splitlines()]
p.write_text("\n".join(lines) + "\n")
PY
  else
    printf '%s=%s\n' "${key}" "${value}" >> .env.production
  fi
done < <(aws ssm get-parameters-by-path --region "${AWS_REGION}" --path "${SSM_PREFIX}" \
           --with-decryption --recursive --query 'Parameters[].[Name,Value]' --output text)
if grep -q '<SSM:' .env.production; then
  echo "ERROR: quedan secretos sin resolver en .env.production" >&2
  grep -o '^[A-Z_]*=<SSM:[^>]*>' .env.production >&2
  exit 1
fi
{
  echo "GIT_SHA=$(git -C "${APP_DIR}" rev-parse --short HEAD)"
  echo "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ -n "${REGISTRY}" ]] && echo "REGISTRY=${REGISTRY}"
} >> .env.production

echo ">> Imágenes"
if [[ -n "${REGISTRY}" ]]; then
  aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"
  docker compose -f docker-compose.prod.yml --env-file .env.production pull migrate dashboard
else
  docker compose -f docker-compose.prod.yml --env-file .env.production build
fi

echo ">> Arranque"
docker compose -f docker-compose.prod.yml --env-file .env.production up -d
docker compose -f docker-compose.prod.yml --env-file .env.production ps

echo ">> Servicio systemd (re-arranque tras reboot)"
cat > /etc/systemd/system/risk-engine.service <<UNIT
[Unit]
Description=AI Quant Risk Engine stack
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${APP_DIR}/deployment
ExecStart=/usr/bin/docker compose -f docker-compose.prod.yml --env-file .env.production up -d
ExecStop=/usr/bin/docker compose -f docker-compose.prod.yml --env-file .env.production down
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable risk-engine.service
echo ">> Bootstrap completado"
