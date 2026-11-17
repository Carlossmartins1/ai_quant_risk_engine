# syntax=docker/dockerfile:1.7
# =============================================================================
#  AI-Driven Quant Risk & Liquidation Engine — imagen multi-target
#
#    docker build -f deployment/Dockerfile --target engine    -t risk-engine:latest .
#    docker build -f deployment/Dockerfile --target dashboard -t risk-dashboard:latest .
#
#  * engine    : orquestador, motor de liquidación, migraciones y log-shipper (LangChain, SDK HL, boto3...)
#  * dashboard : Plotly Dash (gunicorn) + lectura de PostgreSQL (≈ 1/3 del tamaño; sin LangChain ni SDKs de exchange)
#
#  Optimizaciones: builder separado (sin compiladores en runtime), venv copiado entero, caché de pip
#  con BuildKit, capas de dependencias antes que el código, usuario no root (UID 10001), tini como PID 1
#  (señales y zombies), PYTHONDONTWRITEBYTECODE (compatible con read_only rootfs).
# =============================================================================
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- base común
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app

# ------------------------------------------------------------- builders (venv)
FROM base AS builder-engine
RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel && pip install -r /tmp/requirements.txt

FROM base AS builder-dashboard
RUN python -m venv /opt/venv
COPY requirements-dashboard.txt /tmp/requirements-dashboard.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel && pip install -r /tmp/requirements-dashboard.txt

# ------------------------------------------------------------- runtime común
FROM base AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /var/log/risk-engine \
 && chown app:app /var/log/risk-engine
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="ai-quant-risk-engine" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}"
ENV APP_GIT_SHA=${GIT_SHA} \
    LOG_FORMAT=json \
    HOME=/tmp
COPY --chown=app:app deployment/scripts/ /app/deployment/scripts/
RUN chmod 0555 /app/deployment/scripts/*.sh
ENTRYPOINT ["/usr/bin/tini", "--", "/app/deployment/scripts/entrypoint.sh"]

# ---------------------------------------------------------------------- engine
FROM runtime AS engine
COPY --from=builder-engine /opt/venv /opt/venv
COPY --chown=app:app common/          /app/common/
COPY --chown=app:app database/        /app/database/
COPY --chown=app:app exchange_connector/ /app/exchange_connector/
COPY --chown=app:app quant_models/    /app/quant_models/
COPY --chown=app:app risk_engine/     /app/risk_engine/
COPY --chown=app:app ai_agents/       /app/ai_agents/
USER app
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "/app/deployment/scripts/healthcheck.py"]
CMD ["orchestrator"]

# ------------------------------------------------------------------- dashboard
FROM runtime AS dashboard
COPY --from=builder-dashboard /opt/venv /opt/venv
COPY --chown=app:app common/          /app/common/
COPY --chown=app:app database/        /app/database/
COPY --chown=app:app risk_engine/__init__.py risk_engine/margin_calculator.py /app/risk_engine/
COPY --chown=app:app ai_agents/__init__.py ai_agents/control_plane.py         /app/ai_agents/
COPY --chown=app:app dashboard/       /app/dashboard/
USER app
EXPOSE 8501
ENV SERVICE_ROLE=dashboard
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["curl", "-fsS", "http://127.0.0.1:8501/healthz"]
CMD ["dashboard"]
