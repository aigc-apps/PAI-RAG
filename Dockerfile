FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /app/frontends/react

ENV NEXT_TELEMETRY_DISABLED=1

COPY frontends/react/package*.json ./
RUN npm ci

COPY frontends/react ./
RUN BACKEND_BASE_URL=http://127.0.0.1:8682 npm run build \
    && npm prune --omit=dev \
    && npm cache clean --force


FROM node:22-bookworm-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    NEXT_TELEMETRY_DISABLED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PORT=8680 \
    FRONTEND_PORT=8681 \
    BACKEND_PORT=8682 \
    HOST=0.0.0.0 \
    RUNNER_BACKEND=celery \
    START_REDIS=auto \
    REDIS_URL=redis://127.0.0.1:6379/0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gettext-base \
        nginx \
        procps \
        python3 \
        python3-pip \
        python3-venv \
        redis-server \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

RUN curl -fsSL -o /tmp/aliyun-cli.tgz \
        https://easyrec.oss-cn-beijing.aliyuncs.com/aliyun-cli-spec/aliyun-cli-linux-latest-amd64.tgz \
    && tar -xzf /tmp/aliyun-cli.tgz -C /usr/local/bin aliyun \
    && chmod +x /usr/local/bin/aliyun \
    && rm /tmp/aliyun-cli.tgz \
    && aliyun version

RUN aliyun plugin install --names \
        aliyun-cli-eas \
        aliyun-cli-pairecservice \
        aliyun-cli-pai-dsw \
    && aliyun plugin list

RUN mkdir -p /app/skills/alibabacloud-pai-rec-diagnosis \
    && curl -fsSL -o /tmp/pai-rec-diagnosis.zip \
        "https://skills.aliyun.com/api/public/skills/alibabacloud-pai-rec-diagnosis/download?spm=aliyun-agent-skills-portal.skill_detail.0.0.490556efPbwzVC" \
    && python3 -m zipfile -e /tmp/pai-rec-diagnosis.zip /app/skills/alibabacloud-pai-rec-diagnosis/ \
    && rm /tmp/pai-rec-diagnosis.zip \
    && ls /app/skills/alibabacloud-pai-rec-diagnosis/

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /app/frontends/react/.next ./frontends/react/.next
COPY --from=frontend-builder /app/frontends/react/node_modules ./frontends/react/node_modules
COPY docker/entrypoint.sh /usr/local/bin/pai-rag-entrypoint

RUN chmod +x /app/scripts/start.sh /usr/local/bin/pai-rag-entrypoint \
    && mkdir -p /app/workspaces /app/memory /app/skills /app/.tmp /app/services/agent-arena/logs /app/services/agent-arena/data \
    && rm -f /etc/nginx/sites-enabled/default

EXPOSE 8680

VOLUME ["/app/workspaces", "/app/memory", "/app/skills"]

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/pai-rag-entrypoint"]
CMD ["./scripts/start.sh"]
