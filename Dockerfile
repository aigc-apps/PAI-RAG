# syntax=docker/dockerfile:1
# =============================================================================
# PAI-Loop — combined image (Frontend + Backend)
#
# One container serves the whole app:
#   * nginx serves the built SPA and reverse-proxies /v1 -> uvicorn (loopback)
#   * uvicorn runs the lean agent service (app.lean_main:app) on 127.0.0.1:8000
#
# Build:   docker build -t pai-loop .
# Run:     docker run -p 8080:80 \
#            -e DASHSCOPE_API_KEY=... \
#            -v $(pwd)/backend/data:/app/backend/data \   # optional: persist db/config/skills
#            pai-loop
# =============================================================================

# --------------------------------------------------------------------------- #
# Stage 1 — build the frontend SPA
# --------------------------------------------------------------------------- #
FROM node:20-alpine AS frontend
WORKDIR /frontend
# Lockfile-driven install first for layer caching.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # tsc -b && vite build  ->  /frontend/dist

# --------------------------------------------------------------------------- #
# Stage 2 — resolve backend dependencies into a venv with uv
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS backend
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/backend/.venv
WORKDIR /app/backend
RUN pip install --no-cache-dir uv

# Set to "false" for the lean base (no pdf/docx/pptx/xlsx upload parsers).
ARG INSTALL_PARSERS=true

# Dependency layer: only the lockfile + manifest, so source edits don't bust it.
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_PARSERS" = "true" ]; then \
      uv sync --frozen --no-dev --extra parsers; \
    else \
      uv sync --frozen --no-dev; \
    fi

# Application source (package = false, so nothing to build — just copy).
COPY backend/ ./

# --------------------------------------------------------------------------- #
# Stage 3 — runtime: nginx + the resolved backend
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx curl \
 && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/nginx/sites-enabled/default

WORKDIR /app/backend

# Backend (source + venv) and the built SPA.
COPY --from=backend /app/backend /app/backend
COPY --from=frontend /frontend/dist /usr/share/nginx/html

# nginx vhost + process supervisor.
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/app/backend/.venv/bin:${PATH}" \
    BACKEND_PORT=8000 \
    WEB_CONCURRENCY=1

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost/ >/dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
