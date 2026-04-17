#!/usr/bin/env bash
# Minimal launcher for /v1/files local demo.
#   - starts redis if not running
#   - applies alembic migrations
#   - starts a Celery worker in the background
#   - starts the API in the foreground
# Ctrl-C tears everything down.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8682}"
API_HOST="${API_HOST:-0.0.0.0}"

echo "→ Project root: $ROOT"

# ---- 1. Redis ----
if ! pgrep -x redis-server >/dev/null 2>&1; then
  echo "→ Starting local redis-server (daemonized)"
  redis-server --daemonize yes
else
  echo "→ redis-server already running"
fi

# ---- 2. Migrations ----
echo "→ alembic upgrade head"
poetry run alembic upgrade head >/dev/null

# ---- 3. Worker ----
WORKER_LOG="$(mktemp -t pairag-worker.XXXXXX.log)"
echo "→ Starting Celery worker (logs: $WORKER_LOG)"
cd "$ROOT/backend"
poetry run celery -A app.worker worker --loglevel=info \
  > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!
cd "$ROOT"

cleanup() {
  echo
  echo "→ Cleaning up (worker PID $WORKER_PID)"
  kill "$WORKER_PID" 2>/dev/null || true
  wait "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Give the worker a moment to register tasks
sleep 2

# ---- 4. API ----
echo "→ API on http://$API_HOST:$PORT   (Ctrl-C to stop)"
echo "→ OpenAPI docs: http://$API_HOST:$PORT/docs"
cd "$ROOT/backend"
exec poetry run uvicorn app.main:app --host "$API_HOST" --port "$PORT" --reload
