#!/usr/bin/env bash
# PAI-Loop combined image entrypoint: run uvicorn (backend) + nginx (SPA/proxy)
# in one container and tie their lifecycles together so the orchestrator can
# restart the whole thing if either half dies.
set -euo pipefail

: "${BACKEND_PORT:=8000}"
: "${WEB_CONCURRENCY:=1}"   # sqlite is single-writer; scale via Postgres, not workers.

cd /app/backend

# Backend on the loopback — nginx is the only public listener (:80).
uvicorn app.lean_main:app \
  --host 127.0.0.1 --port "${BACKEND_PORT}" \
  --workers "${WEB_CONCURRENCY}" &
backend_pid=$!

nginx -g 'daemon off;' &
nginx_pid=$!

shutdown() { kill -TERM "${backend_pid}" "${nginx_pid}" 2>/dev/null || true; }
trap shutdown TERM INT

# Exit as soon as either process exits, propagating its status.
wait -n "${backend_pid}" "${nginx_pid}"
status=$?
shutdown
exit "${status}"
