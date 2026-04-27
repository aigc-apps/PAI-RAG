#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontends/react"
NGINX_TEMPLATE="$ROOT_DIR/scripts/nginx.template.conf"
NGINX_CONFIG="${NGINX_CONFIG:-/etc/nginx/conf.d/pai-rag.conf}"

PORT="${PORT:-8680}"
FRONTEND_PORT="${FRONTEND_PORT:-8681}"
BACKEND_PORT="${BACKEND_PORT:-8682}"
API_INSTANCE_COUNT="${API_INSTANCE_COUNT:-1}"
WORKER_INSTANCE_COUNT="${WORKER_INSTANCE_COUNT:-2}"
DEV_MODE="${DEV_MODE:-false}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
START_REDIS="${START_REDIS:-auto}"
RUNNER_BACKEND="${RUNNER_BACKEND:-}"
HOST="${HOST:-0.0.0.0}"

API_PID=""
FRONTEND_PID=""
WORKER_PID=""
REDIS_PID=""

usage() {
  cat <<'EOF'
Usage: ./start.sh [options]

Options:
  --port PORT                App entry port. In dev mode this is the frontend port.
                             In production mode this is the nginx port. Default: 8680
  --frontend-port PORT       Frontend service port in production mode. Default: 8681
  --backend-port PORT        Backend service port. Default: 8682
  --api-instances N          Number of Uvicorn workers. Default: 1
  --worker-instances N, -w   Celery worker concurrency. 0 disables local worker. Default: 2
  --dev                      Start without nginx and expose Next.js directly.
  --help, -h                 Show this help.

Environment:
  RUNNER_BACKEND             thread or celery. Auto: celery when worker/api scaling needs it.
  REDIS_URL                  Redis URL for Celery mode. Default: redis://127.0.0.1:6379/0
  START_REDIS                auto, true, or false. Default: auto
  NGINX_CONFIG               Nginx output config path. Default: /etc/nginx/conf.d/pai-rag.conf
  SERVER_API_KEY             Optional service token used by the Next.js server proxy.

Examples:
  ./start.sh --dev --port 3001 --backend-port 8000 --worker-instances 0
  ./start.sh --port 8680 --frontend-port 8681 --backend-port 8682 --api-instances 4 --worker-instances 4
EOF
}

validate_port() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 || "$value" -gt 65535 ]]; then
    echo "Error: $name must be a valid port between 1 and 65535" >&2
    exit 1
  fi
}

validate_count() {
  local name="$1"
  local value="$2"
  local min="$3"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt "$min" ]]; then
    echo "Error: $name must be an integer >= $min" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      validate_port "--port" "$PORT"
      shift 2
      ;;
    --frontend-port)
      FRONTEND_PORT="${2:-}"
      validate_port "--frontend-port" "$FRONTEND_PORT"
      shift 2
      ;;
    --backend-port)
      BACKEND_PORT="${2:-}"
      validate_port "--backend-port" "$BACKEND_PORT"
      shift 2
      ;;
    --api-instances)
      API_INSTANCE_COUNT="${2:-}"
      validate_count "--api-instances" "$API_INSTANCE_COUNT" 1
      shift 2
      ;;
    --worker-instances|-w)
      WORKER_INSTANCE_COUNT="${2:-}"
      validate_count "--worker-instances" "$WORKER_INSTANCE_COUNT" 0
      shift 2
      ;;
    --dev)
      DEV_MODE=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

validate_port "PORT" "$PORT"
validate_port "FRONTEND_PORT" "$FRONTEND_PORT"
validate_port "BACKEND_PORT" "$BACKEND_PORT"
validate_count "API_INSTANCE_COUNT" "$API_INSTANCE_COUNT" 1
validate_count "WORKER_INSTANCE_COUNT" "$WORKER_INSTANCE_COUNT" 0

if [[ -z "$RUNNER_BACKEND" ]]; then
  if [[ "$WORKER_INSTANCE_COUNT" -gt 0 || "$API_INSTANCE_COUNT" -gt 1 ]]; then
    RUNNER_BACKEND="celery"
  else
    RUNNER_BACKEND="thread"
  fi
fi

if [[ "$RUNNER_BACKEND" != "thread" && "$RUNNER_BACKEND" != "celery" ]]; then
  echo "Error: RUNNER_BACKEND must be thread or celery" >&2
  exit 1
fi

if [[ "$RUNNER_BACKEND" == "thread" && "$API_INSTANCE_COUNT" -gt 1 ]]; then
  echo "Error: thread runner cannot safely run with multiple API instances. Use RUNNER_BACKEND=celery." >&2
  exit 1
fi

require_command() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: missing command '$cmd'. $hint" >&2
    exit 1
  fi
}

is_process_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

stop_process() {
  local name="$1"
  local pid="$2"
  if is_process_alive "$pid"; then
    echo "Stopping $name (pid $pid)..."
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT TERM INT
  echo "Cleaning up..."
  stop_process "celery worker" "$WORKER_PID"
  stop_process "frontend" "$FRONTEND_PID"
  stop_process "api" "$API_PID"
  stop_process "redis" "$REDIS_PID"
  exit "$exit_code"
}

trap cleanup EXIT TERM INT

render_nginx_config() {
  local target="$1"
  if command -v envsubst >/dev/null 2>&1; then
    PORT="$PORT" FRONTEND_PORT="$FRONTEND_PORT" BACKEND_PORT="$BACKEND_PORT" \
      envsubst '${PORT} ${FRONTEND_PORT} ${BACKEND_PORT}' < "$NGINX_TEMPLATE" > "$target"
  else
    sed \
      -e "s|\${PORT}|$PORT|g" \
      -e "s|\${FRONTEND_PORT}|$FRONTEND_PORT|g" \
      -e "s|\${BACKEND_PORT}|$BACKEND_PORT|g" \
      "$NGINX_TEMPLATE" > "$target"
  fi
}

setup_nginx() {
  require_command nginx "Install nginx or start with --dev."

  if [[ ! -f "$NGINX_TEMPLATE" ]]; then
    echo "Error: nginx template not found: $NGINX_TEMPLATE" >&2
    exit 1
  fi

  local config_dir
  config_dir="$(dirname "$NGINX_CONFIG")"
  if [[ ! -d "$config_dir" || ! -w "$config_dir" ]]; then
    echo "Error: cannot write nginx config to $NGINX_CONFIG. Run as root or set NGINX_CONFIG." >&2
    exit 1
  fi

  echo "Configuring nginx: $NGINX_CONFIG"
  render_nginx_config "$NGINX_CONFIG"
  nginx -t

  if command -v service >/dev/null 2>&1; then
    service nginx start >/dev/null 2>&1 || true
    service nginx reload
  else
    nginx -s reload
  fi
}

ensure_redis() {
  if [[ "$RUNNER_BACKEND" != "celery" ]]; then
    return
  fi

  if command -v redis-cli >/dev/null 2>&1 && redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
    echo "Redis is already reachable: $REDIS_URL"
    return
  fi

  if [[ "$START_REDIS" == "false" ]]; then
    echo "Error: Redis is not reachable and START_REDIS=false." >&2
    exit 1
  fi

  if [[ "$REDIS_URL" != "redis://127.0.0.1:6379/0" && "$REDIS_URL" != "redis://localhost:6379/0" ]]; then
    echo "Error: Redis is not reachable at $REDIS_URL. Start it manually or use the default local URL." >&2
    exit 1
  fi

  require_command redis-server "Install Redis or set START_REDIS=false and start Redis yourself."
  echo "Starting local redis-server..."
  redis-server --save "" --appendonly no &
  REDIS_PID=$!
  sleep 1

  if ! is_process_alive "$REDIS_PID"; then
    echo "Error: redis-server exited immediately." >&2
    exit 1
  fi

  if command -v redis-cli >/dev/null 2>&1 && ! redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
    echo "Error: redis-server started but is not reachable at $REDIS_URL" >&2
    exit 1
  fi
}

ensure_frontend_deps() {
  if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "Installing frontend dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
  fi
}

start_frontend() {
  ensure_frontend_deps

  local frontend_port="$FRONTEND_PORT"
  local command_name="start"
  if [[ "$DEV_MODE" == "true" ]]; then
    frontend_port="$PORT"
    command_name="dev"
  elif [[ ! -d "$FRONTEND_DIR/.next" ]]; then
    echo "Building frontend..."
    (cd "$FRONTEND_DIR" && BACKEND_BASE_URL="http://127.0.0.1:$BACKEND_PORT" npm run build)
  fi

  echo "Starting frontend on port $frontend_port..."
  (
    cd "$FRONTEND_DIR"
    BACKEND_BASE_URL="http://127.0.0.1:$BACKEND_PORT" \
      npm run "$command_name" -- --hostname "$HOST" --port "$frontend_port"
  ) &
  FRONTEND_PID=$!
}

start_api() {
  require_command uvicorn "Install Python dependencies with: pip install -r requirements.txt"

  echo "Starting FastAPI on port $BACKEND_PORT with $API_INSTANCE_COUNT worker(s)..."
  (
    cd "$ROOT_DIR"
    RUNNER_BACKEND="$RUNNER_BACKEND" REDIS_URL="$REDIS_URL" \
      uvicorn backend.server:app --host "$HOST" --port "$BACKEND_PORT" --workers "$API_INSTANCE_COUNT"
  ) &
  API_PID=$!
}

start_worker() {
  if [[ "$RUNNER_BACKEND" != "celery" || "$WORKER_INSTANCE_COUNT" -eq 0 ]]; then
    return
  fi

  require_command celery "Install Python dependencies with: pip install -r requirements.txt"
  echo "Starting Celery worker with concurrency $WORKER_INSTANCE_COUNT..."
  (
    cd "$ROOT_DIR"
    RUNNER_BACKEND="$RUNNER_BACKEND" REDIS_URL="$REDIS_URL" \
      celery -A backend.celery_app worker --loglevel=info --concurrency="$WORKER_INSTANCE_COUNT"
  ) &
  WORKER_PID=$!
}

if [[ ! -f "$ROOT_DIR/config.py" ]]; then
  echo "Error: config.py not found. Run: cp config_template.py config.py" >&2
  exit 1
fi

require_command npm "Install Node.js and npm."

echo "Starting PAI-RAG services"
echo "  mode:              $([[ "$DEV_MODE" == "true" ]] && echo dev || echo production)"
echo "  app port:          $PORT"
echo "  frontend port:     $([[ "$DEV_MODE" == "true" ]] && echo "$PORT" || echo "$FRONTEND_PORT")"
echo "  backend port:      $BACKEND_PORT"
echo "  api instances:     $API_INSTANCE_COUNT"
echo "  runner backend:    $RUNNER_BACKEND"
echo "  worker instances:  $WORKER_INSTANCE_COUNT"
echo "  redis url:         $REDIS_URL"

if [[ "$DEV_MODE" != "true" ]]; then
  setup_nginx
fi

ensure_redis
start_api
start_frontend
start_worker

echo "Services started."
if [[ "$DEV_MODE" == "true" ]]; then
  echo "Open: http://127.0.0.1:$PORT"
else
  echo "Open: http://127.0.0.1:$PORT"
fi

wait "$API_PID"
