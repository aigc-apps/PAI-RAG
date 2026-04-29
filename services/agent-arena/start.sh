#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8787}"
INSTALL_FRONTEND_DEPS="${INSTALL_FRONTEND_DEPS:-auto}"
BUILD_FRONTEND="${BUILD_FRONTEND:-true}"

usage() {
  cat <<'EOF'
Usage: ./start.sh [options]

Options:
  --host HOST          Bind host. Default: 127.0.0.1
  --port PORT          AgentArena port. Default: 8787
  --skip-build         Do not build the frontend before starting backend.
  --no-install         Do not run npm install automatically.
  --help, -h           Show this help.

Environment:
  ARENA_API_KEY        Optional service token for AgentArena /api/*.
  HISTORY_DB_PATH      SQLite history path. Default: data/arena_history.sqlite3
  AGENT_A_*            Agent A endpoint/model/key settings.
  AGENT_B_*            Agent B endpoint/model/key settings.
  JUDGE_*              Optional OpenAI-compatible Judge settings.

Examples:
  ./start.sh --port 8787
  ./start.sh --host 0.0.0.0 --port 8787
EOF
}

validate_port() {
  local value="$1"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 || "$value" -gt 65535 ]]; then
    echo "Error: --port must be a valid port between 1 and 65535" >&2
    exit 1
  fi
}

require_command() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: missing command '$cmd'. $hint" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      validate_port "$PORT"
      shift 2
      ;;
    --skip-build)
      BUILD_FRONTEND=false
      shift
      ;;
    --no-install)
      INSTALL_FRONTEND_DEPS=false
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

validate_port "$PORT"
require_command uvicorn "Install Python dependencies with: pip install -r backend/requirements.txt"
require_command npm "Install Node.js and npm."

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Warning: .env not found. Run: cp .env.example .env"
fi

if [[ "$INSTALL_FRONTEND_DEPS" != "false" && ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing AgentArena frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
fi

if [[ "$BUILD_FRONTEND" == "true" ]]; then
  echo "Building AgentArena frontend..."
  (
    cd "$FRONTEND_DIR"
    ARENA_PUBLIC_BASE="/" ARENA_BACKEND_PORT="$PORT" npm run build
  )
fi

echo "Starting AgentArena on http://$HOST:$PORT"
cd "$ROOT_DIR"
exec uvicorn backend.server:app --host "$HOST" --port "$PORT"
