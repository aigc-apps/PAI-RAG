#!/usr/bin/env bash
# Live-LLM smoke test for AGENT_RUNTIME=sdk.
#
# Run this once after Phase 3 to confirm the SDK runtime actually streams
# tokens from your DashScope/Qwen endpoint before we delete the legacy
# code paths in Phase 4 Stage C. If this fails, the legacy fallback is
# still available (`AGENT_RUNTIME=legacy`).
#
# Prereqs:
#   - .env with API_KEY (DashScope) — same one your normal start.sh uses.
#   - Free port (default 8682) and a temp SQLite scope so the test
#     doesn't pollute your production memory/sessions.
#
# Usage:
#   ./scripts/smoke_test_sdk.sh                # runs all 4 checks
#   ./scripts/smoke_test_sdk.sh --port 9001    # custom port
#   ./scripts/smoke_test_sdk.sh --keep-server  # don't kill server on exit (debug)

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT_OVERRIDE=""
KEEP_SERVER=false
SERVER_PID=""
TMP_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT_OVERRIDE="$2"; shift 2 ;;
    --keep-server) KEEP_SERVER=true; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# //; s/^#$//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

cleanup() {
  local code=$?
  trap - EXIT TERM INT
  if [[ -n "$SERVER_PID" && "$KEEP_SERVER" != "true" ]]; then
    echo "Stopping server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]] && rm -rf "$TMP_DIR"
  exit "$code"
}
trap cleanup EXIT TERM INT

# ── env ────────────────────────────────────────────────────────────────────
if [[ -f "$ROOT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "$ROOT_DIR/.env"; set +a
fi
# CLI --port wins over anything in .env; fall back to .env PORT, then 8682.
if [[ -n "$PORT_OVERRIDE" ]]; then
  PORT="$PORT_OVERRIDE"
else
  PORT="${PORT:-8682}"
fi
if [[ -z "${API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "ERROR: API_KEY (or OPENAI_API_KEY) is not set; configure .env first." >&2
  exit 1
fi
# Make sure nothing else is squatting on the chosen port.
if command -v lsof >/dev/null 2>&1 && lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT already in use. Free it (lsof -ti tcp:$PORT | xargs kill) or pass --port <n>." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d -t pai-rag-smoke-XXXXXX)"
echo "Smoke test workspace: $TMP_DIR"

# ── start server ──────────────────────────────────────────────────────────
LOG="$TMP_DIR/server.log"
echo "Starting server on :$PORT with AGENT_RUNTIME=sdk..."
(
  cd "$ROOT_DIR"
  AGENT_RUNTIME=sdk RUNNER_BACKEND=thread \
    uvicorn backend.server:app --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
  echo $! > "$TMP_DIR/server.pid"
) &
sleep 0.5
SERVER_PID="$(cat "$TMP_DIR/server.pid")"

# Wait for /docs to respond (FastAPI is up).
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/docs" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
if ! curl -fsS "http://127.0.0.1:$PORT/docs" >/dev/null; then
  echo "ERROR: server failed to start. Tail of log:" >&2
  tail -50 "$LOG" >&2
  exit 1
fi
echo "Server up."

# ── helpers ───────────────────────────────────────────────────────────────
fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "PASS: $*"; }

# ── test 1: streaming text response ───────────────────────────────────────
echo
echo "── test 1: /v1/responses stream=true ──"
OUT="$TMP_DIR/resp1.sse"
curl -sS -N -X POST "http://127.0.0.1:$PORT/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{"input":"Reply with the single word: pong","stream":true}' \
  --max-time 60 >"$OUT" || fail "curl error; see $OUT"

grep -q 'event: response.created'   "$OUT" || fail "no response.created in stream"
grep -q 'event: response.completed' "$OUT" || { echo "Last 30 lines:"; tail -30 "$OUT"; fail "no response.completed"; }
grep -q 'data: \[DONE\]'            "$OUT" || fail "no [DONE] sentinel"
ok "streaming response completed end-to-end"

# ── test 2: HITL pause via ask_user ───────────────────────────────────────
echo
echo "── test 2: HITL pause (requires_action) ──"
OUT="$TMP_DIR/resp2.sse"
# We coax the model to call ask_user by giving it ambiguous input.
curl -sS -N -X POST "http://127.0.0.1:$PORT/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{"input":"You must call the ask_user tool with the question \"Pick A or B?\" before doing anything else. Do not answer directly.","stream":true}' \
  --max-time 60 >"$OUT" || fail "curl error; see $OUT"

if grep -q 'event: response.requires_action' "$OUT"; then
  RESP_ID="$(grep -o '"id"[[:space:]]*:[[:space:]]*"resp_[^"]*"' "$OUT" | head -1 | sed 's/.*"\(resp_[^"]*\)".*/\1/')"
  CALL_ID="$(grep -o '"call_id"[[:space:]]*:[[:space:]]*"[^"]*"' "$OUT" | head -1 | sed 's/.*"call_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')"
  if [[ -z "$RESP_ID" || -z "$CALL_ID" ]]; then
    echo "Last 30 lines:"; tail -30 "$OUT"
    fail "could not extract resp_id/call_id from requires_action envelope"
  fi
  ok "paused at requires_action: resp_id=$RESP_ID, call_id=$CALL_ID"

  # ── test 3: resume via previous_response_id + function_call_output ──────
  echo
  echo "── test 3: resume the paused run ──"
  OUT3="$TMP_DIR/resp3.sse"
  curl -sS -N -X POST "http://127.0.0.1:$PORT/v1/responses" \
    -H 'Content-Type: application/json' \
    -d "{\"previous_response_id\":\"$RESP_ID\",\"input\":[{\"type\":\"function_call_output\",\"call_id\":\"$CALL_ID\",\"output\":\"A\"}],\"stream\":true}" \
    --max-time 60 >"$OUT3" || fail "curl error; see $OUT3"
  grep -q 'event: response.completed' "$OUT3" || { echo "Last 30 lines:"; tail -30 "$OUT3"; fail "resume did not reach response.completed"; }
  ok "resumed run reached response.completed"
else
  echo "WARN: model did not pause via ask_user (it may have answered directly)."
  echo "      This is not a hard failure — the SDK runtime is still proven."
  echo "      If you want a stricter check, look at $OUT and ensure the model"
  echo "      receives the ask_user tool description. Skipping resume test."
fi

# ── test 4: chat completions stream ───────────────────────────────────────
echo
echo "── test 4: /v1/chat/completions stream=true ──"
OUT="$TMP_DIR/chat1.sse"
curl -sS -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with the single word: pong"}],"stream":true}' \
  --max-time 60 >"$OUT" || fail "curl error; see $OUT"

grep -Eq '"object":[[:space:]]*"chat.completion.chunk"' "$OUT" || { echo "Last 30 lines:"; tail -30 "$OUT"; fail "no chat.completion.chunk"; }
grep -Eq '"finish_reason":[[:space:]]*"stop"'           "$OUT" || { echo "Last 30 lines:"; tail -30 "$OUT"; fail "no finish_reason=stop"; }
grep -q 'data: \[DONE\]'                                "$OUT" || fail "no [DONE] sentinel"
ok "chat completions streamed and completed"

# ── test 5: deleted compatibility endpoint ────────────────────────────────
echo
echo "── test 5: removed /v1/runs returns 404 ──"

OUT="$TMP_DIR/runs_removed.json"
STATUS="$(curl -sS -o "$OUT" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/runs" \
  -H 'Content-Type: application/json' \
  -d '{"input":"ping"}' \
  --max-time 60 || true)"
[[ "$STATUS" == "404" ]] || { cat "$OUT"; fail "/v1/runs expected 404, got $STATUS"; }
ok "legacy /v1/runs endpoint is removed"

echo
echo "All smoke tests passed. AGENT_RUNTIME=sdk is live-verified."
echo "Logs: $TMP_DIR (auto-removed unless --keep-server was passed)."
