#!/usr/bin/env bash
# Build a sandbox template image. Always builds the shared base first — the
# per-template Dockerfiles do `FROM sandbox-base:latest`, which does not resolve
# on a machine that has never built it.
#
#   sandbox/build.sh pairec
#   sandbox/build.sh turbox            # needs GITLAB_TOKEN + intranet access
#
# BASE_IMAGE  the official AgentRun code-interpreter image ref (console:
#             code-interpreter template -> image address).
# GITLAB_TOKEN  read token for gitlab.alibaba-inc.com; turbox only. Passed as a
#             BuildKit secret, never as a build-arg (build-args land in
#             `docker history`).
set -euo pipefail

TEMPLATE="${1:-}"
BASE_IMAGE="${BASE_IMAGE:-registry.aliyuncs.com/agentrun/sandbox-code-interpreter:latest}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$TEMPLATE" in
  pairec|turbox) ;;
  *) echo "usage: $0 {pairec|turbox}" >&2; exit 2 ;;
esac

# Checked before the base build starts: turbox needs GITLAB_TOKEN, and we want
# a missing token to fail fast (no docker invocation at all) rather than after
# spending a full base build reaching for a registry.
if [ "$TEMPLATE" = "turbox" ]; then
  : "${GITLAB_TOKEN:?turbox needs GITLAB_TOKEN (read token for gitlab.alibaba-inc.com)}"
fi

echo "==> building sandbox-base:latest (BASE_IMAGE=${BASE_IMAGE})"
DOCKER_BUILDKIT=1 docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t sandbox-base:latest \
  "${here}/base"

echo "==> building sandbox-${TEMPLATE}:latest"
if [ "$TEMPLATE" = "turbox" ]; then
  cred="$(mktemp)"
  trap 'rm -f "$cred"' EXIT
  printf 'http://oauth2:%s@gitlab.alibaba-inc.com\n' "$GITLAB_TOKEN" > "$cred"
  DOCKER_BUILDKIT=1 docker build \
    --secret "id=gitcred,src=${cred}" \
    -t "sandbox-${TEMPLATE}:latest" \
    "${here}/${TEMPLATE}"
else
  DOCKER_BUILDKIT=1 docker build \
    -t "sandbox-${TEMPLATE}:latest" \
    "${here}/${TEMPLATE}"
fi

echo "==> built sandbox-${TEMPLATE}:latest"
