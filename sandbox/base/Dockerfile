# syntax=docker/dockerfile:1
# =============================================================================
# PAI-Loop AgentRun sandbox — shared base layer.
#
# Owns the runtime contract (docs/design/skill_install_mount_dependencies.md):
# the three NAS mount points, the AGENT_* env vars, the startup validation, and
# the toolchain every template shares. Per-template images (sandbox/pairec,
# sandbox/turbox) FROM this and add only their code layer and aliyun plugins.
#
#   docker build -t sandbox-base:latest \
#     --build-arg BASE_IMAGE=<official-code-interpreter-image-ref> sandbox/base/
#
# Not pushed to a registry — sandbox/build.sh builds it locally first. A clean
# CI machine must run both steps or `FROM sandbox-base:latest` will not resolve.
# =============================================================================

ARG BASE_IMAGE=registry.aliyuncs.com/agentrun/sandbox-code-interpreter:latest
FROM ${BASE_IMAGE}

ARG ALIYUN_CLI_URL=https://easyrec.oss-cn-beijing.aliyuncs.com/aliyun-cli-spec/aliyun-cli-linux-latest-amd64.tgz
ARG YQ_VERSION=v4.44.3

# The platform runs the sandbox as uid/gid 1000 (matches nasConfig.userId /
# groupId). Content for /mnt/system and /mnt/skills arrives via read-only NAS
# mounts at sandbox start — they stay empty dirs in the image.
USER root
RUN mkdir -p /mnt/system /mnt/skills /mnt/user /opt/code /home/user \
 && chown -R 1000:1000 /mnt/user /home/user

# Toolchain shared by every template: ripgrep/jq for the browse+grep workload the
# agent is prompted to run against /opt/code, git for template code layers that
# are working copies rather than snapshots.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ripgrep jq git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && rg --version && jq --version && git --version

# yq is not in Debian main; pinned binary from the upstream release.
RUN curl -fsSL "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64" \
      -o /usr/local/bin/yq \
 && chmod 0755 /usr/local/bin/yq && yq --version

# Aliyun CLI binary only — each template installs its own plugins as uid 1000.
RUN tmpdir="$(mktemp -d)" \
    && curl -fsSL "${ALIYUN_CLI_URL}" -o /tmp/aliyun-cli.tgz \
    && tar -xzf /tmp/aliyun-cli.tgz -C "${tmpdir}" \
    && install -m 0755 "${tmpdir}/aliyun" /usr/local/bin/aliyun \
    && rm -rf "${tmpdir}" /tmp/aliyun-cli.tgz \
    && aliyun version

# Secret-free credential helper. `credential.helper = env` makes git exec
# git-credential-env from PATH; the token itself arrives at runtime in
# $GITLAB_TOKEN via the agent service's env contract (~/.bash_env), so nothing
# secret is baked into any layer. Templates that need it bind it to a host.
RUN printf '%s\n' \
      '#!/bin/sh' \
      '# git credential helper: reads the token from the environment.' \
      '[ "$1" = "get" ] || exit 0' \
      'echo "username=oauth2"' \
      'echo "password=${GITLAB_TOKEN}"' \
      > /usr/local/bin/git-credential-env \
 && chmod 0755 /usr/local/bin/git-credential-env

# Startup validation: fails fast if a contract mount is missing/unreadable, so
# a mis-mounted sandbox surfaces a clear error instead of a confusing 404 later.
COPY agent-sandbox-bootstrap /usr/local/bin/agent-sandbox-bootstrap
RUN chmod +x /usr/local/bin/agent-sandbox-bootstrap

# Runtime env contract. HOME is explicit because a numeric USER does not always
# populate it, and the aliyun plugins each template installs must land in $HOME.
ENV AGENT_SYSTEM_PATH=/mnt/system \
    AGENT_SKILL_PATH=/mnt/skills \
    AGENT_USER_PATH=/mnt/user \
    AGENT_CODE_PATH=/opt/code \
    AGENT_ENV_PATH=/mnt/system/skill-envs/current \
    PATH=/mnt/system/skill-envs/current/bin:/usr/local/bin:/mnt/system/bin:${PATH} \
    VIRTUAL_ENV=/mnt/system/skill-envs/current \
    HOME=/home/user

# NOTE: /mnt/skills is intentionally NOT added to PYTHONPATH globally —
# cross-skill module name collisions. Skill scripts are invoked by absolute path.

ENV SANDBOX_BASE_ENTRYPOINT=/usr/local/bin/entrypoint.sh \
    SANDBOX_BASE_CMD="process-compose up --tui=false --no-server"

USER 1000

# FC uses the image ENTRYPOINT as the start command and CMD as its args.
# Setting ENTRYPOINT cleared the CMD inherited from the base image (which is
# how we ended up with CMD=null and the instance exiting 0 — FC reports that
# as "Function instance exited unexpectedly", with nothing pointing at the
# ENTRYPOINT override as the cause), so restore the base startup chain
# explicitly; keep CMD's first arg as "process-compose" so entrypoint.sh
# takes its config-injection branch.
ENTRYPOINT ["agent-sandbox-bootstrap"]
CMD ["/usr/local/bin/entrypoint.sh", "process-compose", "up", "--tui=false", "--no-server"]
