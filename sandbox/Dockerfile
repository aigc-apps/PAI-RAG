# syntax=docker/dockerfile:1
# =============================================================================
# PAI-RAG AgentRun sandbox image
#
# Implements the runtime contract documented in
#   docs/design/skill_install_mount_dependencies.md
# so the AGENT_* env vars resolve and the per-user / per-skill NAS mounts land
# on pre-created, correctly-owned directories. Also bakes the Aliyun CLI +
# PAI plugins so skill scripts can drive Aliyun services from inside the box,
# and a release-pinned read-only code layer at /opt/code for grep/read.
#
# Build on top of the official AgentRun code-interpreter image. Pass its
# registry ref via --build-arg BASE_IMAGE=... (find it in the AgentRun console
# under the code-interpreter template's image address).
#
#   docker build -t sandbox-code-feiyue:latest \
#     --build-arg BASE_IMAGE=<official-code-interpreter-image-ref> \
#     sandbox/
#
# Then push to the registry AgentRun reads from, and register it as a template
# named `sandbox-code-feiyue` (or whatever sandbox.default.template_name points
# at) in the AgentRun console. See sandbox/README.md for the full flow.
# =============================================================================

ARG BASE_IMAGE=registry.aliyuncs.com/agentrun/sandbox-code-interpreter:latest
FROM ${BASE_IMAGE}

# Aliyun CLI download URL (bundled CLI + PAI plugins). Override with
# --build-arg ALIYUN_CLI_URL=... if a different release is needed.
ARG ALIYUN_CLI_URL=https://easyrec.oss-cn-beijing.aliyuncs.com/aliyun-cli-spec/aliyun-cli-linux-latest-amd64.tgz

# Read-only code layer baked into the image at /opt/code: a release-pinned
# snapshot of the source repositories (one per subdirectory) the agent greps and
# reads when the knowledge base can't answer. Baked rather than NAS-mounted so
# grep/read hit local disk instead of NFS. Bump the URL to ship newer source;
# override with --build-arg PAIREC_CODE_ARCHIVE_URL="" to build without the layer.
ARG PAIREC_CODE_ARCHIVE_URL=https://pai-rag.oss-cn-hangzhou.aliyuncs.com/production_artifacts/pairec_code/pairec_code_20260623.tar.gz

# The platform runs the sandbox as uid/gid 1000 (matches nasConfig.userId /
# groupId). Create the three contract mount points as root, ensure the
# runtime user's home exists and is owned by 1000, and make /mnt/user owned
# by 1000 so it can write. Content for /mnt/system and /mnt/skills arrives
# via read-only NAS mounts at sandbox start — they stay empty dirs in the
# image.
USER root
RUN mkdir -p /mnt/system /mnt/skills /mnt/user /opt/code /home/user \
 && chown -R 1000:1000 /mnt/user /home/user

# Bake the read-only code layer into /opt/code. Kept in its own layer (and last
# among the root steps that change rarely) so it re-downloads only when the
# archive URL changes. Root-owned + world-readable: the runtime user (uid 1000)
# only ever reads it. Guarded so an empty PAIREC_CODE_ARCHIVE_URL builds an
# image with an empty /opt/code (the code layer stays off via config).
RUN if [ -n "${PAIREC_CODE_ARCHIVE_URL}" ]; then \
      curl -fsSL "${PAIREC_CODE_ARCHIVE_URL}" -o /tmp/pairec_code.tar.gz \
      && tar -xzf /tmp/pairec_code.tar.gz -C /opt/code \
      && rm -f /tmp/pairec_code.tar.gz \
      && chmod -R a+rX /opt/code \
      && echo "baked code layer:" && ls -1 /opt/code ; \
    fi

# Startup validation: fails fast if a contract mount is missing/unreadable,
# so a mis-mounted sandbox surfaces a clear error instead of a confusing 404
# later. It execs the base image's CMD afterwards.
COPY agent-sandbox-bootstrap /usr/local/bin/agent-sandbox-bootstrap
RUN chmod +x /usr/local/bin/agent-sandbox-bootstrap

# Aliyun CLI binary (system-wide, root-owned). Requires curl + tar in the
# base image — the official code-interpreter base ships them; if yours does
# not, add an `apt-get install -y --no-install-recommends curl tar` first.
RUN tmpdir="$(mktemp -d)" \
    && curl -fsSL "${ALIYUN_CLI_URL}" -o /tmp/aliyun-cli.tgz \
    && tar -xzf /tmp/aliyun-cli.tgz -C "${tmpdir}" \
    && install -m 0755 "${tmpdir}/aliyun" /usr/local/bin/aliyun \
    && rm -rf "${tmpdir}" /tmp/aliyun-cli.tgz \
    && aliyun version

# Runtime env contract. PATH/VIRTUAL_ENV point at the shared skill env
# (populated by the builder in phase 2); AGENT_*_PATH are the mount markers
# the agent service also injects via the platform `envs` field at create time.
# HOME is set explicitly because a numeric USER does not always populate it,
# and the Aliyun CLI plugins below must land in the agent's $HOME.
ENV AGENT_SYSTEM_PATH=/mnt/system \
    AGENT_SKILL_PATH=/mnt/skills \
    AGENT_USER_PATH=/mnt/user \
    AGENT_CODE_PATH=/opt/code \
    AGENT_ENV_PATH=/mnt/system/skill-envs/current \
    PATH=/mnt/system/skill-envs/current/bin:/usr/local/bin:/mnt/system/bin:${PATH} \
    VIRTUAL_ENV=/mnt/system/skill-envs/current \
    HOME=/home/user

# NOTE: /mnt/skills is intentionally NOT added to PYTHONPATH globally —
# cross-skill module name collisions. Skill scripts are invoked by absolute
# path.

# Base startup chain defaults (used by agent-sandbox-bootstrap when the
# platform invokes the image with no CMD args). The base code-interpreter
# image starts via /usr/local/bin/entrypoint.sh -> process-compose; entrypoint.sh
# only auto-injects its process-compose config when $1 == "process-compose",
# so the CMD below must keep that exact shape.
ENV SANDBOX_BASE_ENTRYPOINT=/usr/local/bin/entrypoint.sh \
    SANDBOX_BASE_CMD="process-compose up --tui=false --no-server"

USER 1000

# Aliyun CLI plugins — installed AS the runtime user (uid 1000) so they land
# in the agent's $HOME (~/.aliyun) and are available to skill scripts at run
# time. Requires network access during build.
RUN aliyun plugin install --names \
       aliyun-cli-eas \
       aliyun-cli-pairecservice \
       aliyun-cli-pai-dsw \
       aliyun-cli-paifeaturestore \
 && aliyun plugin list

# FC uses the image ENTRYPOINT as the start command and CMD as its args.
# Setting ENTRYPOINT cleared the CMD inherited from the base image (which is
# how we ended up with CMD=null and the instance exiting 0), so restore the
# base startup chain explicitly: bootstrap validates mounts, then execs
# entrypoint.sh which preps env + injects the process-compose config and
# launches the long-running FC runtime. Keep CMD's first arg as
# "process-compose" so entrypoint.sh takes its config-injection branch.
ENTRYPOINT ["agent-sandbox-bootstrap"]
CMD ["/usr/local/bin/entrypoint.sh", "process-compose", "up", "--tui=false", "--no-server"]
