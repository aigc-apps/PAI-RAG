# PAI-RAG AgentRun Sandbox Image

This directory builds the **sandbox image** — the AgentRun template that runs
inside the sandbox (Function Compute) and fulfills the runtime contract so the
`AGENT_*` env vars and the per-user / per-skill NAS mounts resolve.

The agent service (this repo's `newbackend/`) only **references** this template
by name (`sandbox.default.template_name`, e.g. `sandbox-code-feiyue`) and sends
`nasConfig` + `envs` at sandbox create time. The image itself is built and
registered **outside the agent service**. See
`docs/design/skill_install_mount_dependencies.md` → "Sandbox Image Contract"
for the full contract.

## What the image does

- `mkdir /mnt/system /mnt/skills /mnt/user` (empty dirs — content comes from
  NAS mounts at sandbox start, never baked in) + `chown -R 1000:1000 /mnt/user`
  (the platform runs the sandbox as uid 1000, matching `nasConfig.userId`).
- Ensures `/home/user` exists and is owned by 1000, sets `HOME=/home/user`.
- Bakes the **Aliyun CLI** (`/usr/local/bin/aliyun`) and the PAI plugins
  (`aliyun-cli-eas`, `aliyun-cli-pairecservice`, `aliyun-cli-pai-dsw`,
  `aliyun-cli-paifeaturestore`) installed as uid 1000 into `~/.aliyun`, so
  skill scripts can drive Aliyun services from inside the sandbox.
- Bakes `AGENT_SYSTEM_PATH` / `AGENT_SKILL_PATH` / `AGENT_USER_PATH` /
  `AGENT_ENV_PATH` / `PATH` / `VIRTUAL_ENV` env vars.
- Runs `agent-sandbox-bootstrap` as ENTRYPOINT: validates the three mounts
  exist and are readable on start, fails fast with a contract-version marker
  if not, then execs the base image's startup chain. The CMD is set explicitly
  to `/usr/local/bin/entrypoint.sh process-compose up --tui=false --no-server`
  (the base code-interpreter chain) — Docker clears the inherited CMD when
  ENTRYPOINT is overridden, so it must be restored or FC runs the bootstrap
  with no args and the instance exits 0 ("Function instance exited
  unexpectedly"). `entrypoint.sh` only injects its process-compose config when
  `$1 == process-compose`, so CMD's first arg must stay `process-compose`.

`/mnt/skills` is intentionally **not** on `PYTHONPATH` — cross-skill module
name collisions. Skill scripts are invoked by absolute path.

## Build

```bash
# BASE_IMAGE = the official AgentRun code-interpreter image ref.
# Find it in the AgentRun console: code-interpreter template → image address.
docker build \
  --build-arg BASE_IMAGE=<official-code-interpreter-image-ref> \
  -t sandbox-code-feiyue:latest \
  sandbox/

# Override the Aliyun CLI release if needed (defaults to the URL in the
# Dockerfile's ARG ALIYUN_CLI_URL):
# docker build --build-arg ALIYUN_CLI_URL=https://.../aliyun-cli-linux-latest-amd64.tgz ...
```

The build needs network access (Aliyun CLI download + plugin install). The
base image must have `curl` and `tar` — the official code-interpreter base
ships them; if yours does not, add `apt-get install -y --no-install-recommends
curl tar` before the CLI install step.

If you don't have the official ref handy, set `BASE_IMAGE` to your own
python:3.11-based image and add the FC sandbox runtime components yourself
(the AgentRun platform expects the FC agent / runtime to be present in the
image — the official code-interpreter base already has it).

## Register in AgentRun

1. Push the built image to a registry AgentRun can read (e.g. your ACR).
2. In the [AgentRun console](https://functionai.console.aliyun.com/cn-hangzhou/agent/infra),
   create a new code-interpreter template whose image is the pushed ref,
   **named exactly** `sandbox-code-feiyue` (must match
   `sandbox.default.template_name` in `data/config.yaml`). Network mode:
   public (so it can reach the NAS mount target and the agent gateway).
3. Make sure the template is created in the same **region** (`cn-hangzhou`)
   and under the same **account_id** (`AGENTRUN_ACCOUNT_ID`) the agent
   service uses — a region/account mismatch is the most common cause of
   `404 template not found` on sandbox create.

## Wire to the agent service

In `data/config.yaml` (or via the Settings → Sandbox UI):

```yaml
providers:
  - id: sandbox.default
    settings:
      provider: agentrun_rest
      template_name: sandbox-code-feiyue   # matches the registered template
      account_id_env: AGENTRUN_ACCOUNT_ID
      api_key_env: AGENTRUN_SANDBOX_API_KEY
      # endpoint auto-derived as https://{account_id}.agentrun-data.cn-hangzhou.aliyuncs.com
```

After changing `.env` or the template name, hit
`POST /v1/config/reload-env` (see `newbackend/app/routes/config.py`) so the
running provider picks it up without a restart.

## Contract version

The bootstrap script emits `contract v1`. If the mount layout or env-var set
changes, bump `CONTRACT_VERSION` in `agent-sandbox-bootstrap` and in the
agent service's `_build_env_contract` so a stale image vs. a stale agent
service can detect the mismatch.
