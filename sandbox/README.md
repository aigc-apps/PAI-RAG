# PAI-Loop AgentRun Sandbox Images

This directory builds the **sandbox images** — AgentRun templates that run
inside the sandbox (Function Compute) and fulfill the runtime contract so the
`AGENT_*` env vars and the per-user / per-skill NAS mounts resolve.

There are two templates, sharing one base:

```
sandbox/
  base/Dockerfile               # contract + shared toolchain (not pushed; local only)
  base/agent-sandbox-bootstrap  # startup validation, copied into base
  pairec/Dockerfile             # + release-pinned source snapshot + PAI plugins
  turbox/Dockerfile             # + live git working copies + pai-dsw plugin
  build.sh                      # build.sh <pairec|turbox>: builds base, then the template
  README.md
```

- **`pairec`** — recommendation-system source (`/opt/code` from an OSS
  tarball), root-owned and read-only, plus the PAI recommendation Aliyun
  plugins.
- **`turbox`** — two intranet gitlab repositories cloned as live git working
  copies at build time, owned by the runtime user (uid 1000) so the agent can
  check out branches, plus the `aliyun-cli-pai-dsw` plugin.

The agent service (this repo's `backend/`) only **references** a template by
name via `sandbox.default.settings.templates` (a map of template-key →
`{name, code_writable?, env_refs?}`) plus `settings.default_template`, and
sends `nasConfig` + the env contract at sandbox create time. Each Agent binds
to a key via `agents[].sandbox.template` (blank uses `default_template`). The
images themselves are built and registered **outside** the agent service. See
`docs/design/skill_install_mount_dependencies.md` → "Sandbox Image Contract"
for the full mount contract.

## Path convention: `/opt/*` vs `/mnt/*`

The two prefixes are **not** interchangeable — each encodes how the files got
there, so keep new paths on the right side of the line:

- **`/opt/*` = image layer (local disk).** Currently just `/opt/code` (the code
  layer). Baked rather than mounted because local disk beats NFS for the
  browse/grep workload; content is fixed at image-build time. Whether it is
  writable is per-template: pai-rec bakes a root-owned snapshot, turbox clones
  git working copies owned by the runtime user so the agent can check out
  branches. Writes land in the container's per-instance layer and are discarded
  with the sandbox.
- **`/mnt/*` = NAS mounts, attached per-sandbox at create time** via
  `nasConfig`. These are the three contract paths: `/mnt/system`, `/mnt/skills`
  (read-only), and `/mnt/user` (read-write, per-user). Content is dynamic and
  lives on the NAS, not in the image; `agent-sandbox-bootstrap` validates all
  three exist on start (contract v1).

Do **not** move a NAS mount (e.g. skills) under `/opt`, or a baked layer under
`/mnt` — that mixes backing stores within one prefix and breaks the mount
contract the bootstrap enforces. Skills stay at `/mnt/skills/<mount_id>`; code
stays at `/opt/code`.

## What the images do

`base/Dockerfile` (shared by both templates):

- `mkdir /mnt/system /mnt/skills /mnt/user /opt/code /home/user` (empty dirs —
  `/mnt/*` content comes from NAS mounts at sandbox start, never baked in) +
  `chown -R 1000:1000 /mnt/user /home/user` (the platform runs the sandbox as
  uid 1000, matching `nasConfig.userId`).
- Installs the shared toolchain: **ripgrep** (`rg`) and **jq** for the
  browse/grep workload the agent is prompted to run against `/opt/code`, plus
  **git** (templates clone or manage working copies there), **curl**, and
  **ca-certificates**.
- Installs **yq** (pinned binary from the upstream GitHub release — not in
  Debian main).
- Bakes the **Aliyun CLI binary** only (`/usr/local/bin/aliyun`); each
  template installs its own plugins as uid 1000 so they land in `~/.aliyun`.
- Bakes a secret-free `git-credential-env` helper
  (`/usr/local/bin/git-credential-env`): with `credential.helper = env`, git
  execs it from `PATH` and it prints `password=${GITLAB_TOKEN}` — the token
  itself never touches this layer, only the mechanism that reads it at
  runtime. The gitlab-host binding (`credential."http://gitlab...".helper
  env`) lives in `turbox/Dockerfile`, not here, since pai-rec never needs it.
- Bakes `AGENT_SYSTEM_PATH` / `AGENT_SKILL_PATH` / `AGENT_USER_PATH` /
  `AGENT_CODE_PATH=/opt/code` / `AGENT_ENV_PATH` / `PATH` / `VIRTUAL_ENV` env
  vars, and `HOME=/home/user`.
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

`pairec/Dockerfile` (`FROM sandbox-base:latest`):

- Downloads `--build-arg CODE_ARCHIVE_URL=<archive.tar.gz>` (defaults to a
  release-pinned tarball) and extracts it into `/opt/code` with
  `--strip-components=1` (so repos land at `/opt/code/<repo>`, not
  `/opt/code/<wrapper>/<repo>`), then `chmod -R a+rX /opt/code`. Pass an empty
  URL to build without the layer. Root-owned, matching `templates.pairec`
  having no `code_writable`.
- Installs the PAI Aliyun plugins as uid 1000: `aliyun-cli-eas`,
  `aliyun-cli-pairecservice`, `aliyun-cli-pai-dsw`, `aliyun-cli-paifeaturestore`.

`turbox/Dockerfile` (`FROM sandbox-base:latest`):

- Clones `--build-arg TURBOX_REPOS` (defaults to
  `gitlab.alibaba-inc.com/PAI/image-metadata` and
  `gitlab.alibaba-inc.com/pai-ee/pai-wiki`, both `@master`) into
  `/opt/code/<repo>` with `git clone --depth 1 --no-single-branch`, which
  fetches every branch tip at one commit each so the agent can `checkout` any
  branch that existed at build time **offline**, and `git fetch --deepen` /
  `--unshallow` for history once the sandbox itself reaches the intranet.
  `.git` is deliberately **retained** (this is a working copy, not a
  snapshot).
- `chown -R 1000:1000 /opt/code` — this is what makes checkout possible, and
  is why `templates.turbox` must set `code_writable: true` (see below).
- Installs `aliyun-cli-pai-dsw` as uid 1000.

**No token reaches any image layer.** Build-time auth is a BuildKit secret
(`--mount=type=secret,id=gitcred`) consumed via `git -c
credential.helper='store --file=/root/.git-credentials'`, which keeps the
remote clone URL clean. Embedding the token in the URL instead
(`http://user:token@gitlab...`) would persist it in
`/opt/code/<repo>/.git/config` — and since `.git` is retained here, the token
would ship with the image. Runtime auth is a different, separate path:
`credential.helper=env` → the base's `git-credential-env` script →
`$GITLAB_TOKEN`, injected per-template via `env_refs` (see "Wire to the agent
service" below) — not baked into the image at all.

## Build

```bash
sandbox/build.sh pairec
sandbox/build.sh turbox        # needs GITLAB_TOKEN + intranet access to gitlab.alibaba-inc.com
```

`build.sh` always builds `sandbox-base:latest` first, then the requested
template `FROM`s it. `sandbox-base` is **not pushed to a registry** — a clean
CI machine that has never run `build.sh` has no local `sandbox-base:latest`,
so `FROM sandbox-base:latest` in `pairec/Dockerfile` / `turbox/Dockerfile`
fails to resolve unless the base step runs first. Both steps run automatically
every time you invoke `build.sh`; there is no separate "build base once, reuse
forever" flow.

Environment variables `build.sh` reads:

- `BASE_IMAGE` — the official AgentRun code-interpreter image ref. Find it in
  the AgentRun console: code-interpreter template → image address. Defaults to
  `registry.aliyuncs.com/agentrun/sandbox-code-interpreter:latest`. Only used
  for the base build.
- `GITLAB_TOKEN` — read token for `gitlab.alibaba-inc.com`. Required for
  `sandbox/build.sh turbox` (checked before the base build even starts, so a
  missing token fails fast instead of after a full base build). Not read for
  `pairec`. Passed to Docker as a **BuildKit secret**, never as a
  `--build-arg` — build-args land in `docker history` and would leak the
  token into the image metadata.

`docker build` runs with `DOCKER_BUILDKIT=1` under the hood (required for
`--mount=type=secret` in `turbox/Dockerfile`).

Both builds need network access (Aliyun CLI download, apt packages, yq
release, plugin installs; `pairec` also needs the OSS tarball URL reachable,
`turbox` needs `gitlab.alibaba-inc.com` reachable).

## Register in AgentRun

You register **two** templates — one per image — both under the same account
and region:

1. Push both built images (`sandbox-pairec:latest`, `sandbox-turbox:latest`)
   to a registry AgentRun can read (e.g. your ACR).
2. In the [AgentRun console](https://functionai.console.aliyun.com/cn-hangzhou/agent/infra),
   create two code-interpreter templates:
   - **pai-rec** — image `sandbox-pairec:latest`. Network mode **public**
     (reaches the NAS mount target and the agent gateway; no intranet gitlab
     access needed).
   - **turbo-x** — image `sandbox-turbox:latest`. Network mode **VPC**
     (vswitch + security group that can reach `gitlab.alibaba-inc.com`, since
     the agent's runtime `git fetch --deepen` / `--unshallow` needs intranet
     access; the NAS mounts and agent gateway must also be reachable from that
     VPC).
3. Make sure both templates are created in the same **region**
   (`cn-hangzhou`) and under the same **account_id** (`AGENTRUN_ACCOUNT_ID`)
   the agent service uses — a region/account mismatch is the most common cause
   of `404 template not found` on sandbox create. Network placement differs
   per template, but endpoint and credentials stay provider-level.

## Wire to the agent service

In `data/config.yaml` (or via the Settings → Sandbox UI):

```yaml
providers:
  - id: sandbox.default
    settings:
      provider: agentrun_rest
      account_id_env: AGENTRUN_ACCOUNT_ID
      api_key_env: AGENTRUN_SANDBOX_API_KEY
      # endpoint auto-derived as https://{account_id}.agentrun-data.cn-hangzhou.aliyuncs.com
      templates:
        pairec:
          name: sandbox-code-feiyue        # console template name, network mode: public
        turbox:
          name: sandbox-turbox-feiyue      # console template name, network mode: VPC
          code_writable: true              # /opt/code is chown 1000 in this image — must match
          env_refs:
            GITLAB_TOKEN: GITLAB_TOKEN     # value = which process env var to read
      default_template: pairec

agents:
  - id: main
    code: { enabled: true, manifest: "..." }
  - id: turbox-helper
    sandbox: { template: turbox }
    code: { enabled: true, manifest: "image-metadata / pai-wiki ..." }
```

Each `templates` entry's `name` is the console template name and must match
what you registered above. An Agent picks a template via
`agents[].sandbox.template`; a blank value falls back to
`settings.default_template`. An unknown key is a hard error at sandbox-create
time (it lists the valid keys) — there is no silent fallback, because handing
a turbo-x Agent a pai-rec sandbox would give it an `/opt/code` full of
recommendation-system source while its manifest promises `image-metadata` /
`pai-wiki`, and the agent would stay confused indefinitely with no error to
point at.

`code_writable` **must match the image**, not express a per-agent permission.
It is a fact about how the template's `/opt/code` was built: turbox's
`chown -R 1000:1000 /opt/code` makes it writable by the runtime user; pairec's
`chmod -R a+rX /opt/code` (root-owned) does not. `code_writable` drives the
wording of the code-layer guidance in the system prompt (checkoutable working
copies vs. read-only reference material — see `backend/agent/soul.py`'s
`_code_layer_block`). Setting `code_writable: true` on a template whose image
is root-owned tells the agent it may check out branches in a directory it
cannot actually write to.

`env_refs` maps a variable name the sandbox should see (`GITLAB_TOKEN`) to the
name of the process environment variable to read it from — usually the same
name, but the indirection lets you rename without touching the sandbox side.
Only templates that declare an `env_refs` entry receive that variable; pairec
declares none, so its sandboxes never see `GITLAB_TOKEN`. Resolution happens
inside the sandbox provider and rides the same delivery path as the rest of
the env contract (`~/.bash_env`, sourced by every non-interactive shell) — it
is not a new mechanism.

After changing `.env` or `data/config.yaml`, hit
`POST /v1/config/reload-env` (see `backend/app/routes/config.py`) so the
running provider picks up the change without a restart.

## Contract version

The bootstrap script emits `contract v1`. It validates only the three
`/mnt/*` mounts (`/mnt/system`, `/mnt/skills`, `/mnt/user`) — `/opt/code` is
outside its checks, and this split (base vs. per-template images) does not
change the contract. If the mount layout or env-var set changes, bump
`CONTRACT_VERSION` in `agent-sandbox-bootstrap` and in the agent service's
`_build_env_contract` so a stale image vs. a stale agent service can detect
the mismatch.
