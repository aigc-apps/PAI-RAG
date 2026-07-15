# Skill Install, Mount, and Dependency Design

## Goal

Support custom skills in a multi-user web agent service without letting ordinary
user requests mutate the agent runtime. Skills are installed by admins, assigned
to agents, mounted into sandboxes by agent profile, and backed by prebuilt
dependency environments.

## Concepts

- **Skill package**: A directory under `skills.root` with `skill.yaml`,
  `SKILL.md`, optional `scripts/`, `resources/`, `requirements.txt`, and
  `package.json`.
- **Skill mount**: A read-only sandbox mount for an enabled skill, for example
  `/mnt/skills/report-writer`.
- **Skill env**: A prebuilt dependency environment for the current agent's
  enabled skills, for example `/mnt/skill-envs/<env_fingerprint>`.
- **Runtime tool**: A normal tool available during user requests, such as
  `web_search`, `knowledge_search`, and `code_interpreter`.
- **Control-plane tool**: An admin-only tool that changes platform state, such
  as `install_skill`, `rebuild_skill_env`, and `enable_skill_for_agent`.

## Config Shape

```yaml
skills:
  root: ./data/skills        # deployed on a NAS filesystem shared read-only with sandboxes
  mount:
    mount_root: /mnt/skills
    nas:
      server_addr: xxxx.nas.aliyuncs.com:/
      remote_path_prefix: skills
      read_only: true
  install:
    enabled: true
    admin_only: true
    allow_sources:
      - zip_upload
      - url
      - git
    production_allow_sources:
      - zip_upload
    require_review: true
    max_download_mb: 50
  dependencies:
    mode: prebuilt
    env_root: /mnt/skill-envs
    admin_build_enabled: true
    runtime_install_enabled: false
    build_timeout_seconds: 600
  installed: []
```

The sandbox provider (`sandbox.default`) carries the per-user NAS mount and
the env-var contract:

```yaml
providers:
  - id: sandbox.default
    settings:
      provider: agentrun_rest
      nas_config:
        user_id: 1000
        group_id: 1000
        user_server_addr: xxxx.nas.aliyuncs.com:/
        user_remote_path_template: /users/{user_id}
        user_read_only: false
      inject_env_contract: true
      extra_envs: {}
      # Code layer at /opt/code is baked into the sandbox image, not NAS-mounted.
      # Keyed by template so an agent can be bound to a different image/repo set;
      # `code_writable` is an image fact (does this template's /opt/code exist as
      # a writable git working copy, or a root-owned snapshot?), not a per-agent
      # permission. See "Sandbox Image Contract" below.
      templates:
        pairec:
          name: sandbox-code-feiyue
        turbox:
          name: sandbox-turbox-feiyue
          code_writable: true
          env_refs:
            GITLAB_TOKEN: GITLAB_TOKEN
      default_template: pairec
```

Agents bind to one template by key via `agents[].sandbox.template` (blank uses
`default_template`):

```yaml
agents:
  - id: main
    code: { enabled: true, manifest: "..." }
  - id: turbox-helper
    sandbox: { template: turbox }
    code: { enabled: true, manifest: "image-metadata / pai-wiki ..." }
```

Agents choose from installed skills:

```yaml
agents:
  - id: main
    skills:
      enabled:
        - skill.report-writer
```

## Install Policy

`install_skill` is an admin-only control-plane tool. The backend enforces this
in tool dispatch using request metadata, so prompt instructions alone cannot
grant access.

The Settings UI uses three control-plane HTTP endpoints:

- `POST /v1/skills/uploads`: Admin-only multipart upload for a skill zip. Returns
  an `upload_id`.
- `POST /v1/skills/install`: Admin-only install action. Accepts
  `zip_upload`, `url`, or `git` sources and refreshes the runtime config after
  install.
- `POST /v1/skills/enable`: Admin-only enable/disable of an installed, `ready`
  skill for one agent (`{skill_id, agent_id?, enabled?}`; `agent_id` defaults to
  the platform default agent). Rejects unknown or not-ready skills and refreshes
  the runtime config after the change.

Each endpoint has a matching admin-only control-plane tool (`install_skill`,
`enable_skill_for_agent`) gated by `permission="admin"` in tool dispatch, so an
operator/admin agent can drive the same actions. The tools persist to the agent
config on disk and reload the running registry + agent config in place (via the
`on_config_change` reloader) so changes take effect without a restart.

Allowed source types:

- `zip_upload`: Uploaded zip from the platform upload service.
- `url`: HTTPS zip download. Development/staging only by default.
- `git`: HTTPS Git clone with optional `ref` and package `path`.
  Development/staging only by default.

Production policy:

- URL and Git installs are disabled when `APP_ENV=production`.
- Production should use reviewed upload packages or preinstalled packages.
- Runtime user requests must never run package installation.

Minimum install safeguards:

- Admin permission required.
- HTTPS-only URL/Git sources.
- Download size limit.
- Optional SHA-256 checksum for URL installs.
- Git installs record the resolved commit SHA.
- Zip extraction rejects absolute paths and `..`.
- Skill id must be a simple package id.
- Existing package directories require `overwrite=true`.

## Dependency Policy

Dependencies are built outside ordinary user request execution.

Recommended lifecycle:

```text
installing
  -> reviewing
  -> building_dependencies
  -> ready
  -> enabled_for_agent
```

Skill dependencies may be declared through:

- `requirements.txt`
- `package.json`
- `skill.yaml.runtime`

The dependency environment should be built for the current agent's enabled skill
set, not for each skill independently:

```text
env_fingerprint =
  hash(all enabled skills' requirements.txt + package.json + runtime config)
```

The sandbox scope should include:

```text
conversation:<id>:agent:<agent_id>:skills:<skill_fingerprint>:env:<env_fingerprint>
```

First implementation status:

- `install_skill` installs the skill package and reports whether dependency
  build is required.
- `enable_skill_for_agent` enables/disables an installed, `ready` skill for an
  agent (control-plane tool + `POST /v1/skills/enable`), blocking not-ready
  skills so an agent never mounts a skill whose dependencies are unbuilt.
- Dependency build execution is reserved for `rebuild_skill_env`.
- Runtime install remains disabled.

## Sandbox Mounting

Four runtime paths are contracted inside the sandbox. Three are NAS-mounted at
create time; the code layer is baked into the image (present on every sandbox
template today — `pairec` and `turbox` each ship one — so it is no longer
gated by a provider-level flag; whether an *agent* is told about it is a
separate, per-agent decision, see below):

```text
/mnt/system   # agent-level shared, read-only; heavy deps baked into the sandbox image
/mnt/skills   # agent-level shared, read-only skill packages
/mnt/user     # per-user isolated, writable (outputs, memory)
/opt/code     # agent-level shared, source repos, BAKED into the image; writability is
              # per-template (root-owned snapshot for pairec, git working copies owned
              # by the runtime user for turbox — see templates[key].code_writable below)
```

The active agent determines skill mounts:

```text
agent.skills.enabled -> SkillMount[]
```

For each enabled and ready skill, the backend emits a NAS mount point:

```text
/mnt/skills/<skill-id>  <- nas: <server>:/skills/<skill-id>@<version>  (read-only)
```

Plus one per-user mount:

```text
/mnt/user  <- nas: <server>:/users/<user_id>  (read-write)
```

The code layer is **not** a NAS mount — it is baked into the sandbox image at
`/opt/code` (a single dir whose subdirectories are repositories; the agent
discovers them by `ls /opt/code`). Baked rather than mounted because the
workload is pure grep/read, where local disk beats NFS. It therefore
contributes no `mountPoint`. What's baked differs per template: `pairec`
extracts a release-pinned OSS tarball into a root-owned, world-readable
snapshot; `turbox` `git clone`s two intranet repositories as working copies
`chown`ed to the runtime user, so the agent can check out other branches
(`--depth 1 --no-single-branch` fetches every branch tip offline; deepening
history needs the sandbox's own network reach). Either way, changes made
inside the sandbox land in the container's per-instance writable layer and are
discarded when the sandbox ends — the baked image is never mutated.

The remaining mount points share `userId/groupId = 1000` (the platform default
for NAS mounts) at the `nasConfig` top level. `mountDir` values must not collide;
skill mounts are leaf dirs under `/mnt/skills/<id>` and the user mount is
`/mnt/user`, so they never overlap. Empty mount configs are omitted from the
create payload to avoid provider validation errors.

The env-var contract exposes each contracted path: `AGENT_SYSTEM_PATH`,
`AGENT_SKILL_PATH`, `AGENT_USER_PATH`, and `AGENT_CODE_PATH=/opt/code` — all
four are baked unconditionally into `sandbox/base/Dockerfile`, since every
template ships a code layer today. What varies is whether the *agent* is
told about it: `agents[].code.enabled` (plus an optional `code.manifest`)
gates the code-layer guidance in the system prompt, and
`agents[].sandbox.template` (resolved through
`sandbox.default.settings.templates`) picks which image — and therefore
which repositories and which `code_writable` — that agent actually gets. To
ship newer pairec source, bump `pairec/Dockerfile`'s `CODE_ARCHIVE_URL`
build-arg and rebuild; to change turbox's repositories, edit its
`TURBOX_REPOS` build-arg. See `sandbox/README.md` for the full build and
registration flow.

Skill content lives on the NAS filesystem. `install_skill` writes packages to
`skills.root`, which is deployed on the same NAS the sandbox mounts read-only —
so no separate upload/sync step is needed. The sandbox sees only the enabled
skills' mount points (per-skill mount points, not a shared root), preserving
enabled-only visibility across agents sharing the NAS.

## Runtime Env Contract

AgentRun accepts an `envs` field on sandbox creation (string map, per-sandbox,
inherited by all processes including the execution kernel). The backend injects
the `AGENT_*` marker vars:

```text
AGENT_SYSTEM_PATH=/mnt/system
AGENT_SKILL_PATH=/mnt/skills
AGENT_USER_PATH=/mnt/user
AGENT_USER_ID=<scope.user_id>
AGENT_SESSION_ID=<scope_key>
```

`PATH` and `PYTHONPATH` are intentionally NOT injected here — the flat string
map cannot interpolate, and setting them would clobber the image default. They
are baked into the sandbox image (see below). `extra_envs` merges user-supplied
values; `inject_env_contract: false` disables injection when the image bakes
the full contract.

## Sandbox Image Contract

Each sandbox image is an AgentRun template. The build recipe lives in this
repo at `sandbox/` — a shared `base/Dockerfile` (contract + toolchain,
`agent-sandbox-bootstrap`) that `pairec/Dockerfile` and `turbox/Dockerfile`
each build `FROM`, adding their own code layer and Aliyun plugins (see
`sandbox/README.md` for the full layout and build steps). The images
themselves are built and registered in AgentRun outside the agent service.
Every template must fulfill the runtime contract so the `AGENT_*` vars
resolve:

```dockerfile
RUN mkdir -p /mnt/system /mnt/skills /mnt/user \
 && chown -R 1000:1000 /mnt/user
ENV AGENT_SYSTEM_PATH=/mnt/system
ENV AGENT_SKILL_PATH=/mnt/skills
ENV AGENT_USER_PATH=/mnt/user
ENV AGENT_ENV_PATH=/mnt/system/skill-envs/current
ENV PATH=/mnt/system/skill-envs/current/bin:/mnt/system/bin:${PATH}
ENV VIRTUAL_ENV=/mnt/system/skill-envs/current
# Do NOT put /mnt/skills on PYTHONPATH globally (cross-skill name collisions);
# invoke skill scripts by absolute path.
```

`/mnt/system`, `/mnt/skills`, `/mnt/user` are pre-created as empty dirs only —
actual content comes from dynamic NAS mounts at sandbox start, never baked in.
A bootstrap script (`/usr/local/bin/agent-sandbox-bootstrap`) should validate on
start that the three mounts exist and are readable by uid 1000, and fail fast
with a contract-version marker if a mount is missing.

## Progressive Disclosure (how skills reach the model)

Skills follow the Agent Skills three-level progressive-disclosure model so the
context stays lean until a skill is actually needed. Nothing is gated on a
lexical query match (the earlier `_skill_matches` substring/keyword gate failed
for cross-language queries and for community `SKILL.md`-only skills that carry no
trigger keywords — those skills were effectively invisible to the agent).

- **L1 — catalog (always injected).** `render_skill_catalog` emits a one-line
  `name + description + capability_id` entry for every skill enabled on the
  current agent, into the per-turn `context_block`. This is what makes the agent
  *aware* of its skills. The catalog text instructs the model to call
  `load_skill` before acting. Cost is ~1 line per skill.
- **L2 — full instructions (on demand).** The `load_skill(skill_id)` tool returns
  the skill's complete `SKILL.md` body and, when configured, its exact read-only
  sandbox mount path. It does not enumerate bundled files. The model decides when
  to load instructions based on the L1 catalog; loaded text remains naturally
  sticky in the turn's tool-result history.
- **L3 — bundled resources (on demand, sandbox only).** After loading the skill,
  the model uses `shell` or `code_interpreter` in the exact `/mnt/skills/<id>`
  directory reported by `load_skill`. It starts with `ls` or `find`, then reads or
  runs the required templates, references, and scripts there. The backend exposes
  no separate resource reader and produces no host-side file manifest.

`load_skill` authorizes against `ToolScope.skill_mounts`, the per-request set of
mounts already filtered to skills enabled for the current Agent. Sandbox mounts
use that same Agent-scoped set. Pure instruction skills can work without a
sandbox; skills that require bundled resources have no host-side fallback when
sandbox access is unavailable.

## User Roles

- **Admin**: Can install/update/remove skills, rebuild dependency envs, and
  assign skills to agents.
- **Agent owner/workspace admin**: May enable existing approved skills for an
  agent if product policy allows it.
- **End user**: Can only use ready skills already enabled for the selected
  agent.
