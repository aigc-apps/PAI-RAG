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
        # Optional read-only code layer at /mnt/code (repos are subdirs).
        # Set code_server_addr to enable; empty => layer off.
        code_server_addr: xxxx.nas.aliyuncs.com:/
        code_remote_path: /code
        code_read_only: true
      inject_env_contract: true
      extra_envs: {}
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

Four runtime paths are contracted inside the sandbox (the code layer is optional
— present only when `code_server_addr` is configured):

```text
/mnt/system   # agent-level shared, read-only; heavy deps baked into the sandbox image
/mnt/skills   # agent-level shared, read-only skill packages
/mnt/user     # per-user isolated, writable (outputs, memory)
/mnt/code     # agent-level shared, read-only source repos (optional; explore when the KB misses)
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

And, when configured, one shared read-only code layer (a single export whose
subdirectories are repositories; the agent discovers them by `ls /mnt/code`):

```text
/mnt/code  <- nas: <code_server>:/code  (read-only)
```

All mount points share `userId/groupId = 1000` (the platform default for NAS
mounts) at the `nasConfig` top level. `mountDir` values must not collide; skill
mounts are leaf dirs under `/mnt/skills/<id>`, the user mount is `/mnt/user`,
and the code layer is `/mnt/code`, so they never overlap. Empty mount configs
are omitted from the create payload to avoid provider validation errors.

The env-var contract exposes each mounted path: `AGENT_SYSTEM_PATH`,
`AGENT_SKILL_PATH`, `AGENT_USER_PATH`, and (only when the code layer is
configured) `AGENT_CODE_PATH=/mnt/code`. Deployment side: the sandbox image must
`mkdir /mnt/code`, and the read-only code NAS export must be mounted there.

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

The sandbox image is an AgentRun template. The build recipe lives in this repo
at `sandbox/` (`Dockerfile` + `agent-sandbox-bootstrap` + `README.md`); the
image itself is built and registered in AgentRun outside the agent service. It
must fulfill the runtime contract so the `AGENT_*` vars resolve:

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
  the skill's complete `SKILL.md` body plus a manifest of its bundled files. The
  model decides when to load, based on the L1 catalog — far more reliable than
  the host guessing by substring. Loaded text lives in the tool-result history,
  so it is naturally sticky across the turn's tool loop without being re-injected
  every turn.
- **L3 — bundled resources (on demand).** The `read_skill_resource(skill_id,
  path)` tool reads one bundled file (template, reference, script) from the skill
  directory. It is **host-side and path-jailed** to the skill's `source_path`, so
  it works with no sandbox and no NAS — important because the local `source_path`
  is never bind-mounted (only NAS-backed mounts with a `serverAddr` bind at
  `/mnt/skills/<id>`), so a sandbox `cat /mnt/skills/...` only works on a NAS
  deployment while `read_skill_resource` always works.

Both `load_skill` and `read_skill_resource` are read-only and authorize against
`ToolScope.skill_mounts` — the per-request set of mounts already filtered to the
skills enabled for this agent. A request for a skill not on that list is refused
with the list of available ids. They are registered whenever a skill source is
configured (`skill_sources(agent_config.skills)` is non-empty).

## User Roles

- **Admin**: Can install/update/remove skills, rebuild dependency envs, and
  assign skills to agents.
- **Agent owner/workspace admin**: May enable existing approved skills for an
  agent if product policy allows it.
- **End user**: Can only use ready skills already enabled for the selected
  agent.
