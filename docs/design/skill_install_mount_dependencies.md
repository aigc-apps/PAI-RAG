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
  `web_search`, `knowledge_search`, and `code_sandbox`.
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

The Settings UI uses two control-plane HTTP endpoints:

- `POST /v1/skills/uploads`: Admin-only multipart upload for a skill zip. Returns
  an `upload_id`.
- `POST /v1/skills/install`: Admin-only install action. Accepts
  `zip_upload`, `url`, or `git` sources and refreshes the runtime config after
  install.

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
- Dependency build execution is reserved for `rebuild_skill_env`.
- Runtime install remains disabled.

## Sandbox Mounting

Three runtime paths are contracted inside the sandbox:

```text
/mnt/system   # agent-level shared, read-only; heavy deps baked into the sandbox image
/mnt/skills   # agent-level shared, read-only skill packages
/mnt/user     # per-user isolated, writable (outputs, memory)
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

All mount points share `userId/groupId = 1000` (the platform default for NAS
mounts) at the `nasConfig` top level. `mountDir` values must not collide; skill
mounts are leaf dirs under `/mnt/skills/<id>` and the user mount is `/mnt/user`,
so they never overlap. Empty mount configs are omitted from the create payload
to avoid provider validation errors.

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

## User Roles

- **Admin**: Can install/update/remove skills, rebuild dependency envs, and
  assign skills to agents.
- **Agent owner/workspace admin**: May enable existing approved skills for an
  agent if product policy allows it.
- **End user**: Can only use ready skills already enabled for the selected
  agent.
