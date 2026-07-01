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
  root: ./data/skills
  mount:
    mount_root: /mnt/skills
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

The active agent determines mounts:

```text
agent.skills.enabled -> SkillMount[]
```

For each enabled and ready skill:

```text
/mnt/skills/<skill-id>
```

When an external object store is configured, the backend emits dynamic OSS mount
points during sandbox creation. Empty mount configs are omitted to avoid provider
validation errors.

## User Roles

- **Admin**: Can install/update/remove skills, rebuild dependency envs, and
  assign skills to agents.
- **Agent owner/workspace admin**: May enable existing approved skills for an
  agent if product policy allows it.
- **End user**: Can only use ready skills already enabled for the selected
  agent.
