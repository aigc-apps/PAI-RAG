# Sandbox-Only Skill Resources Design

## Goal

Establish one unambiguous boundary for skill access:

- the backend reads `SKILL.md` on demand through `load_skill`;
- every other file bundled with a skill is listed, read, or executed only inside the sandbox through its read-only `/mnt/skills/<mount-id>` mount.

The backend must not provide a second host-side resource-reading path.

## Access Model

`load_skill(skill_id)` remains the progressive-disclosure entry point. It authorizes the requested skill against the current Agent's enabled skill mounts, loads the local package metadata and `SKILL.md`, and returns:

- the skill identity;
- the full `SKILL.md` instructions;
- the exact read-only sandbox mount directory when one is configured;
- a concise instruction to use `shell` or `code_interpreter` in that directory for all other files.

`load_skill` does not enumerate bundled files. The Agent discovers resources itself with commands such as `ls` or `find` in the provided mount directory, then uses normal sandbox operations to read or execute them.

## Tool Surface

The `read_skill_resource` built-in tool is removed completely:

- no tool implementation;
- no default-registry registration;
- no builder force-inclusion;
- no shell collision guard entry;
- no prompt or catalog references;
- no frontend localized name or title-summary configuration.

`load_skill` is the only host-side skill-reading tool. Existing installation and Agent enablement tools are unaffected.

## Sandbox Availability

Pure instruction skills continue to work without a sandbox because `load_skill` can still return `SKILL.md`.

Skills that require templates, references, scripts, or other bundled files require `shell` or `code_interpreter` and a working skill mount. There is no host-side fallback when those sandbox capabilities are unavailable. The Agent should report that the required sandbox resource access is unavailable rather than attempting another tool.

## Rendering Contract

The skill detail renderer no longer calls the bundled-file scanner and no longer emits a resource manifest.

When a mount path is present, it appends guidance equivalent to:

> Skill files are mounted read-only at `/mnt/skills/<mount-id>`. Use shell or code_interpreter in that exact directory to list, read, or run bundled resources.

When no mount path is present, it returns the skill instructions without inventing a sandbox location or suggesting a host-side resource tool.

## Frontend Impact

The frontend tool-call display policy removes `read_skill_resource` and its Chinese and English display-name keys. All remaining built-in tools retain their existing localized title behavior.

The built-in coverage audit changes from 16 tools to 15 after the backend tool is removed.

## Documentation

Current architecture documentation must describe sandbox-only bundled resources. Historical Superpowers design and plan documents remain unchanged as implementation history, but active design documents and inline comments must not recommend `read_skill_resource`.

## Testing

Backend tests cover:

- `load_skill` returns full `SKILL.md` instructions and the exact mount path;
- the returned detail contains sandbox `ls`/read guidance and no bundled-file manifest;
- the returned detail never mentions `read_skill_resource`;
- a missing mount path does not invent `/mnt/skills` or a fallback tool;
- the default registry and Agent builder include `load_skill` but not `read_skill_resource`;
- the shell collision guard no longer treats `read_skill_resource` as an Agent tool;
- a repository scan finds no executable or active-document references to `read_skill_resource`.

Frontend tests cover:

- the display policy no longer recognizes `read_skill_resource` as a built-in tool;
- the English and Chinese dictionaries no longer contain its display-name key;
- every remaining backend built-in tool still has a display policy.

Run the complete backend and frontend test suites relevant to changed modules, plus the frontend production build.

## Out of Scope

- Changing skill installation, enablement, authorization, or NAS mount construction.
- Moving `SKILL.md` loading into the sandbox.
- Automatically invoking `ls` after `load_skill`.
- Adding a new resource proxy or fallback API.
- Changing the behavior of `shell` or `code_interpreter` beyond removing the obsolete collision-guard name.
