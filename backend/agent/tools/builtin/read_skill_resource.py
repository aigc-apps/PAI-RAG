from __future__ import annotations

from pathlib import Path

from agent.custom_skills import find_enabled_skill_mount
from agent.tools.base import Tool
from agent.tools.scope import get_current_tool_scope

# Cap the returned text so a large bundled asset can't blow the context window.
_MAX_BYTES = 256 * 1024


def _available_ids(mounts) -> str:
    ids = [str(m.get("id")) for m in (mounts or []) if isinstance(m, dict) and m.get("id")]
    return ", ".join(sorted(ids)) if ids else "(none enabled for this agent)"


def make_read_skill_resource_tool() -> Tool:
    """Level-3 progressive disclosure. Reads one bundled file (template, reference,
    script, ...) from a skill package on demand. Host-side and path-jailed to the
    skill's own directory, so it works without a sandbox or NAS mount — the local
    skill ``source_path`` is never bind-mounted, so a shell ``cat`` of
    /mnt/skills/<id> depends on the NAS skill mount being present; this tool always
    works regardless."""

    async def fn(skill_id: str, path: str) -> str:
        scope = get_current_tool_scope()
        mount = find_enabled_skill_mount(scope.skill_mounts, skill_id)
        if mount is None:
            return (
                f"read_skill_resource failed: '{skill_id}' is not an available skill "
                f"for this agent. Available skills: {_available_ids(scope.skill_mounts)}."
            )
        source_path = str(mount.get("source_path") or "")
        if not source_path:
            return f"read_skill_resource failed: skill '{skill_id}' has no source path."
        base = Path(source_path).resolve()
        try:
            target = (base / path).resolve()
        except Exception as ex:  # pragma: no cover - defensive
            return f"read_skill_resource failed: invalid path {path!r}: {ex}"
        # Path jail: the resolved target must stay inside the skill directory.
        if target != base and base not in target.parents:
            return (
                f"read_skill_resource failed: {path!r} escapes the skill directory. "
                "Only files bundled inside the skill can be read."
            )
        if not target.is_file():
            return f"read_skill_resource failed: {path!r} is not a file in skill '{skill_id}'."
        try:
            data = target.read_bytes()
        except Exception as ex:
            return f"read_skill_resource failed: could not read {path!r}: {ex}"
        truncated = len(data) > _MAX_BYTES
        chunk = data[:_MAX_BYTES]
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            return (
                f"read_skill_resource: {path!r} is a binary file ({len(data)} bytes); "
                "it is not text-readable. Reference it by path instead."
            )
        if truncated:
            text += f"\n\n[...truncated at {_MAX_BYTES} bytes of {len(data)}...]"
        return text

    return Tool(
        name="read_skill_resource",
        description=(
            "Read one bundled file from a skill package (a template, reference doc, "
            "or script listed by load_skill under 'Bundled files'). Use this to pull "
            "in the concrete assets a skill ships with. The path is relative to the "
            "skill's directory, e.g. 'resources/template.html'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "The skill's id, e.g. 'skill.architecture-diagram'.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File path relative to the skill directory, e.g. "
                        "'resources/template.html' (as listed by load_skill)."
                    ),
                },
            },
            "required": ["skill_id", "path"],
        },
        fn=fn,
    )
