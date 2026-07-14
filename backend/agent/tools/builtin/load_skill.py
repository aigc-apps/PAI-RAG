from __future__ import annotations

from pathlib import Path

from agent.custom_skills import (
    find_enabled_skill_mount,
    load_skill_package,
    render_skill_detail,
)
from agent.tools.base import Tool
from agent.tools.scope import get_current_tool_scope


def _available_ids(mounts) -> str:
    ids = [str(m.get("id")) for m in (mounts or []) if isinstance(m, dict) and m.get("id")]
    return ", ".join(sorted(ids)) if ids else "(none enabled for this agent)"


def make_load_skill_tool() -> Tool:
    """Level-2 progressive disclosure. The agent sees only a one-line summary of
    each skill in the always-present ``# Available Skills`` catalog; when a task
    needs one, it calls ``load_skill`` to pull the full SKILL.md instructions plus
    a manifest of the skill's bundled files. The heavy instruction text is loaded
    on demand instead of injected into every turn, and only skills enabled for the
    current agent (present on the tool scope) are loadable."""

    async def fn(skill_id: str) -> str:
        scope = get_current_tool_scope()
        mount = find_enabled_skill_mount(scope.skill_mounts, skill_id)
        if mount is None:
            return (
                f"load_skill failed: '{skill_id}' is not an available skill for this "
                f"agent. Available skills: {_available_ids(scope.skill_mounts)}."
            )
        source_path = str(mount.get("source_path") or "")
        try:
            package = load_skill_package(Path(source_path)) if source_path else None
        except Exception as ex:  # pragma: no cover - defensive
            return f"load_skill failed: could not read skill '{skill_id}': {ex}"
        if package is None:
            return (
                f"load_skill failed: skill '{skill_id}' is registered but its package "
                f"at {source_path!r} could not be loaded."
            )
        return render_skill_detail(package, mount_path=mount.get("mount_path"))

    return Tool(
        name="load_skill",
        description=(
            "Load the full instructions for one of your available skills (see the "
            "# Available Skills catalog). Call this before performing a task that a "
            "skill covers — the catalog only shows a summary; the real step-by-step "
            "guidance, conventions, and the list of bundled files come from here. "
            "Pass the skill id exactly as shown in the catalog, e.g. "
            "'skill.architecture-diagram'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": (
                        "The skill's id from the # Available Skills catalog, e.g. "
                        "'skill.architecture-diagram' (the 'skill.' prefix is optional)."
                    ),
                },
            },
            "required": ["skill_id"],
        },
        fn=fn,
    )
