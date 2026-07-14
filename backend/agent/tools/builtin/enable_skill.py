from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, Optional

from agent.tools.base import Tool


def make_enable_skill_for_agent_tool(
    settings,
    agent_config,
    on_config_change: Optional[Callable[[], Any]] = None,
) -> Tool:
    """Admin-only control-plane tool that enables/disables an installed skill for
    an agent by mutating the persisted agent config.

    Persists to ``settings.config_path`` and, when ``on_config_change`` is wired,
    refreshes the running registry + agent config so the change takes effect for
    subsequent requests without a restart. Enable is blocked unless the skill is
    installed and ``ready`` (see ``set_agent_skill_enabled``).
    """
    config_path = str(getattr(settings, "config_path", "") or "")
    default_agent = str(getattr(agent_config, "default_agent", "main") or "main")

    def _apply(skill_id: str, agent_id: str, enabled: bool) -> Dict[str, Any]:
        # Imported lazily to avoid pulling the app layer into tool construction
        # (mirrors search_providers' app.agent_config import at call boundaries).
        from app.agent_config import (
            apply_runtime_status,
            load_agent_config,
            save_agent_config,
            set_agent_skill_enabled,
        )

        doc = load_agent_config(config_path)
        # Validate against a runtime view so freshly discovered skills and their
        # computed statuses are visible; mutate + persist the raw doc.
        runtime = apply_runtime_status(doc, settings, None)
        result = set_agent_skill_enabled(
            doc,
            agent_id=agent_id,
            skill_id=skill_id,
            enabled=enabled,
            installed=runtime.skills.installed,
        )
        save_agent_config(config_path, doc)
        return result

    async def _enable_skill_for_agent(
        skill_id: str,
        agent_id: Optional[str] = None,
        enabled: bool = True,
    ) -> str:
        target_agent = str(agent_id or default_agent)
        result = await asyncio.to_thread(_apply, skill_id, target_agent, bool(enabled))
        if on_config_change is not None:
            outcome = on_config_change()
            if asyncio.iscoroutine(outcome):
                await outcome
            result["runtime_reloaded"] = True
        else:
            result["runtime_reloaded"] = False
        return json.dumps(result, ensure_ascii=False, indent=2)

    return Tool(
        name="enable_skill_for_agent",
        description=(
            "Admin-only control-plane tool. Enable (or disable) an installed, "
            "ready skill for an agent. Enabling a not-ready skill is rejected. "
            "Defaults to the platform default agent when agent_id is omitted."
        ),
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "Skill id, e.g. 'skill.report-writer' or 'report-writer'.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Target agent id; defaults to the platform default agent.",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "True to enable (default), false to disable.",
                },
            },
            "required": ["skill_id"],
            "additionalProperties": False,
        },
        fn=_enable_skill_for_agent,
        permission="admin",
    )
