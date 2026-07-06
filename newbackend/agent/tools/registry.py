from __future__ import annotations
from typing import Dict, List, Optional
from loguru import logger
from agent.tools.base import Tool, ToolBox


class ToolRegistry:
    """Named collection of Tools; builds the ToolBox the agent runs with.
    Skills and MCP servers register into the same registry, so tool selection
    (soul.tools_enabled) is uniform regardless of where a tool came from."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        # Set by build_default_registry when a sandbox provider is configured, so
        # the /v1/files endpoint can reuse the same warm session for readback.
        self.sandbox_provider: Optional[object] = None

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"tool '{tool.name}' re-registered; overwriting")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def build_toolbox(self, names: Optional[List[str]] = None) -> ToolBox:
        if names is None:
            return ToolBox(list(self._tools.values()))
        out: List[Tool] = []
        for n in names:
            tool = self._tools.get(n)
            if tool is None:
                logger.warning(f"tool '{n}' requested but not registered; skipping")
            else:
                out.append(tool)
        return ToolBox(out)
