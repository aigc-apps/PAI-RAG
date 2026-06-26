from __future__ import annotations
import json
from typing import Awaitable, Callable, List
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry

# Injected by a live MCP client: call(remote_tool_name, args) -> result (awaitable).
CallFn = Callable[[str, dict], Awaitable[object]]


def mcp_tool_to_tool(spec: dict, call: CallFn, prefix: str = "") -> Tool:
    """Map an MCP tool descriptor ({name, description, inputSchema}) to a Tool.
    The Tool's fn forwards parsed kwargs to the injected MCP `call` and stringifies
    the result. `prefix` namespaces the local name to avoid collisions."""
    remote_name = spec["name"]
    local_name = f"{prefix}{remote_name}" if prefix else remote_name
    parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}

    async def fn(**kwargs) -> str:
        result = await call(remote_name, kwargs)
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    return Tool(
        name=local_name,
        description=spec.get("description", ""),
        parameters=parameters,
        fn=fn,
    )


def register_mcp_tools(
    specs: List[dict], call: CallFn, registry: ToolRegistry, prefix: str = ""
) -> List[str]:
    """Register all tools a (single) MCP server exposes, returning the local names."""
    names: List[str] = []
    for spec in specs:
        tool = mcp_tool_to_tool(spec, call, prefix=prefix)
        registry.register(tool)
        names.append(tool.name)
    return names
