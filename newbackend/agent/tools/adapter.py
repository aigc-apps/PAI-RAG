from __future__ import annotations
from typing import TYPE_CHECKING
from agent.tools.base import Tool

if TYPE_CHECKING:  # pragma: no cover - typing only, not imported at runtime
    from llama_index.core.tools.function_tool import FunctionTool


def tool_from_function_tool(ft: "FunctionTool") -> Tool:
    """Adapt a llama_index FunctionTool into a clean Tool, preserving name,
    description, JSON-schema parameters, return_direct, and async dispatch.

    llama_index is imported lazily (only when this adapter is actually called)
    so importing ``agent.tools`` stays free of the heavy ML stack."""
    meta = ft.metadata
    openai_tool = meta.to_openai_tool(skip_length_check=True)
    params = openai_tool.get("function", {}).get("parameters", {"type": "object", "properties": {}})

    async def _fn(**kwargs) -> str:
        out = await ft.acall(**kwargs)
        return out.content if hasattr(out, "content") else str(out)

    return Tool(
        name=meta.name,
        description=meta.description or meta.name,
        parameters=params,
        fn=_fn,
        return_direct=getattr(meta, "return_direct", False),
    )
