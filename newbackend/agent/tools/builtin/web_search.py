from __future__ import annotations
from typing import Dict, List, Protocol
from agent.tools.base import Tool


class SearchProvider(Protocol):
    async def search(self, query: str, num_results: int) -> List[Dict]: ...


def _format(results: List[Dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.get('title', '(no title)')}\n   {r.get('url', '')}\n   {r.get('snippet', '')}"
        )
    return "\n".join(lines)


def make_web_search_tool(provider: SearchProvider) -> Tool:
    async def fn(query: str, num_results: int = 5) -> str:
        try:
            results = await provider.search(query, num_results)
            return _format(results)
        except Exception as ex:
            return f"web_search failed: {ex}"

    return Tool(
        name="web_search",
        description="Search the web for current information. Returns ranked results with titles, URLs, and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_results": {"type": "integer", "description": "How many results to return (default 5)."},
            },
            "required": ["query"],
        },
        fn=fn,
    )
