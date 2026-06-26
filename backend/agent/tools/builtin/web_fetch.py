from __future__ import annotations
import re
from typing import Callable, Optional
import httpx
from agent.tools.base import Tool

_DROP_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " "}


def _html_to_text(html: str, limit: int) -> str:
    html = _DROP_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", html)
    for ent, ch in _ENTITIES.items():
        text = text.replace(ent, ch)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def make_web_fetch_tool(
    client_factory: Optional[Callable[[], object]] = None, limit: int = 8000
) -> Tool:
    """`client_factory()` returns an async-context HTTP client with `.get(url)`
    (default: a real httpx.AsyncClient). Injected as a fake in tests."""

    async def fn(url: str) -> str:
        try:
            client_cm = (
                client_factory()
                if client_factory is not None
                else httpx.AsyncClient(timeout=15, follow_redirects=True)
            )
            async with client_cm as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return _html_to_text(resp.text, limit)
        except Exception as ex:  # never raise into the agent loop
            return f"web_fetch failed: {ex}"

    return Tool(
        name="web_fetch",
        description="Fetch a URL and return its readable text content.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
            "required": ["url"],
        },
        fn=fn,
    )
