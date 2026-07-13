"""knowledge_search — the agent-facing online query interface into the KB.

The knowledge base's HTTP query routes (``/v1/knowledge/query/*``) serve the
frontend; this tool gives the *agent* the same retrieval so a chat turn can
answer questions grounded in ingested docs (the ``skill.knowledge_qa`` flow).

Knowledge-base QA is a basic capability, not a skill: registering this tool (the
``knowledge`` capability) is what grounds a chat turn in ingested docs and mounts
the KB guidance into the system prompt — there is no separate skill to enable.

It calls ``KnowledgeService.search`` in-process — no HTTP hop — and derives the
caller from the active :class:`ToolScope` (set per turn by the run loop), so
permission scoping (private vs workspace/public KBs) is identical to the REST
path. With no ``kb_ids`` argument it searches every KB the caller may query,
which is what a general QA turn wants; pass ``kb_ids`` to narrow.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from agent.tools.base import Tool
from agent.tools.scope import get_current_tool_scope, scope_default_kb_ids


# A chunk can be long; the model only needs enough to ground an answer and cite.
_MAX_SNIPPET_CHARS = 700
_DEFAULT_TOP_K = 6


def _scope_user():
    """Build a store ``User`` from the active tool scope for permission checks.

    Imported lazily so the agent package carries no hard dependency on the app
    layer (keeps ``agent`` importable in isolation / other hosts)."""
    from app.store.base import User

    scope = get_current_tool_scope()
    role = "admin" if scope.is_admin else "user"
    return User(id=scope.user_id or "", role=role)


def _format(hits, *, query: str) -> str:
    if not hits:
        return (
            f'No relevant passages found for "{query}". The knowledge base may '
            "not cover this topic, or nothing has been ingested yet."
        )
    lines: List[str] = [f'Top {len(hits)} passages for "{query}":', ""]
    for i, h in enumerate(hits, 1):
        text = (h.text or "").strip()
        if len(text) > _MAX_SNIPPET_CHARS:
            text = text[:_MAX_SNIPPET_CHARS].rstrip() + " …"
        title = h.title or "(untitled)"
        src = h.source_uri or h.document_id
        lines.append(f"[{i}] {title}  (score {h.score:.3f})")
        lines.append(f"    source: {src}")
        # Surface the (short) document id so the model can hand it straight to
        # view_file (read the whole file) or grep_file (find an exact term in it);
        # the chunk id opens this exact passage in situ via view_file locate.
        lines.append(f"    doc: {h.document_id}")
        if getattr(h, "chunk_id", None):
            lines.append(f"    chunk: {h.chunk_id}")
        lines.append(f"    {text}")
        lines.append("")
    lines.append(
        "Cite sources by their [n] / title, and say plainly if the passages do "
        "not contain the answer. Keep each cited document's title and source so you "
        "can list them (title + link, when the source is a URL) in a references "
        "section (\"参考文献\" / \"References\") at the end of your answer. To read a "
        "hit in its surrounding document, use view_file(chunk_id=…, mode=\"locate\")."
    )
    return "\n".join(lines).rstrip()


def make_knowledge_search_tool(knowledge_service) -> Tool:
    """Build the ``knowledge_search`` tool bound to a live ``KnowledgeService``."""

    async def fn(
        query: str,
        kb_ids: Optional[List[str]] = None,
        top_k: int = _DEFAULT_TOP_K,
        mode: str = "hybrid",
    ) -> str:
        if not query or not query.strip():
            return "knowledge_search requires a non-empty 'query'."
        user = _scope_user()
        try:
            # Explicit kb_ids win; else the agent's soft default (if any); else
            # every KB the user can reach. search() re-checks each id per user, so
            # a soft default can only narrow, never leak.
            if kb_ids:
                targets = list(kb_ids)
            else:
                targets = scope_default_kb_ids() or [
                    kb.id for kb in await knowledge_service.list_kbs(user=user)
                ]
            if not targets:
                return (
                    "No knowledge bases are available to search. Ask an admin to "
                    "create one and ingest documents first."
                )
            hits, _total = await knowledge_service.search(
                user=user,
                kb_ids=targets,
                query=query.strip(),
                top_k=max(1, min(int(top_k or _DEFAULT_TOP_K), 20)),
                mode=mode if mode in {"hybrid", "vector", "keyword"} else "hybrid",
            )
            return _format(hits, query=query.strip())
        except Exception as ex:  # never surface a raw traceback to the model
            logger.warning(f"knowledge_search failed: {ex!r}")
            return f"knowledge_search failed: {ex}"

    return Tool(
        name="knowledge_search",
        description=(
            "Search the configured knowledge base(s) for passages relevant to a "
            "question and return ranked snippets with their sources. Use this to "
            "answer questions grounded in ingested documentation before relying on "
            "general knowledge. Searches every knowledge base you can access unless "
            "'kb_ids' is given."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language question or keywords to retrieve on.",
                },
                "kb_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of knowledge base IDs to restrict the search to. Omit to search all accessible bases.",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"Number of passages to return (default {_DEFAULT_TOP_K}, max 20).",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "keyword"],
                    "description": "Retrieval mode (default hybrid).",
                },
            },
            "required": ["query"],
        },
        fn=fn,
    )
