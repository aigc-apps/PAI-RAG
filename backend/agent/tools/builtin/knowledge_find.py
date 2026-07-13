"""knowledge_find — exact literal substring search across knowledge-base chunks.

``knowledge_search`` retrieves by *meaning* (vector) and *tokens* (BM25). Neither
reliably matches an exact literal string — a jargon term, an error code, an API
symbol, a part number, or a substring inside a larger token — because the BM25
analyzer segments text. ``knowledge_find`` closes that gap: a case-insensitive
``LIKE '%query%'`` match with no tokenization, so exactly-spelled terms are found.

Backed in-process by :meth:`KnowledgeService.grep_chunks`, permission-scoped like
``knowledge_search`` (greps every accessible KB unless ``kb_ids`` is given). The
caller is derived from the active :class:`ToolScope`.
"""

from __future__ import annotations

from typing import List, Optional

from agent.tools.base import Tool
from agent.tools.builtin.knowledge import _scope_user, _tool_failure
from agent.tools.scope import scope_default_kb_ids


_DEFAULT_LIMIT = 10
# Characters of context to show on each side of a match.
_SNIPPET_PAD = 120


def _snippet(text: str, query: str) -> str:
    """A one-line window centered on the first case-insensitive match of ``query``."""
    text = " ".join((text or "").split())  # collapse whitespace to keep it one line
    idx = text.lower().find(query.lower())
    if idx < 0:  # matched on a different chunk field / whitespace variant — show head
        head = text[: _SNIPPET_PAD * 2]
        return head + (" …" if len(text) > len(head) else "")
    start = max(0, idx - _SNIPPET_PAD)
    end = min(len(text), idx + len(query) + _SNIPPET_PAD)
    body = text[start:end]
    return f"{'… ' if start > 0 else ''}{body}{' …' if end < len(text) else ''}"


def make_knowledge_find_tool(knowledge_service) -> Tool:
    """Build the ``knowledge_find`` tool bound to a live ``KnowledgeService``."""

    async def fn(
        query: str,
        kb_ids: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> str:
        if not query or not query.strip():
            return "knowledge_find requires a non-empty 'query'."
        q = query.strip()
        user = _scope_user()
        try:
            # Explicit kb_ids win; else the agent's soft default; else None =
            # every accessible KB. grep_chunks permission-checks each id per user.
            targets = list(kb_ids) if kb_ids else (scope_default_kb_ids() or None)
            matches = await knowledge_service.grep_chunks(
                user=user,
                query=q,
                kb_ids=targets,
                document_id=document_id,
                limit=max(1, min(int(limit or _DEFAULT_LIMIT), 50)),
            )
        except Exception as ex:  # never surface exception details to the model
            return _tool_failure("knowledge_find", ex)

        if not matches:
            return (
                f'No chunk contains the literal string "{q}". knowledge_find matches exact '
                "text (no tokenization); try knowledge_search for a semantic/keyword "
                "query, or check spelling and casing."
            )
        lines: List[str] = [f'{len(matches)} chunk(s) contain "{q}":', ""]
        for m in matches:
            title = m.get("title") or "(untitled)"
            lines.append(
                f"{title}  ({m.get('document_id')} #{m.get('chunk_index')})"
            )
            if m.get("chunk_id"):
                lines.append(f"    chunk: {m.get('chunk_id')}")
            lines.append(f"    {_snippet(m.get('text') or '', q)}")
            lines.append("")
        lines.append(
            "To read a match in its surrounding document, use "
            "knowledge_read(chunk_id=…, mode=\"locate\")."
        )
        return "\n".join(lines).rstrip()

    return Tool(
        name="knowledge_find",
        description=(
            "Find the exact literal string in knowledge-base documents — a "
            "case-insensitive substring match with no tokenization. Use this for "
            "precise terms that semantic/keyword search may miss: error codes, API "
            "names, identifiers, part numbers, or any exactly-spelled jargon. "
            "Searches every accessible knowledge base unless 'kb_ids' is given; pass "
            "'document_id' to grep within one file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Exact substring to find (case-insensitive, literal — not regex).",
                },
                "kb_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional knowledge base IDs to restrict to. Omit to grep all accessible bases.",
                },
                "document_id": {
                    "type": "string",
                    "description": "Optional document id to grep within a single file.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max matching chunks to return (default {_DEFAULT_LIMIT}, max 50).",
                },
            },
            "required": ["query"],
        },
        fn=fn,
    )
