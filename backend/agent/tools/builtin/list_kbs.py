"""list_knowledge_bases — enumerate the knowledge bases the caller can search.

``knowledge_search`` and ``grep_file`` already query *every* accessible KB when no
``kb_ids`` is given, and accept a list to search several specific ones at once. But
the model can't *see* what bases exist to decide whether to scope a query (e.g. "only
the HR-policies base"). This tool lists them — id, name, description, and size — so
the model can then pass the right ``kb_ids`` to the search tools.

Backed in-process by :meth:`KnowledgeService.list_kbs`, permission-scoped to the
active :class:`ToolScope` (private bases only for their owner; workspace/public for
everyone; admins see all).
"""

from __future__ import annotations

from loguru import logger

from agent.tools.base import Tool
from agent.tools.builtin.knowledge import _scope_user


def make_list_kbs_tool(knowledge_service) -> Tool:
    """Build the ``list_knowledge_bases`` tool bound to a live ``KnowledgeService``."""

    async def fn() -> str:
        user = _scope_user()
        try:
            kbs = await knowledge_service.list_kbs(user=user)
        except Exception as ex:  # never surface a raw traceback to the model
            logger.warning(f"list_knowledge_bases failed: {ex!r}")
            return f"list_knowledge_bases failed: {ex}"

        if not kbs:
            return (
                "No knowledge bases are available to you. Ask an admin to create one "
                "and ingest documents before using knowledge_search or grep_file."
            )
        lines = [f"{len(kbs)} knowledge base(s) you can search:", ""]
        for kb in kbs:
            docs = getattr(kb, "document_count", 0) or 0
            chunks = getattr(kb, "chunk_count", 0) or 0
            tag = " (empty)" if docs == 0 else ""
            lines.append(
                f"{kb.name}  ({kb.id})  [{kb.visibility}]  · {docs} docs / {chunks} chunks{tag}"
            )
            if (kb.description or "").strip():
                lines.append(f"    {kb.description.strip()}")
        lines.append("")
        lines.append(
            "Pass one or more ids above as 'kb_ids' to narrow knowledge_search or "
            "grep_file to specific bases."
        )
        return "\n".join(lines)

    return Tool(
        name="list_knowledge_bases",
        description=(
            "List the knowledge bases you can search, with their ids, names, "
            "descriptions, and sizes. Call this to discover what documentation is "
            "available and to get the 'kb_ids' for scoping knowledge_search or "
            "grep_file to specific bases. (Both tools already search every accessible "
            "base at once when kb_ids is omitted, so this is only needed to narrow.)"
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        fn=fn,
    )
