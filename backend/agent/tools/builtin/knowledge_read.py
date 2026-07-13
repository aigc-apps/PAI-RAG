"""knowledge_read — read a knowledge-base document (or a single chunk) in full.

``knowledge_search`` returns ranked *snippets*; once a passage looks relevant the
model often needs the surrounding document to ground a complete answer. This tool
takes the short ``document_id`` (or ``chunk_id``) printed by ``knowledge_search``
and returns the file's text, paginated for long documents.

Backed in-process by :meth:`KnowledgeService.fetch_document_or_chunk`, which
resolves the owning KB from the id and applies the same private/workspace/public
permission scoping as the REST path. The caller is derived from the active
:class:`ToolScope`.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from agent.tools.base import Tool
from agent.tools.builtin.knowledge import _scope_user


_DEFAULT_MAX_CHARS = 6000


def make_knowledge_read_tool(knowledge_service) -> Tool:
    """Build the ``knowledge_read`` tool bound to a live ``KnowledgeService``."""

    async def fn(
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        mode: str = "full_doc",
        max_chars: int = _DEFAULT_MAX_CHARS,
        offset: int = 0,
    ) -> str:
        if not (document_id or chunk_id):
            return "knowledge_read requires a 'document_id' or 'chunk_id'."
        user = _scope_user()
        max_chars = max(1, min(int(max_chars or _DEFAULT_MAX_CHARS), 50000))
        offset = max(0, int(offset or 0))
        mode = mode if mode in {"full_doc", "chunk", "chunk_neighbors", "locate"} else "full_doc"
        try:
            result = await knowledge_service.fetch_document_or_chunk(
                user=user,
                document_id=document_id,
                chunk_id=chunk_id,
                mode=mode,
                max_chars=max_chars,
                offset=offset,
            )
        except LookupError as ex:
            return f"knowledge_read: {ex}"
        except PermissionError:
            return "knowledge_read: you do not have access to that document."
        except Exception as ex:  # never surface a raw traceback to the model
            logger.warning(f"knowledge_read failed: {ex!r}")
            return f"knowledge_read failed: {ex}"

        text = result.get("text") or ""
        header = [
            f"{result.get('title') or '(untitled)'}",
            f"document_id: {result.get('document_id')}",
        ]
        if result.get("chunk_id"):
            header.append(f"chunk_id: {result.get('chunk_id')}")
        if result.get("source_uri"):
            header.append(f"source: {result.get('source_uri')}")
        body = [*header, "", text]
        # The service slices to exactly max_chars when there is more to read; hint
        # the model how to page forward rather than silently truncating.
        if len(text) >= max_chars:
            body.append(
                f"\n… truncated at {max_chars} chars. Call knowledge_read again with "
                f"offset={offset + max_chars} to continue."
            )
        return "\n".join(body).rstrip()

    return Tool(
        name="knowledge_read",
        description=(
            "Read a knowledge-base document in full (or a single chunk) by its id. "
            "Use after knowledge_search or knowledge_find surfaces a relevant passage and "
            "you need the surrounding document to answer completely. Pass the "
            "'document_id' (printed by knowledge_search) for the whole file, or a "
            "'chunk_id' for one passage. Long documents are paginated via 'offset'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document id to read in full (e.g. doc_7Kf9Qw2mAbc).",
                },
                "chunk_id": {
                    "type": "string",
                    "description": "Chunk id (e.g. chk_7Kf9Qw2mAbc) to read a single passage, or to locate within the document.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["full_doc", "chunk", "chunk_neighbors", "locate"],
                    "description": (
                        "full_doc (default): whole document. chunk: only the given "
                        "chunk_id. chunk_neighbors: the chunk plus its immediate "
                        "neighbors for context. locate: open the whole document at "
                        "the chunk_id's position (with preceding context) — best "
                        "after knowledge_find/knowledge_search to read a hit in situ."
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "description": f"Max characters to return (default {_DEFAULT_MAX_CHARS}, max 50000).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start from, for paging long documents (default 0).",
                },
            },
            "required": [],
        },
        fn=fn,
    )
