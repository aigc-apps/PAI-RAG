"""General per-KB agent tools for browsing, searching and reading files.

Three tools, all spanning the **whole** knowledge base (manual uploads included),
complementing the semantic ``aget_knowledgebase_tool``:

- ``catalog`` — list files (name, title, source) by metadata; no body read.
- ``grep`` — exact/literal keyword lookup over file bodies, with line numbers
  and surrounding context.
- ``fetch`` — return a file's full text (or a coarse fallback) by ``file_id`` /
  ``doc_id``, reading the stored content from the file store.
"""

import json
from typing import Annotated, Optional, TYPE_CHECKING

from llama_index.core.tools import FunctionTool
from sqlmodel import select
from loguru import logger

from db.models.knowledgebase.datasource import DataSourceDocumentEntity

if TYPE_CHECKING:
    from service.knowledgebase.rag_service import RagService

# Hard caps so a caller-supplied argument can't blow the agent's context window.
# DEFAULT_FETCH_MAX_CHARS is both the default and the ceiling for `fetch`; the
# agent pages with offset/next_offset, or narrows with grep, to read more.
DEFAULT_FETCH_MAX_CHARS = 6000
MAX_CATALOG_LIMIT = 200
MAX_GREP_LIMIT = 200
MAX_GREP_CONTEXT = 10


async def aget_kb_catalog_tool(
    kb_id: str,
    tenant_id: str,
    rag_service: "RagService" = None,
):
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    description = (
        f"List files in knowledge base '{knowledgebase.name}' — file name, title, and "
        f"source — without reading any body.\n"
        f"\n# When to use\n"
        f"To answer \"which files exist / is there a file about X\", to see what's available, "
        f"or to narrow down before `grep`/`fetch`. Returns file names + ids (not content).\n"
        f"\n# Parameters\n"
        f"- query (optional): case-insensitive substring match over file name / title "
        f"(not fuzzy — the literal substring must appear); omit to just browse.\n"
        f"- limit (optional): max results (default 20)."
    )

    async def kb_catalog_handler(
        query: Annotated[str, "Case-insensitive substring match over file name / title; omit to browse."] = "",
        limit: Annotated[int, "Max results (default 20, hard cap 200)."] = 20,
    ) -> str:
        try:
            limit = min(max(1, limit), MAX_CATALOG_LIMIT)
            page_result = await rag_service.list_files(
                kb_id=kb_id, tenant_id=tenant_id, page=1, size=limit, query=query or None
            )
            items = page_result.items or []
            results = []
            for f in items:
                md = f.file_metadata or {}
                results.append({
                    "file_id": f.id,
                    "doc_id": md.get("source_doc_id"),
                    "title": md.get("title") or f.file_name,
                    "file_name": f.file_name,
                    "source_url": f.file_source or md.get("source_url"),
                    "status": f.status,
                })
            return json.dumps(
                {"ok": True, "results": results, "total": page_result.total},
                ensure_ascii=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"kb catalog failed: {e}")
            return json.dumps({"ok": False, "error": "catalog_failed", "message": str(e)}, ensure_ascii=False)

    return FunctionTool.from_defaults(
        async_fn=kb_catalog_handler,
        name=f"catalog-{kb_id[:8]}",
        description=description,
        return_direct=False,
    )


async def aget_kb_grep_tool(
    kb_id: str,
    tenant_id: str,
    rag_service: "RagService" = None,
):
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    description = (
        f"Exact (literal) keyword/identifier lookup across the files of knowledge base "
        f"'{knowledgebase.name}'. Returns matching lines with line numbers + surrounding "
        f"context.\n"
        f"\n# When to use\n"
        f"For exact tokens that semantic search misses — error codes (e.g. 137), config keys "
        f"(e.g. eventTime), function names, flags. Not a regex; matches the literal string.\n"
        f"\n# Parameters\n"
        f"- pattern (required): the literal string to find.\n"
        f"- context (optional): lines of context around each match (default 2).\n"
        f"- limit (optional): max matches (default 20)."
    )

    async def kb_grep_handler(
        pattern: Annotated[str, "Literal string to find (not a regex)."] = "",
        context: Annotated[int, "Lines of context around each match (default 2, hard cap 10)."] = 2,
        limit: Annotated[int, "Max matches (default 20, hard cap 200)."] = 20,
    ) -> str:
        try:
            if not pattern:
                return json.dumps({"ok": True, "results": [], "total": 0}, ensure_ascii=False)
            context = min(max(0, context), MAX_GREP_CONTEXT)
            limit = min(max(1, limit), MAX_GREP_LIMIT)
            out = await rag_service.keyword_search(
                kb_id=kb_id, tenant_id=tenant_id, pattern=pattern,
                context=context, limit=limit,
            )
            results = out.get("results", [])
            return json.dumps({
                "ok": True,
                "degraded": "scan_capped" if out.get("scan_capped") else None,
                "results": results,
                "total": len(results),
            }, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"kb grep failed: {e}")
            return json.dumps({"ok": False, "error": "grep_failed", "message": str(e)}, ensure_ascii=False)

    return FunctionTool.from_defaults(
        async_fn=kb_grep_handler,
        name=f"grep-{kb_id[:8]}",
        description=description,
        return_direct=False,
    )


async def _resolve_file_id(session, kb_id, tenant_id, doc_id, file_id) -> Optional[str]:
    if file_id:
        return file_id
    if not doc_id:
        return None
    # 1) treat doc_id as a data-source manifest doc_id ("{datasource_key}/{path}")
    result = await session.exec(
        select(DataSourceDocumentEntity.file_id).where(
            DataSourceDocumentEntity.kb_id == kb_id,
            DataSourceDocumentEntity.doc_id == doc_id,
            DataSourceDocumentEntity.tenant_id == tenant_id,
        )
    )
    fid = result.first()
    if fid:
        return fid
    # 2) fall back: results expose doc_id == file_id when source_doc_id is absent,
    #    so the model may pass a file_id in the doc_id slot. Accept that.
    from db.models.knowledgebase.file import KbFileEntity
    result2 = await session.exec(
        select(KbFileEntity.id).where(
            KbFileEntity.id == doc_id,
            KbFileEntity.kb_id == kb_id,
            KbFileEntity.tenant_id == tenant_id,
        )
    )
    return result2.first()


async def aget_kb_fetch_tool(
    kb_id: str,
    tenant_id: str,
    rag_service: "RagService" = None,
):
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    description = (
        f"Fetch the text of a file from knowledge base '{knowledgebase.name}'.\n"
        f"\n# When to use\n"
        f"After `grep` or a semantic search, when the returned snippets are too sparse and "
        f"you need more of the file to answer accurately. Pass the `file_id` (or `doc_id`) "
        f"from a result.\n"
        f"\n# Long files\n"
        f"Output is capped at ~{DEFAULT_FETCH_MAX_CHARS} characters to protect context. If the "
        f"response has `truncated: true`, either call again with `offset = next_offset` to page "
        f"forward, or (better) use `grep` to jump to the relevant part instead of reading the "
        f"whole file.\n"
        f"\n# Parameters\n"
        f"- file_id OR doc_id (one required): identifies the file.\n"
        f"- offset (optional, default 0): start character for paging.\n"
        f"- max_chars (optional): characters to return (default ~{DEFAULT_FETCH_MAX_CHARS})."
    )

    async def kb_fetch_handler(
        file_id: Annotated[Optional[str], "KB file id from a result."] = None,
        doc_id: Annotated[Optional[str], "Document id from a result (alternative to file_id)."] = None,
        offset: Annotated[int, "Start character offset for paging long files (default 0)."] = 0,
        max_chars: Annotated[int, f"Max characters to return; default and hard cap {DEFAULT_FETCH_MAX_CHARS} (page with offset for more)."] = DEFAULT_FETCH_MAX_CHARS,
    ) -> str:
        try:
            session = rag_service.session
            resolved_file_id = await _resolve_file_id(session, kb_id, tenant_id, doc_id, file_id)
            if not resolved_file_id:
                return json.dumps(
                    {"ok": False, "error": "not_found", "message": "File not found for the given ref."},
                    ensure_ascii=False,
                )

            file_service = await rag_service._get_file_service()
            entity = await file_service.get_file(kb_id=kb_id, file_id=resolved_file_id, tenant_id=tenant_id)
            if not entity:
                return json.dumps(
                    {"ok": False, "error": "not_found", "message": "KB file not found."},
                    ensure_ascii=False,
                )

            degraded = None
            content = await _read_full_doc(entity, tenant_id)
            if content is None:
                # fallback: reassemble from chunks
                content = await _reassemble_from_chunks(rag_service, kb_id, resolved_file_id, tenant_id)
                if content is not None:
                    degraded = "full_doc_from_chunks"

            if content is None:
                return json.dumps(
                    {"ok": False, "error": "empty", "message": "No content available for this file."},
                    ensure_ascii=False,
                )

            # Cap output to protect the agent's context; expose paging info.
            # max_chars may only request *less* than the ceiling — a large value
            # cannot widen the window past DEFAULT_FETCH_MAX_CHARS. To read more,
            # the agent pages via offset/next_offset.
            total = len(content)
            start = max(0, offset or 0)
            limit = DEFAULT_FETCH_MAX_CHARS
            if max_chars and max_chars > 0:
                limit = min(max_chars, DEFAULT_FETCH_MAX_CHARS)
            windowed = content[start:start + limit]
            truncated = (start + len(windowed)) < total

            md = entity.file_metadata or {}
            return json.dumps({
                "ok": True,
                "degraded": degraded,
                "doc_id": md.get("source_doc_id") or resolved_file_id,
                "file_id": resolved_file_id,
                "title": md.get("title") or entity.file_name,
                "file_name": entity.file_name,
                "content": windowed,
                "content_length": total,
                "offset": start,
                "returned_chars": len(windowed),
                "truncated": truncated,
                "next_offset": (start + len(windowed)) if truncated else None,
                "source_url": entity.file_source or md.get("source_url"),
            }, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"kb fetch failed: {e}")
            return json.dumps({"ok": False, "error": "fetch_failed", "message": str(e)}, ensure_ascii=False)

    return FunctionTool.from_defaults(
        async_fn=kb_fetch_handler,
        name=f"fetch-{kb_id[:8]}",
        description=description,
        return_direct=False,
    )


async def _read_full_doc(entity, tenant_id: str) -> Optional[str]:
    """Read the original content from the file store; None if unavailable."""
    if not entity.file_path:
        return None
    try:
        from pairag.file.store.file_store_helper import file_store
        stream = await file_store.read_async(file_path=entity.file_path, tenant_id=tenant_id)
        if stream is None:
            return None
        raw = stream.read() if hasattr(stream, "read") else stream
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"file_store read failed for {entity.file_path}: {e}")
        return None


async def _reassemble_from_chunks(rag_service, kb_id, file_id, tenant_id) -> Optional[str]:
    chunk_service = await rag_service._get_chunk_service()
    chunks = await chunk_service.get_chunks_by_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
    if not chunks:
        return None
    ordered = sorted(chunks, key=lambda c: c.index)
    return "\n\n".join(c.text for c in ordered if c.text)
