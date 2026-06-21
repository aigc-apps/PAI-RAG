"""Agent access-protocol tools for data-source-backed knowledge bases.

Two orthogonal tools (spec §5), both per-KB and spanning the data sources bound
to that KB:

- ``search`` — semantic + keyword hybrid recall over the data sources, with
  optional ``datasource`` / ``product`` / ``section`` / ``lang`` filters. Reuses
  ``RagService.aquery`` and maps the result into the spec's chunk shape.
- ``fetch`` — return a full document (or a coarse fallback) by ``doc_id`` /
  ``file_id``, reading the stored markdown from the file store.
"""

import json
from typing import Annotated, Optional, TYPE_CHECKING

from llama_index.core.tools import FunctionTool
from sqlmodel import select
from loguru import logger

from common.chat.models import (
    MetadataFilteringCondition,
    Condition,
    RetrievalSetting,
)
from db.models.knowledgebase.datasource import DataSourceDocumentEntity

if TYPE_CHECKING:
    from service.knowledgebase.rag_service import RagService

# Cap fetch output so a large document can't blow the agent's context window.
# The agent pages with offset/next_offset, or narrows with search/keyword.
DEFAULT_FETCH_MAX_CHARS = 6000

# filter name (tool arg) -> chunk metadata key
_FILTER_FIELDS = {
    "datasource": "datasource_key",
    "product": "product",
    "section": "section",
    "lang": "lang",
}


def _build_filter_condition(filters: dict) -> Optional[MetadataFilteringCondition]:
    # Always scope to data-source-origin chunks (they carry datasource_key);
    # manually-uploaded files have no datasource_key and are excluded. This keeps
    # the datasource `search` tool consistent with "data-source documents".
    conditions = [Condition(name="datasource_key", comparison_operator="not empty")]
    for arg, value in filters.items():
        if value is None or value == "":
            continue
        meta_key = _FILTER_FIELDS.get(arg)
        if not meta_key:
            continue
        conditions.append(Condition(name=meta_key, comparison_operator="is", value=value))
    return MetadataFilteringCondition(logical_operator="and", conditions=conditions)


def _to_spec_results(records) -> list:
    results = []
    for r in records:
        md = r.metadata or {}
        results.append({
            "chunk_id": r.id,
            "doc_id": md.get("source_doc_id") or md.get("doc_id"),
            "file_id": md.get("doc_id"),  # KB file id (== chunk metadata doc_id), used by fetch
            "score": r.score,
            "text": r.content,
            "heading_path": md.get("heading_path"),
            "metadata": {
                "title": r.title,
                "datasource": md.get("datasource_key"),
                "product": md.get("product"),
                "section": md.get("section"),
                "lang": md.get("lang"),
                "source_url": r.url or md.get("file_source") or md.get("source_url"),
            },
        })
    return results


async def aget_datasource_search_tool(
    kb_id: str,
    tenant_id: str,
    user_id: Optional[str] = None,
    rag_service: "RagService" = None,
):
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    description = (
        f"Semantic + keyword search over the external documentation data sources bound to "
        f"knowledge base '{knowledgebase.name}'.\n"
        f"\n# When to use\n"
        f"Use for fuzzy/intent questions about the documentation (\"how does X work\", "
        f"\"what is Y\"). Returns the top matching chunks, each with chunk_id / doc_id / "
        f"source_url so you can cite or fetch the full document.\n"
        f"\n# Parameters\n"
        f"- query (required): a clear, standalone search query.\n"
        f"- datasource/product/section/lang (optional): narrow the search to one source, "
        f"product, section or language.\n"
        f"- k (optional): number of chunks to return (default 8)."
    )

    async def datasource_search_handler(
        query: Annotated[str, "A clear, standalone natural-language search query."] = "",
        datasource: Annotated[Optional[str], "Optional: restrict to one data source key."] = None,
        product: Annotated[Optional[str], "Optional: restrict to one product."] = None,
        section: Annotated[Optional[str], "Optional: restrict to one section."] = None,
        lang: Annotated[Optional[str], "Optional: restrict to a language (e.g. zh / en)."] = None,
        k: Annotated[int, "Number of chunks to return (default 8)."] = 8,
    ) -> str:
        try:
            if not query:
                return json.dumps({"ok": True, "degraded": None, "results": [], "total": 0}, ensure_ascii=False)
            cond = _build_filter_condition(
                {"datasource": datasource, "product": product, "section": section, "lang": lang}
            )
            retrieval_setting = RetrievalSetting(top_k=k)
            records = await rag_service.aquery(
                query=query, kb_id=kb_id, user_id=user_id, tenant_id=tenant_id,
                retrieval_setting=retrieval_setting, metadata_condition=cond,
            )
            results = _to_spec_results(records)
            return json.dumps(
                {"ok": True, "degraded": None, "results": results, "total": len(results)},
                ensure_ascii=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"datasource search failed: {e}")
            return json.dumps({"ok": False, "error": "search_failed", "message": str(e)}, ensure_ascii=False)

    return FunctionTool.from_defaults(
        async_fn=datasource_search_handler,
        name=f"datasource-search-{kb_id[:8]}",
        description=description,
        return_direct=False,
    )


async def aget_datasource_catalog_tool(
    kb_id: str,
    tenant_id: str,
    rag_service: "RagService" = None,
):
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    description = (
        f"Browse / locate documents by metadata in the data sources of knowledge base "
        f"'{knowledgebase.name}' — no document body is read.\n"
        f"\n# When to use\n"
        f"To answer \"is there a doc about X / which docs exist\", to list what's available, or "
        f"to narrow down before `search`/`fetch`. Returns document titles + ids (not content).\n"
        f"\n# Parameters\n"
        f"- query (optional): fuzzy match over title / path / summary; omit to just browse.\n"
        f"- product / section / lang (optional): structured filters.\n"
        f"- limit (optional): max results (default 20)."
    )

    async def datasource_catalog_handler(
        query: Annotated[str, "Fuzzy match over title/path/summary; omit to browse."] = "",
        product: Annotated[Optional[str], "Optional product filter."] = None,
        section: Annotated[Optional[str], "Optional section filter."] = None,
        lang: Annotated[Optional[str], "Optional language filter (e.g. zh / en)."] = None,
        limit: Annotated[int, "Max results (default 20)."] = 20,
    ) -> str:
        try:
            from service.knowledgebase.datasource_service import DataSourceService
            svc = DataSourceService(rag_service.session)
            results = await svc.catalog_search(
                kb_id=kb_id, tenant_id=tenant_id, query=query or None,
                product=product, section=section, lang=lang, limit=limit,
            )
            return json.dumps({"ok": True, "results": results, "total": len(results)}, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"datasource catalog failed: {e}")
            return json.dumps({"ok": False, "error": "catalog_failed", "message": str(e)}, ensure_ascii=False)

    return FunctionTool.from_defaults(
        async_fn=datasource_catalog_handler,
        name=f"catalog-{kb_id[:8]}",
        description=description,
        return_direct=False,
    )


async def aget_datasource_keyword_tool(
    kb_id: str,
    tenant_id: str,
    rag_service: "RagService" = None,
):
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    description = (
        f"Exact (literal) keyword/identifier lookup across the data-source documents of "
        f"knowledge base '{knowledgebase.name}'. Returns matching lines with line numbers + "
        f"surrounding context.\n"
        f"\n# When to use\n"
        f"For exact tokens that semantic search misses — error codes (e.g. 137), config keys "
        f"(e.g. eventTime), function names, flags. Not a regex; matches the literal string.\n"
        f"\n# Parameters\n"
        f"- pattern (required): the literal string to find.\n"
        f"- datasource / path_prefix / doc_id (optional): narrow the scope (recommended).\n"
        f"- context (optional): lines of context around each match (default 2).\n"
        f"- limit (optional): max matches (default 20)."
    )

    async def datasource_keyword_handler(
        pattern: Annotated[str, "Literal string to find (not a regex)."] = "",
        datasource: Annotated[Optional[str], "Optional: restrict to one data source key."] = None,
        path_prefix: Annotated[Optional[str], "Optional: restrict to a path prefix."] = None,
        doc_id: Annotated[Optional[str], "Optional: restrict to a single document."] = None,
        context: Annotated[int, "Lines of context around each match (default 2)."] = 2,
        limit: Annotated[int, "Max matches (default 20)."] = 20,
    ) -> str:
        try:
            if not pattern:
                return json.dumps({"ok": True, "results": [], "total": 0}, ensure_ascii=False)
            out = await rag_service.keyword_search(
                kb_id=kb_id, tenant_id=tenant_id, pattern=pattern,
                doc_id=doc_id, path_prefix=path_prefix, datasource=datasource,
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
            logger.exception(f"datasource keyword failed: {e}")
            return json.dumps({"ok": False, "error": "keyword_failed", "message": str(e)}, ensure_ascii=False)

    return FunctionTool.from_defaults(
        async_fn=datasource_keyword_handler,
        name=f"keyword-{kb_id[:8]}",
        description=description,
        return_direct=False,
    )


async def _resolve_file_id(session, kb_id, tenant_id, doc_id, file_id) -> Optional[str]:
    if file_id:
        return file_id
    if not doc_id:
        return None
    # 1) treat doc_id as the manifest doc_id ("{datasource_key}/{path}")
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
    # 2) fall back: search results expose doc_id == file_id when source_doc_id is
    #    absent, so the model may pass a file_id in the doc_id slot. Accept that.
    from db.models.knowledgebase.file import KbFileEntity
    result2 = await session.exec(
        select(KbFileEntity.id).where(
            KbFileEntity.id == doc_id,
            KbFileEntity.kb_id == kb_id,
            KbFileEntity.tenant_id == tenant_id,
        )
    )
    return result2.first()


async def aget_datasource_fetch_tool(
    kb_id: str,
    tenant_id: str,
    rag_service: "RagService" = None,
):
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    description = (
        f"Fetch the text of a document from the data sources bound to knowledge base "
        f"'{knowledgebase.name}'.\n"
        f"\n# When to use\n"
        f"After `search`, when the returned chunks are too sparse and you need more of the "
        f"document to answer accurately. Pass the `doc_id` (or `file_id`) from a search result.\n"
        f"\n# Long documents\n"
        f"Output is capped at ~{DEFAULT_FETCH_MAX_CHARS} characters to protect context. If the "
        f"response has `truncated: true`, either call again with `offset = next_offset` to page "
        f"forward, or (better) use `search`/`keyword` to jump to the relevant part instead of "
        f"reading the whole document.\n"
        f"\n# Parameters\n"
        f"- doc_id OR file_id (one required): identifies the document.\n"
        f"- offset (optional, default 0): start character for paging.\n"
        f"- max_chars (optional): characters to return (default ~{DEFAULT_FETCH_MAX_CHARS}).\n"
        f"- mode (optional): 'full_doc' (default). 'section'/'chunk_neighbors' are coarse for "
        f"now and fall back to the full document."
    )

    async def datasource_fetch_handler(
        doc_id: Annotated[Optional[str], "Document id from a search result (datasource/path)."] = None,
        file_id: Annotated[Optional[str], "KB file id from a search result (alternative to doc_id)."] = None,
        mode: Annotated[str, "full_doc (default) / section / chunk_neighbors."] = "full_doc",
        offset: Annotated[int, "Start character offset for paging long docs (default 0)."] = 0,
        max_chars: Annotated[int, f"Max characters to return (default {DEFAULT_FETCH_MAX_CHARS})."] = DEFAULT_FETCH_MAX_CHARS,
    ) -> str:
        try:
            session = rag_service.session
            resolved_file_id = await _resolve_file_id(session, kb_id, tenant_id, doc_id, file_id)
            if not resolved_file_id:
                return json.dumps(
                    {"ok": False, "error": "not_found", "message": "Document not found for the given ref."},
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
            content = None
            if mode == "full_doc":
                content = await _read_full_doc(entity, tenant_id)
            if content is None:
                # fallback: reassemble from chunks (also covers section/chunk_neighbors for MVP)
                content = await _reassemble_from_chunks(rag_service, kb_id, resolved_file_id, tenant_id)
                if mode != "full_doc":
                    degraded = f"{mode}_unsupported_returned_full_doc"
                elif content is not None:
                    degraded = "full_doc_from_chunks"

            if content is None:
                return json.dumps(
                    {"ok": False, "error": "empty", "message": "No content available for this document."},
                    ensure_ascii=False,
                )

            # Cap output to protect the agent's context; expose paging info.
            total = len(content)
            start = max(0, offset or 0)
            limit = max_chars if (max_chars and max_chars > 0) else DEFAULT_FETCH_MAX_CHARS
            windowed = content[start:start + limit]
            truncated = (start + len(windowed)) < total

            md = entity.file_metadata or {}
            return json.dumps({
                "ok": True,
                "degraded": degraded,
                "doc_id": md.get("source_doc_id") or resolved_file_id,
                "file_id": resolved_file_id,
                "title": entity.file_name,
                "content": windowed,
                "mode": mode,
                "content_length": total,
                "offset": start,
                "returned_chars": len(windowed),
                "truncated": truncated,
                "next_offset": (start + len(windowed)) if truncated else None,
                "source_url": entity.file_source,
                "metadata": {
                    "datasource": md.get("datasource_key"),
                    "product": md.get("product"),
                    "section": md.get("section"),
                    "lang": md.get("lang"),
                },
            }, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"datasource fetch failed: {e}")
            return json.dumps({"ok": False, "error": "fetch_failed", "message": str(e)}, ensure_ascii=False)

    return FunctionTool.from_defaults(
        async_fn=datasource_fetch_handler,
        name=f"datasource-fetch-{kb_id[:8]}",
        description=description,
        return_direct=False,
    )


async def _read_full_doc(entity, tenant_id: str) -> Optional[str]:
    """Read the original markdown from the file store; None if unavailable."""
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
