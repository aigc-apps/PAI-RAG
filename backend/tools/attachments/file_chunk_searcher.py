"""Agent tool: search within large file attachments.

When a user attaches a file whose extracted text is too long to inline into
the LLM's context (``FileResourceService.LLM_INLINE_TEXT_LIMIT``), the agent
is still given the file's name/id and a ``search_file_chunks`` tool. The LLM
issues natural-language queries against a specific file_id and gets back the
top-k matching chunks — the standard RAG-over-attachment pattern.

The tool opens a fresh DB session per call rather than closing over a
request-scoped session: by the time the LLM decides to call the tool the
original request's session may have been committed/closed by an
intervening streaming step, which would raise MissingGreenlet on access.
"""
import json
from typing import Annotated, List

from llama_index.core.tools import FunctionTool
from loguru import logger

from db.db_context import create_db_session
from service.file.file_resource_service import FileResourceService


def _format_catalog(files: List[dict]) -> str:
    lines = []
    for f in files:
        lines.append(
            f"- file_id={f['file_id']}  name={f['file_name']}  "
            f"chunks={f['chunk_count']}"
        )
    return "\n".join(lines)


async def aget_file_chunk_searcher(
    tenant_id: str,
    files: List[dict],
):
    """Build a FunctionTool bound to a specific set of large file attachments.

    ``files`` is ``[{file_id, file_name, chunk_count}, ...]`` — the tool's
    description enumerates them so the LLM knows which ``file_id`` values are
    valid without an extra discovery round-trip.
    """
    if not files:
        raise ValueError("files is required")

    catalog = _format_catalog(files)
    allowed_ids = {f["file_id"] for f in files}

    async def asearch_file_chunks(
        file_id: Annotated[
            str,
            "Target file id (must be one of the attached large files listed in the tool description).",
        ],
        query: Annotated[
            str,
            "A natural-language query describing what to look for. Keywords are space-split and case-insensitive.",
        ],
        top_k: Annotated[
            int,
            "Maximum number of chunks to return (1-10). Default 5.",
        ] = 5,
    ) -> str:
        if file_id not in allowed_ids:
            return json.dumps({
                "error": f"file_id '{file_id}' is not in the attached file list",
                "available_file_ids": sorted(allowed_ids),
            }, ensure_ascii=False)

        top_k = max(1, min(int(top_k or 5), 10))
        logger.info(
            f"[file_chunk_searcher] file_id={file_id} query={query!r} top_k={top_k}"
        )
        async with create_db_session() as session:
            svc = FileResourceService(session)
            hits = await svc.search_chunks(
                file_id=file_id,
                tenant_id=tenant_id,
                query=query,
                top_k=top_k,
            )
        return json.dumps(
            {"file_id": file_id, "query": query, "hits": hits},
            ensure_ascii=False,
        )

    description = (
        "Search within a specific large attached file and return the top matching text chunks. "
        "Use this when the file is too long to read in full. The available files are:\n\n"
        f"{catalog}\n\n"
        "Parameters:\n"
        "- file_id (str): one of the file_id values above.\n"
        "- query (str): what you want to find in the file.\n"
        "- top_k (int, optional, default=5): max chunks to return."
    )
    return FunctionTool.from_defaults(
        async_fn=asearch_file_chunks,
        name="search-file-chunks",
        description=description,
        return_direct=False,
    )
