"""/v1/files router — new independent File resource."""
import json
import traceback
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from api.api_exception import ApiException
from api.v1.dto.file_dto import (
    FileChunkHit,
    FileChunkSearchResult,
    FileRead,
    FileTextRead,
    FileUrlRead,
)
from common.chat.response_model import success_response
from db.models.file.file import FilePurpose
from service.file.file_resource_service import FileResourceService
from service.injection import get_file_resource_service, get_tenant_id


# Bounds on /text?offset&limit. Max 500KB per response — matches the
# extractor cap so one round-trip can deliver everything stored. Default
# is 50KB which is typical for LLM-context fetches.
_TEXT_DEFAULT_LIMIT = 50_000
_TEXT_MAX_LIMIT = 500_000


files_router = APIRouter()


def _entity_to_read(entity) -> FileRead:
    return FileRead(
        id=entity.id,
        tenant_id=entity.tenant_id,
        purpose=entity.purpose,
        file_name=entity.file_name,
        file_extension=entity.file_extension,
        file_size=entity.file_size or 0,
        file_md5=entity.file_md5,
        mime_type=entity.mime_type,
        status=entity.status,
        failed_reason=entity.failed_reason,
        ref_count=entity.ref_count or 0,
        expires_at=entity.expires_at,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        file_metadata=entity.file_metadata or {},
    )


@files_router.post("", status_code=202)
async def create_file(
    file: UploadFile = File(...),
    purpose: str = Form(...),
    metadata: Optional[str] = Form(None),
    expires_in: Optional[int] = Form(None),
    file_service: FileResourceService = Depends(get_file_resource_service),
    tenant_id: str = Depends(get_tenant_id),
):
    """Upload a new file.

    Returns 202 Accepted with the FileEntity — status begins as `pending`.
    Caller polls GET /v1/files/{id} (or subscribes via SSE /v1/files/events)
    for terminal status (`succeeded` / `failed`).

    Dedup: if a file with the same (tenant, md5, purpose) already exists, the
    existing row is returned — no re-upload, is_new=False.

    TTL: if `expires_in` (seconds) is provided it overrides the purpose default;
    pass 0 to opt out of expiry. Defaults: chat_attachment=7d, vision=24h,
    kb_ingestion=never, avatar=never.
    """
    try:
        purpose_enum = FilePurpose(purpose)
    except ValueError:
        raise ApiException(
            code=400,
            message=f"Invalid purpose '{purpose}'. Must be one of: {[p.value for p in FilePurpose]}",
        )

    if expires_in is not None and expires_in < 0:
        raise ApiException(code=400, message="expires_in must be >= 0 (or omitted)")

    meta: Optional[dict] = None
    if metadata:
        try:
            meta = json.loads(metadata)
            if not isinstance(meta, dict):
                raise ValueError("metadata must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            raise ApiException(code=400, message=f"Invalid metadata: {e}")

    try:
        entity, is_new = await file_service.create_from_upload(
            upload=file,
            purpose=purpose_enum,
            tenant_id=tenant_id,
            metadata=meta,
            expires_in_seconds=expires_in,
        )
    except Exception as e:
        logger.error(f"[/v1/files] upload failed: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to create file: {e}")

    if is_new:
        if _needs_background_processing(entity.file_extension):
            try:
                import app.worker as background_worker
                background_worker.process_file_resource_task.delay(
                    file_id=entity.id,
                    tenant_id=tenant_id,
                )
                logger.info(f"[/v1/files] enqueued file_resource task for {entity.id}")
            except Exception:
                # Enqueue is best-effort; file is already persisted.
                logger.warning(
                    f"[/v1/files] failed to enqueue processing for {entity.id}: "
                    f"{traceback.format_exc()}"
                )
        else:
            # Nothing to extract (images, videos, audio, binary). Mark the row
            # terminal immediately so callers / UIs that wait for succeeded
            # don't spin forever. The agent's multimodal tool reads the raw
            # bytes directly via file_path, independent of status.
            from common.knowledgebase.types import FileStatus
            await file_service.mark_status(
                file_id=entity.id,
                tenant_id=tenant_id,
                status=FileStatus.succeeded,
            )
            # Refresh the local entity so the response body reflects the flip.
            refreshed = await file_service.get_file(
                file_id=entity.id, tenant_id=tenant_id
            )
            if refreshed is not None:
                entity = refreshed

    return success_response(data=_entity_to_read(entity))


@files_router.get("/{file_id}")
async def get_file_api(
    file_id: str,
    file_service: FileResourceService = Depends(get_file_resource_service),
    tenant_id: str = Depends(get_tenant_id),
):
    entity = await file_service.get_file(file_id=file_id, tenant_id=tenant_id)
    if not entity:
        raise ApiException.not_found(file_id, "File")
    return success_response(data=_entity_to_read(entity))


@files_router.get("/{file_id}/content")
async def get_file_content(
    file_id: str,
    file_service: FileResourceService = Depends(get_file_resource_service),
    tenant_id: str = Depends(get_tenant_id),
):
    entity = await file_service.get_file(file_id=file_id, tenant_id=tenant_id)
    if not entity:
        raise ApiException.not_found(file_id, "File")
    stream = await file_service.read_bytes(file_id=file_id, tenant_id=tenant_id)
    if stream is None:
        raise ApiException(code=404, message=f"File {file_id} has no stored bytes")
    media_type = entity.mime_type or "application/octet-stream"
    headers = {}
    if entity.file_name:
        headers["Content-Disposition"] = f'inline; filename="{entity.file_name}"'
    return StreamingResponse(stream, media_type=media_type, headers=headers)


@files_router.get("/{file_id}/text")
async def get_file_text(
    file_id: str,
    offset: int = Query(0, ge=0, description="Character offset into the extracted text"),
    limit: int = Query(
        _TEXT_DEFAULT_LIMIT,
        ge=1,
        le=_TEXT_MAX_LIMIT,
        description=f"Max characters to return (max {_TEXT_MAX_LIMIT})",
    ),
    file_service: FileResourceService = Depends(get_file_resource_service),
    tenant_id: str = Depends(get_tenant_id),
):
    """Return a slice of the extracted text.

    Use `offset` + `limit` to paginate through long files. `has_more` in the
    response indicates whether another page is available. `total_length` lets
    the client decide whether to paginate or fetch in one shot.

    Note: if `truncated_at_extract=true` the extractor itself cut the text off
    at the 500KB safety cap. Paginating past `total_length` will return empty.
    """
    entity = await file_service.get_file(file_id=file_id, tenant_id=tenant_id)
    if not entity:
        raise ApiException.not_found(file_id, "File")
    result = await file_service.get_text_slice(
        file_id=file_id, tenant_id=tenant_id, offset=offset, limit=limit
    )
    if result is None:
        raise ApiException(code=404, message=f"No extracted text available for file {file_id}")
    return success_response(data=FileTextRead(**result))


@files_router.get("/{file_id}/url")
async def get_file_url(
    file_id: str,
    file_service: FileResourceService = Depends(get_file_resource_service),
    tenant_id: str = Depends(get_tenant_id),
):
    entity = await file_service.get_file(file_id=file_id, tenant_id=tenant_id)
    if not entity:
        raise ApiException.not_found(file_id, "File")
    url = await file_service.get_presigned_url(file_id=file_id, tenant_id=tenant_id)
    if not url:
        raise ApiException(code=404, message=f"File {file_id} has no accessible URL")
    return success_response(data=FileUrlRead(file_id=file_id, url=url))


@files_router.get("/{file_id}/chunks")
async def search_file_chunks(
    file_id: str,
    query: str = Query(..., min_length=1, description="Keyword query; whitespace-split into terms"),
    top_k: int = Query(5, ge=1, le=20, description="Max number of chunks to return"),
    file_service: FileResourceService = Depends(get_file_resource_service),
    tenant_id: str = Depends(get_tenant_id),
):
    """Search within a single file's chunks and return the top matches.

    Chunks are produced during extraction for files whose text is large enough
    to warrant retrieval (see ``SEARCHABLE_MIN_CHARS`` in content_extractor).
    Small files return an empty ``hits`` array — the client should fall back
    to ``GET /v1/files/{id}/text`` in that case.
    """
    entity = await file_service.get_file(file_id=file_id, tenant_id=tenant_id)
    if not entity:
        raise ApiException.not_found(file_id, "File")
    total = await file_service.count_chunks(file_id=file_id, tenant_id=tenant_id)
    hits = await file_service.search_chunks(
        file_id=file_id, tenant_id=tenant_id, query=query, top_k=top_k
    )
    return success_response(
        data=FileChunkSearchResult(
            file_id=file_id,
            query=query,
            total_chunks=total,
            hits=[FileChunkHit(**h) for h in hits],
        )
    )


@files_router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    file_service: FileResourceService = Depends(get_file_resource_service),
    tenant_id: str = Depends(get_tenant_id),
):
    """Force-delete a file. Bypasses ref_count — the caller is asserting intent.

    Returns 404 if the file doesn't exist (or belongs to another tenant).
    """
    deleted = await file_service.hard_delete(file_id=file_id, tenant_id=tenant_id)
    if not deleted:
        raise ApiException.not_found(file_id, "File")
    return success_response(data={"id": file_id, "deleted": True})


# File extensions that trigger attachment-style text extraction.
# Text-extractable attachments get previewed; multimodal binaries are served as-is.
_PROCESSABLE_EXTENSIONS = {
    ".xlsx", ".xls", ".csv", ".jsonl",
    ".txt", ".md", ".json", ".yaml", ".yml", ".xml", ".log",
    ".py", ".js", ".ts", ".html", ".css",
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
}


def _needs_background_processing(extension: Optional[str]) -> bool:
    if not extension:
        return False
    return extension.lower() in _PROCESSABLE_EXTENSIONS
