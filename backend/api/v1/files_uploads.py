"""/v1/files/uploads/* — resumable multipart upload endpoints."""
import traceback
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from api.api_exception import ApiException
from api.v1.files import _entity_to_read
from common.chat.response_model import success_response
from db.db_context import get_db_session
from db.models.file.file import FilePurpose
from service.file.upload_session_service import UploadSessionService
from service.injection import get_tenant_id


files_uploads_router = APIRouter()


def _session_summary(row) -> dict:
    return {
        "upload_id": row.id,
        "file_name": row.file_name,
        "purpose": row.purpose,
        "status": row.status,
        "expires_at": row.expires_at.isoformat() + "Z"
        if row.expires_at and row.expires_at.tzinfo is None
        else (row.expires_at.isoformat() if row.expires_at else None),
        "parts": [{"part": p.get("part"), "size": p.get("size"), "md5": p.get("md5")}
                  for p in (row.parts or [])],
        "file_id": row.file_id,
    }


@files_uploads_router.post("")
async def create_upload_session(
    file_name: str = Form(...),
    purpose: str = Form(...),
    expires_in: Optional[int] = Form(None),
    session: AsyncSession = Depends(get_db_session),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        purpose_enum = FilePurpose(purpose)
    except ValueError:
        raise ApiException(
            code=400,
            message=f"Invalid purpose '{purpose}'. Must be one of: {[p.value for p in FilePurpose]}",
        )
    if expires_in is not None and expires_in < 0:
        raise ApiException(code=400, message="expires_in must be >= 0 (or omitted)")

    svc = UploadSessionService(session)
    row = await svc.create_session(
        tenant_id=tenant_id,
        file_name=file_name,
        purpose=purpose_enum,
        expires_in_seconds=expires_in,
    )
    return success_response(data=_session_summary(row))


@files_uploads_router.put("/{upload_id}/parts/{part_number}")
async def put_part(
    upload_id: str,
    part_number: int,
    chunk: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
    tenant_id: str = Depends(get_tenant_id),
):
    svc = UploadSessionService(session)
    try:
        row = await svc.write_part(
            upload_id=upload_id,
            tenant_id=tenant_id,
            part_number=part_number,
            upload=chunk,
        )
    except ValueError as e:
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"[/v1/files/uploads] put_part failed: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to write part: {e}")
    return success_response(data=_session_summary(row))


@files_uploads_router.post("/{upload_id}/complete")
async def complete_upload(
    upload_id: str,
    part_count: Optional[int] = Form(None),
    session: AsyncSession = Depends(get_db_session),
    tenant_id: str = Depends(get_tenant_id),
):
    svc = UploadSessionService(session)
    try:
        entity, is_new = await svc.complete(
            upload_id=upload_id,
            tenant_id=tenant_id,
            expected_part_count=part_count,
        )
    except ValueError as e:
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"[/v1/files/uploads] complete failed: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to complete upload: {e}")

    # Enqueue background processing for text-extractable types. Everything
    # else (images, videos, audio, other binaries) is marked terminal right
    # here — the agent's multimodal tool reads raw bytes via file_path and
    # doesn't need status=succeeded, but callers / UIs that do otherwise
    # would spin forever.
    if is_new:
        from api.v1.files import _needs_background_processing
        if _needs_background_processing(entity.file_extension):
            try:
                import app.worker as background_worker
                background_worker.process_file_resource_task.delay(
                    file_id=entity.id, tenant_id=tenant_id
                )
            except Exception:
                logger.warning(
                    f"[/v1/files/uploads] failed to enqueue processing for {entity.id}"
                )
        else:
            from common.knowledgebase.types import FileStatus
            from service.file.file_resource_service import FileResourceService
            svc = FileResourceService(session)
            await svc.mark_status(
                file_id=entity.id,
                tenant_id=tenant_id,
                status=FileStatus.succeeded,
            )
            refreshed = await svc.get_file(file_id=entity.id, tenant_id=tenant_id)
            if refreshed is not None:
                entity = refreshed

    return success_response(data=_entity_to_read(entity))


@files_uploads_router.delete("/{upload_id}")
async def cancel_upload(
    upload_id: str,
    session: AsyncSession = Depends(get_db_session),
    tenant_id: str = Depends(get_tenant_id),
):
    svc = UploadSessionService(session)
    ok = await svc.cancel(upload_id=upload_id, tenant_id=tenant_id)
    if not ok:
        raise ApiException(
            code=404,
            message=f"Upload session {upload_id} not found or already completed",
        )
    return success_response(data={"upload_id": upload_id, "cancelled": True})
