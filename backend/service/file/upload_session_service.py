"""Resumable / chunked upload sessions for /v1/files.

Flow:
    1. POST /v1/files/uploads              → create session, get upload_id
    2. PUT  /v1/files/uploads/{id}/parts/N → write chunk, record part metadata
    3. POST /v1/files/uploads/{id}/complete→ concat parts, create FileEntity
    4. DELETE /v1/files/uploads/{id}       → cancel; delete temp parts

Chunks are written to the same object store as final files, under a
`uploads/{tenant_id}/{upload_id}/part-{N}` path. Concatenation happens in
memory on complete — adequate for files up to a few hundred MB. Larger uploads
should switch to native multipart upload (deferred).
"""
import hashlib
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Optional, Tuple

from fastapi import UploadFile
from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.models.file.file import FileEntity, FilePurpose
from db.models.file.upload_session import (
    FileUploadSessionEntity,
    UploadSessionStatus,
)
from pairag.file.store.file_store_helper import file_store
from service.file.file_resource_service import FileResourceService
from utils.upload_file_utils import write_upload_to_store


UPLOAD_SESSION_TTL_HOURS = 24


def _chunk_path(tenant_id: str, upload_id: str, part: int) -> str:
    return f"uploads/{tenant_id}/{upload_id}/part-{part:06d}"


class UploadSessionService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_session(
        self,
        *,
        tenant_id: str,
        file_name: str,
        purpose: FilePurpose,
        expires_in_seconds: Optional[int] = None,
    ) -> FileUploadSessionEntity:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        entity = FileUploadSessionEntity(
            tenant_id=tenant_id,
            file_name=file_name,
            purpose=purpose.value,
            expires_in_seconds=expires_in_seconds,
            status=UploadSessionStatus.ACTIVE.value,
            expires_at=now + timedelta(hours=UPLOAD_SESSION_TTL_HOURS),
            parts=[],
        )
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def get_session(
        self, upload_id: str, tenant_id: str
    ) -> Optional[FileUploadSessionEntity]:
        stmt = select(FileUploadSessionEntity).where(
            FileUploadSessionEntity.id == upload_id,
            FileUploadSessionEntity.tenant_id == tenant_id,
        )
        return (await self.session.exec(stmt)).first()

    async def write_part(
        self,
        *,
        upload_id: str,
        tenant_id: str,
        part_number: int,
        upload: UploadFile,
    ) -> FileUploadSessionEntity:
        session_row = await self.get_session(upload_id=upload_id, tenant_id=tenant_id)
        if session_row is None:
            raise ValueError(f"Upload session {upload_id} not found")
        if session_row.status != UploadSessionStatus.ACTIVE.value:
            raise ValueError(
                f"Upload session {upload_id} is in status '{session_row.status}'; "
                f"only ACTIVE sessions accept parts"
            )
        if part_number < 1:
            raise ValueError("part_number must be >= 1")

        # Write the chunk bytes to the store. Compute md5 for integrity.
        path = _chunk_path(tenant_id, upload_id, part_number)
        stored = await write_upload_to_store(
            upload=upload, destination_path=path, tenant_id=tenant_id
        )

        # Replace-in-place if the client re-uploads the same part (resume).
        parts = [p for p in (session_row.parts or []) if p.get("part") != part_number]
        parts.append({
            "part": part_number,
            "size": stored.file_size,
            "path": stored.file_path,
            "md5": stored.file_md5,
        })
        parts.sort(key=lambda p: p["part"])
        session_row.parts = parts
        session_row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(session_row)
        await self.session.commit()
        await self.session.refresh(session_row)
        return session_row

    async def complete(
        self,
        *,
        upload_id: str,
        tenant_id: str,
        expected_part_count: Optional[int] = None,
    ) -> Tuple[FileEntity, bool]:
        session_row = await self.get_session(upload_id=upload_id, tenant_id=tenant_id)
        if session_row is None:
            raise ValueError(f"Upload session {upload_id} not found")
        if session_row.status == UploadSessionStatus.COMPLETED.value and session_row.file_id:
            # Idempotent retry — return the already-created file.
            svc = FileResourceService(self.session)
            existing = await svc.get_file(
                file_id=session_row.file_id, tenant_id=tenant_id
            )
            if existing is not None:
                return existing, False
        if session_row.status != UploadSessionStatus.ACTIVE.value:
            raise ValueError(
                f"Upload session {upload_id} is in status '{session_row.status}'"
            )
        parts = sorted(session_row.parts or [], key=lambda p: p["part"])
        if not parts:
            raise ValueError(f"Upload session {upload_id} has no parts to assemble")
        if expected_part_count is not None and len(parts) != expected_part_count:
            raise ValueError(
                f"Expected {expected_part_count} parts, got {len(parts)}"
            )
        part_numbers = [p["part"] for p in parts]
        if part_numbers != list(range(1, len(parts) + 1)):
            raise ValueError(
                f"Upload session {upload_id} has gaps in parts: {part_numbers}"
            )

        # In-memory concatenation. OK up to ~hundreds of MB.
        buf = BytesIO()
        total_md5 = hashlib.md5()
        total_size = 0
        for p in parts:
            data = await file_store.read_async(
                file_path=p["path"], tenant_id=tenant_id
            )
            chunk = data.read() if hasattr(data, "read") else data
            buf.write(chunk)
            total_md5.update(chunk)
            total_size += len(chunk)

        buf.seek(0)

        # Route through the shared ingest flow so multipart inherits the same
        # dedup / revive / orphan-cleanup policy as single-part upload. In
        # particular a previously-failed row for these bytes gets revived
        # (bytes rewritten, status reset to pending) instead of being reused
        # as-is, which would skip the retry's reprocess.
        #
        # ingest() commits, which expires all session objects — so snapshot
        # everything we still need from session_row before the call.
        extension = os.path.splitext(session_row.file_name or "")[1].lower()
        file_name = session_row.file_name
        purpose_enum = FilePurpose(session_row.purpose)
        expires_in_seconds = session_row.expires_in_seconds
        parts_snapshot = list(session_row.parts or [])

        svc = FileResourceService(self.session)
        final_entity, is_new = await svc.ingest(
            tenant_id=tenant_id,
            purpose=purpose_enum,
            file_name=file_name,
            file_extension=extension,
            file_size=total_size,
            file_md5=total_md5.hexdigest(),
            source_stream=buf,
            metadata={"via": "multipart", "upload_id": upload_id},
            expires_in_seconds=expires_in_seconds,
        )

        # Re-fetch the session row since ingest()'s commit expired the original
        # instance — assigning directly would still work but subsequent reads
        # (e.g. session_row.parts) would trigger a lazy reload under the async
        # greenlet and blow up.
        session_row = await self.get_session(upload_id=upload_id, tenant_id=tenant_id)
        session_row.status = UploadSessionStatus.COMPLETED.value
        session_row.file_id = final_entity.id
        session_row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(session_row)
        await self.session.commit()
        # The commit above re-expires final_entity; refresh so callers can read
        # its attributes synchronously (FastAPI response serialization does).
        await self.session.refresh(final_entity)

        # Best-effort: part blobs are no longer needed whether we wrote or not.
        await self._delete_parts_at_paths(parts_snapshot, tenant_id)
        return final_entity, is_new

    async def cancel(self, *, upload_id: str, tenant_id: str) -> bool:
        session_row = await self.get_session(upload_id=upload_id, tenant_id=tenant_id)
        if session_row is None:
            return False
        if session_row.status == UploadSessionStatus.COMPLETED.value:
            # Too late to cancel — caller should DELETE the resulting file instead.
            return False
        await self._delete_parts(session_row, tenant_id)
        session_row.status = UploadSessionStatus.CANCELLED.value
        session_row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(session_row)
        await self.session.commit()
        return True

    async def _delete_parts(
        self, session_row: FileUploadSessionEntity, tenant_id: str
    ) -> None:
        await self._delete_parts_at_paths(session_row.parts or [], tenant_id)

    async def _delete_parts_at_paths(
        self, parts: list, tenant_id: str
    ) -> None:
        """Stateless version: takes a snapshot of parts so callers can invoke
        it after the associated session row has been expired by an intervening
        commit without triggering a lazy reload under async.
        """
        if not hasattr(file_store, "delete_async"):
            return
        for p in parts:
            try:
                await file_store.delete_async(
                    file_path=p["path"], tenant_id=tenant_id
                )
            except Exception:
                logger.warning(
                    f"[UploadSession] failed to delete part {p.get('part')} "
                    f"at {p.get('path')}"
                )
