"""FileResourceService — backs the new /v1/files resource.

Decoupled from knowledgebase: writes rows into `pai_file` and
`pai_file_text_content` only. Also serves all attachment-consumer code
(agent_service, code_sandbox_tool, message_service).
"""
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import BinaryIO, Dict, List, Optional, Tuple

from fastapi import UploadFile
from loguru import logger
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from common.knowledgebase.types import FileStatus
from db.models.file.chunk import FileChunkEntity
from db.models.file.file import (
    FileEntity,
    FilePurpose,
    FileTextContentEntity,
)
from pairag.file.store.file_store_helper import file_store
from tools.utils.attachments import aget_file_base64_content, get_file_mime_type
from utils.upload_file_utils import preview_upload


def _yyyymm(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y%m")


def _build_storage_path(tenant_id: str, file_id: str, extension: str) -> str:
    # Path shape is decoupled from kb_id intentionally; see plan Phase 1 §3.4.
    return f"files/{tenant_id}/{_yyyymm()}/{file_id}{extension or ''}"


# Default TTL (in days) per purpose. None means "never expires".
_DEFAULT_TTL_BY_PURPOSE: Dict[FilePurpose, Optional[int]] = {
    FilePurpose.CHAT_ATTACHMENT: 7,
    FilePurpose.KB_INGESTION: None,
    FilePurpose.VISION: 1,
    FilePurpose.AVATAR: None,
}


def _compute_expires_at(
    purpose: FilePurpose,
    override_seconds: Optional[int] = None,
) -> Optional[datetime]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if override_seconds is not None:
        # 0 disables expiry explicitly; negative is invalid (caller validated).
        if override_seconds <= 0:
            return None
        return now + timedelta(seconds=override_seconds)
    days = _DEFAULT_TTL_BY_PURPOSE.get(purpose)
    if days is None:
        return None
    return now + timedelta(days=days)


class FileResourceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------- queries -------------
    async def get_file(self, file_id: str, tenant_id: str) -> Optional[FileEntity]:
        stmt = select(FileEntity).where(
            FileEntity.id == file_id,
            FileEntity.tenant_id == tenant_id,
        )
        return (await self.session.exec(stmt)).first()

    async def get_files(self, file_ids: List[str], tenant_id: str) -> List[FileEntity]:
        if not file_ids:
            return []
        stmt = select(FileEntity).where(
            FileEntity.id.in_(file_ids),
            FileEntity.tenant_id == tenant_id,
        )
        return list((await self.session.exec(stmt)).all())

    async def find_by_md5(
        self,
        *,
        md5: str,
        purpose: FilePurpose,
        tenant_id: str,
    ) -> Optional[FileEntity]:
        stmt = select(FileEntity).where(
            FileEntity.tenant_id == tenant_id,
            FileEntity.file_md5 == md5,
            FileEntity.purpose == purpose.value,
        )
        return (await self.session.exec(stmt)).first()

    async def get_text_content(
        self, file_id: str, tenant_id: str
    ) -> Optional[FileTextContentEntity]:
        stmt = select(FileTextContentEntity).where(
            FileTextContentEntity.file_id == file_id,
            FileTextContentEntity.tenant_id == tenant_id,
        )
        return (await self.session.exec(stmt)).first()

    # ------------- mutations -------------
    # Statuses that represent "this row already did useful work or is doing it".
    # Matching against these short-circuits re-upload. Anything else (failed,
    # cancelled) is revived in place so the retry path actually reprocesses.
    _REUSABLE_STATUSES = frozenset({
        FileStatus.pending.value,
        FileStatus.parsing.value,
        FileStatus.persisting.value,
        FileStatus.succeeded.value,
    })

    async def create_from_upload(
        self,
        *,
        upload: UploadFile,
        purpose: FilePurpose,
        tenant_id: str,
        metadata: Optional[dict] = None,
        alias_id: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> Tuple[FileEntity, bool]:
        """Single-part upload. Thin wrapper around :meth:`ingest`.

        Peeks md5/size/ext from the UploadFile, then hands the stream (already
        rewound to 0 by ``preview_upload``) off to the shared ingest flow.
        """
        preview = preview_upload(upload)
        return await self.ingest(
            tenant_id=tenant_id,
            purpose=purpose,
            file_name=preview.file_name,
            file_extension=preview.file_extension,
            file_size=preview.file_size,
            file_md5=preview.file_md5,
            source_stream=upload.file,
            metadata=metadata,
            alias_id=alias_id,
            expires_in_seconds=expires_in_seconds,
        )

    async def ingest(
        self,
        *,
        tenant_id: str,
        purpose: FilePurpose,
        file_name: str,
        file_extension: str,
        file_size: int,
        file_md5: str,
        source_stream: BinaryIO,
        metadata: Optional[dict] = None,
        alias_id: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> Tuple[FileEntity, bool]:
        """Core dedup / revive / create flow for a fully-known blob.

        Shared by :meth:`create_from_upload` (single-part) and
        ``UploadSessionService.complete`` (multipart). Callers pre-compute the
        md5/size/extension/name and hand in a stream positioned at offset 0.

        Returns ``(entity, is_new)``:

        - ``is_new=False`` — an existing row in a reusable status
          (pending/parsing/persisting/succeeded) was returned unchanged; the
          stream is NOT written to storage. Callers MUST NOT re-enqueue
          processing.
        - ``is_new=True`` — either a brand-new row, or a previously
          failed/cancelled row revived in place (bytes written, status reset
          to pending, ``failed_reason`` cleared). Callers SHOULD re-enqueue
          background processing.
        """
        # 1. Pre-write dedup.
        existing = await self.find_by_md5(
            md5=file_md5, purpose=purpose, tenant_id=tenant_id
        )
        if existing is not None and existing.status in self._REUSABLE_STATUSES:
            logger.info(
                f"[FileResource] Dedup hit for tenant={tenant_id} md5={file_md5} "
                f"purpose={purpose.value} → reusing file_id={existing.id}"
            )
            return existing, False

        # 2. Build entity (revive vs fresh). Storage path is keyed by entity.id
        #    so the two flows converge on the same template.
        entity = existing or FileEntity(
            tenant_id=tenant_id,
            purpose=purpose.value,
            alias_id=alias_id,
            file_metadata=metadata or {},
        )
        destination_path = _build_storage_path(
            tenant_id=tenant_id,
            file_id=entity.id,
            extension=file_extension,
        )
        write_result = await file_store.write_async(
            file=source_stream,
            file_name=file_name,
            file_path=destination_path,
            tenant_id=tenant_id,
        )

        # 3. Either way we end up with a pending row pointing at the just-
        #    written bytes. Metadata merges (don't clobber caller keys) and
        #    failed_reason is cleared so revive doesn't poison the retry.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        entity.file_name = file_name
        entity.file_extension = file_extension
        entity.file_size = file_size
        entity.file_md5 = file_md5
        entity.mime_type = get_file_mime_type(file_extension)
        entity.file_path = write_result.file_path
        entity.status = FileStatus.pending.value
        entity.failed_reason = None
        entity.expires_at = _compute_expires_at(purpose, expires_in_seconds)
        entity.updated_at = now
        if metadata:
            merged = dict(entity.file_metadata or {})
            merged.update(metadata)
            entity.file_metadata = merged

        self.session.add(entity)
        try:
            await self.session.commit()
            await self.session.refresh(entity)
        except IntegrityError:
            # Race with a concurrent creator that won the unique constraint.
            # Roll back the ORM state and best-effort remove the bytes we just
            # wrote so they don't linger as orphans (the winner's bytes live
            # at a different path keyed by their entity id).
            await self.session.rollback()
            await self._try_delete_blob(write_result.file_path, tenant_id)
            winner = await self.find_by_md5(
                md5=file_md5, purpose=purpose, tenant_id=tenant_id
            )
            if winner is not None and winner.status in self._REUSABLE_STATUSES:
                return winner, False
            # Winner is failed/cancelled or gone — surface the error rather
            # than silently returning a row the caller can't use.
            raise
        return entity, True

    async def _try_delete_blob(self, file_path: str, tenant_id: str) -> None:
        """Best-effort delete of a stray object in the file_store."""
        if not file_path or not hasattr(file_store, "delete_async"):
            return
        try:
            await file_store.delete_async(file_path=file_path, tenant_id=tenant_id)
        except Exception:
            logger.warning(
                f"[FileResource] failed to clean up orphan blob at {file_path} "
                f"(tenant={tenant_id}); a GC pass will need to sweep it."
            )

    async def write_text_content(
        self,
        *,
        file_id: str,
        tenant_id: str,
        content: str,
        extractor_version: str,
        truncated_at_extract: bool = False,
    ) -> None:
        existing = await self.get_text_content(file_id=file_id, tenant_id=tenant_id)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if existing is None:
            row = FileTextContentEntity(
                file_id=file_id,
                tenant_id=tenant_id,
                content=content,
                content_length=len(content),
                extractor_version=extractor_version,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
        else:
            existing.content = content
            existing.content_length = len(content)
            existing.extractor_version = extractor_version
            existing.updated_at = now
            self.session.add(existing)
        # Record the extraction-time truncation flag on the parent so callers
        # don't need to join the text table to find out.
        file_entity = await self.get_file(file_id=file_id, tenant_id=tenant_id)
        if file_entity is not None:
            meta = dict(file_entity.file_metadata or {})
            meta["truncated_at_extract"] = bool(truncated_at_extract)
            file_entity.file_metadata = meta
            file_entity.updated_at = now
            self.session.add(file_entity)
        await self.session.commit()

    async def get_text_slice(
        self,
        *,
        file_id: str,
        tenant_id: str,
        offset: int,
        limit: int,
    ) -> Optional[dict]:
        """Return a sliced view of the extracted text.

        Returns `None` if no extracted text exists. Slicing happens in Python —
        at MAX_EXTRACT_CHARS (500KB) the cost is negligible and keeps the code
        DB-agnostic (no SUBSTRING dialect differences).
        """
        row = await self.get_text_content(file_id=file_id, tenant_id=tenant_id)
        if row is None:
            return None
        full = row.content or ""
        total = row.content_length or len(full)
        if offset < 0:
            offset = 0
        if offset > total:
            offset = total
        end = offset + limit if limit > 0 else total
        slice_str = full[offset:end]
        # Surface extract-time truncation so the client knows the DB blob is
        # itself a prefix of what's actually in the file.
        file_entity = await self.get_file(file_id=file_id, tenant_id=tenant_id)
        truncated_at_extract = bool(
            (file_entity.file_metadata or {}).get("truncated_at_extract")
            if file_entity
            else False
        )
        return {
            "file_id": file_id,
            "content": slice_str,
            "offset": offset,
            "limit": limit,
            "total_length": total,
            "has_more": (offset + len(slice_str)) < total,
            "truncated_at_extract": truncated_at_extract,
            "extractor_version": row.extractor_version,
        }

    async def mark_status(
        self,
        *,
        file_id: str,
        tenant_id: str,
        status: FileStatus,
        failed_reason: Optional[str] = None,
    ) -> None:
        entity = await self.get_file(file_id=file_id, tenant_id=tenant_id)
        if not entity:
            logger.warning(f"[FileResource] mark_status: file {file_id} not found")
            return
        entity.status = status.value if hasattr(status, "value") else str(status)
        entity.failed_reason = failed_reason
        entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(entity)
        await self.session.commit()

    async def get_presigned_url(
        self, file_id: str, tenant_id: str
    ) -> Optional[str]:
        entity = await self.get_file(file_id=file_id, tenant_id=tenant_id)
        if not entity or not entity.file_path:
            return None
        return await file_store.get_url_async(
            file_path=entity.file_path, tenant_id=tenant_id
        )

    async def read_bytes(self, file_id: str, tenant_id: str):
        entity = await self.get_file(file_id=file_id, tenant_id=tenant_id)
        if not entity or not entity.file_path:
            return None
        return await file_store.read_async(
            file_path=entity.file_path, tenant_id=tenant_id
        )

    # ------------- chunks (in-file retrieval) -------------
    async def replace_chunks(
        self,
        *,
        file_id: str,
        tenant_id: str,
        chunks: List[dict],
    ) -> int:
        """Drop any existing chunks for this file and write the new ones.

        Idempotent: re-running extraction (e.g. after revive) produces a fresh
        set in a single transaction. Returns the number of chunks written.
        """
        from sqlalchemy import delete as sa_delete
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        await self.session.exec(
            sa_delete(FileChunkEntity).where(
                FileChunkEntity.file_id == file_id,
                FileChunkEntity.tenant_id == tenant_id,
            )
        )
        rows: List[FileChunkEntity] = []
        for c in chunks:
            rows.append(FileChunkEntity(
                tenant_id=tenant_id,
                file_id=file_id,
                chunk_index=c["index"],
                content=c["content"],
                start_offset=c.get("start", 0),
                end_offset=c.get("end", 0),
                token_count=c.get("token_count", 0),
                chunk_metadata=c.get("metadata", {}),
                created_at=now,
            ))
        for r in rows:
            self.session.add(r)
        await self.session.commit()
        return len(rows)

    async def count_chunks(self, file_id: str, tenant_id: str) -> int:
        from sqlalchemy import func as sa_func
        stmt = select(sa_func.count()).select_from(FileChunkEntity).where(
            FileChunkEntity.file_id == file_id,
            FileChunkEntity.tenant_id == tenant_id,
        )
        return (await self.session.exec(stmt)).one_or_none() or 0

    async def search_chunks(
        self,
        *,
        file_id: str,
        tenant_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[dict]:
        """Keyword-score chunks against ``query`` and return the top ``top_k``.

        Simple, dependency-free retrieval: case-insensitive occurrence count of
        each whitespace-split query term, summed per chunk. Good enough for a
        demo over a single file; swap in BM25 / embeddings later without
        changing the endpoint contract.
        """
        if not query.strip():
            return []
        stmt = select(FileChunkEntity).where(
            FileChunkEntity.file_id == file_id,
            FileChunkEntity.tenant_id == tenant_id,
        ).order_by(FileChunkEntity.chunk_index)
        rows = list((await self.session.exec(stmt)).all())
        if not rows:
            return []

        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []

        scored: List[tuple[float, FileChunkEntity]] = []
        for row in rows:
            body = (row.content or "").lower()
            score = sum(body.count(t) for t in terms)
            if score > 0:
                scored.append((score, row))
        # Fallback: if nothing matched at all, surface the first N chunks so
        # the caller at least gets some context instead of an empty array.
        if not scored:
            scored = [(0.0, r) for r in rows[:top_k]]
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_index))
        out: List[dict] = []
        for score, row in scored[:top_k]:
            out.append({
                "chunk_id": row.id,
                "chunk_index": row.chunk_index,
                "content": row.content,
                "start_offset": row.start_offset,
                "end_offset": row.end_offset,
                "score": float(score),
            })
        return out

    # ------------- agent-consumer facing helpers -------------
    # Back-compat for the old FileService.get_file_by_id signature used by
    # code_sandbox_tool and the spreadsheet detector in agent_service.
    async def get_file_by_id(
        self, file_id: str, tenant_id: str
    ) -> Optional[FileEntity]:
        return await self.get_file(file_id=file_id, tenant_id=tenant_id)

    async def get_files_by_ids(
        self, file_ids: List[str], tenant_id: str
    ) -> List[FileEntity]:
        return await self.get_files(file_ids=file_ids, tenant_id=tenant_id)

    # Max characters of a single file's extracted text to inject into the
    # LLM's context via the file_reader tool. Keeping this tight so a 500KB
    # extract doesn't blow the context window; the client can paginate via
    # `GET /v1/files/{id}/text?offset&limit` for the remainder.
    LLM_INLINE_TEXT_LIMIT = 5_000

    async def get_file_contents_map(
        self, file_ids: List[str], tenant_id: str
    ) -> Dict[str, str]:
        """Return {file_name: formatted_content} for the file_reader tool.

        Injects only the first `LLM_INLINE_TEXT_LIMIT` characters of extracted
        text and appends a truncation note when the full text is longer.
        Missing rows are surfaced with an empty body.
        """
        if not file_ids:
            return {}
        files = await self.get_files(file_ids=file_ids, tenant_id=tenant_id)
        if not files:
            return {}
        stmt = select(FileTextContentEntity).where(
            FileTextContentEntity.file_id.in_([f.id for f in files]),
            FileTextContentEntity.tenant_id == tenant_id,
        )
        text_rows = list((await self.session.exec(stmt)).all())
        text_by_id = {row.file_id: row for row in text_rows}

        out: Dict[str, str] = {}
        for f in files:
            row = text_by_id.get(f.id)
            header = f"📄 文件“{f.file_name}” 的内容如下：\n\n"
            if not row or not row.content:
                out[f.file_name] = header
                continue
            body = row.content
            total = row.content_length or len(body)
            inline = body[: self.LLM_INLINE_TEXT_LIMIT]
            trailer = ""
            if total > self.LLM_INLINE_TEXT_LIMIT:
                trailer = (
                    f"\n\n[truncated for context; shown {len(inline)} of {total} chars. "
                    f"Call GET /v1/files/{f.id}/text?offset={len(inline)}&limit=... for more.]"
                )
            extract_truncated = (f.file_metadata or {}).get("truncated_at_extract")
            if extract_truncated:
                trailer += (
                    "\n[note: source file exceeded extractor cap; stored text is itself a prefix.]"
                )
            out[f.file_name] = f"{header} {inline}{trailer}"
        return out

    async def get_file_base64_list(
        self, file_ids: List[str], tenant_id: str
    ) -> List[str]:
        """Return base64-encoded data URIs for the multimodal-parser tool.

        Used for images — small enough for context, and many LLM APIs accept
        data URIs inline. Videos should use ``get_file_url_list`` instead
        (see that method's docstring).
        """
        if not file_ids:
            return []
        files = await self.get_files(file_ids=file_ids, tenant_id=tenant_id)
        tasks = [aget_file_base64_content(f) for f in files]
        return await asyncio.gather(*tasks)

    async def get_file_url_list(
        self, file_ids: List[str], tenant_id: str
    ) -> List[str]:
        """Return presigned URLs for a batch of files.

        Used by batched /v1/files/{id}/url-style callers and as a future hook
        if/when a caller needs to pass files to a remote service by URL
        instead of by base64. The multimodal tool chain itself stays on
        base64 (verified on the PAI-RAG feature branch), so this method has
        no current caller inside parse_attachment_tools.
        """
        if not file_ids:
            return []
        files = await self.get_files(file_ids=file_ids, tenant_id=tenant_id)
        urls: List[str] = []
        for f in files:
            if not f.file_path:
                continue
            try:
                url = await file_store.get_url_async(
                    file_path=f.file_path, tenant_id=f.tenant_id
                )
            except Exception:
                logger.warning(
                    f"[FileResource] get_url_async failed for file {f.id}; "
                    f"dropping from multimodal tool payload"
                )
                continue
            if url:
                urls.append(url)
        return urls

    # ------------- ref counting (for message-attached lifecycle) -------------
    # NB: both inc and dec accept a list that MAY contain duplicate file_ids
    # (e.g. the same file attached to two messages in a thread). A naive
    # `UPDATE ... WHERE id IN (...)` would collapse duplicates — SQL evaluates
    # the predicate once per row, so `[A, A]` only moves A by one. We group
    # by id and issue one UPDATE per unique id using the occurrence count.
    async def increment_refs(self, file_ids: List[str], tenant_id: str) -> None:
        if not file_ids:
            return
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for fid, count in Counter(file_ids).items():
            if count <= 0:
                continue
            stmt = (
                update(FileEntity)
                .where(
                    FileEntity.id == fid,
                    FileEntity.tenant_id == tenant_id,
                )
                .values(
                    ref_count=FileEntity.ref_count + count,
                    updated_at=now,
                )
            )
            await self.session.exec(stmt)

    async def decrement_refs(self, file_ids: List[str], tenant_id: str) -> None:
        if not file_ids:
            return
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for fid, count in Counter(file_ids).items():
            if count <= 0:
                continue
            # Only move rows with enough headroom — defensive clamp so an
            # over-release (e.g. a message counted twice by mistake) doesn't
            # drive ref_count negative. A mismatched row is left untouched;
            # inspect logs if you see sweep results that surprise you.
            stmt = (
                update(FileEntity)
                .where(
                    FileEntity.id == fid,
                    FileEntity.tenant_id == tenant_id,
                    FileEntity.ref_count >= count,
                )
                .values(
                    ref_count=FileEntity.ref_count - count,
                    updated_at=now,
                )
            )
            await self.session.exec(stmt)

    # ------------- deletion / GC -------------
    async def hard_delete(self, file_id: str, tenant_id: str) -> bool:
        """Delete the DB row and best-effort remove the object from file_store.

        Returns True if a row was deleted, False if not found. Does NOT check
        ref_count — caller is responsible for policy (`DELETE /v1/files/{id}`
        is always a force-delete per OpenAI convention).
        """
        entity = await self.get_file(file_id=file_id, tenant_id=tenant_id)
        if not entity:
            return False
        # Try to remove the stored object; log but don't fail if the backend
        # doesn't support delete or has already GC'd the object.
        try:
            if entity.file_path and hasattr(file_store, "delete_async"):
                await file_store.delete_async(
                    file_path=entity.file_path, tenant_id=tenant_id
                )
        except Exception:
            logger.warning(
                f"[FileResource] file_store delete failed for {file_id} "
                f"(path={entity.file_path}); proceeding with DB delete."
            )
        # Cascade text content + chunks (no FK on SQLite, so delete manually).
        from sqlalchemy import delete as sa_delete
        text_row = await self.get_text_content(file_id=file_id, tenant_id=tenant_id)
        if text_row is not None:
            await self.session.delete(text_row)
        await self.session.exec(
            sa_delete(FileChunkEntity).where(
                FileChunkEntity.file_id == file_id,
                FileChunkEntity.tenant_id == tenant_id,
            )
        )
        await self.session.delete(entity)
        await self.session.commit()
        return True

    async def sweep_expired_candidates(
        self, *, limit: int = 200
    ) -> List[FileEntity]:
        """Return a batch of rows that are eligible for hard-delete.

        Eligible = `ref_count = 0 AND expires_at IS NOT NULL AND expires_at < now`.
        KB-ingestion / avatar files with `expires_at = NULL` are never swept.
        Intended to be called by the GC Celery task.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        stmt = (
            select(FileEntity)
            .where(
                FileEntity.ref_count == 0,
                FileEntity.expires_at.is_not(None),
                FileEntity.expires_at < now,
            )
            .limit(limit)
        )
        return list((await self.session.exec(stmt)).all())
