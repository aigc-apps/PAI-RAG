"""DataSource service layer: CRUD + management-state queries.

A data source belongs to exactly one knowledge base (KB -> 0..N data sources).
This service owns the data source's *management state* — the document list
(manifest), sync runs (history) and the aggregate sync status. Document parse
status is NOT duplicated here; it is derived on read by joining the manifest's
``file_id`` to ``KbFileEntity.status``.
"""

import time
from datetime import datetime, timezone
from typing import Optional, List, Tuple

from sqlmodel import select, func, delete, update
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import or_, and_
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.knowledgebase.datasource import (
    DataSourceEntity,
    DataSourceDocumentEntity,
    DataSourceSyncRunEntity,
    DataSourceCreate,
    DataSourceUpdate,
)
from db.models.knowledgebase.file import KbFileEntity
from common.chat.response_model import PagedResult
from common.knowledgebase.types import (
    DataSourceStatus,
    DataSourceDocStatus,
    SyncRunStatus,
    SyncTrigger,
    FileStatus,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DataSourceService:
    """CRUD + observability for data sources. Caller commits the session."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------------------------------------------------------------
    # DataSource CRUD
    # ------------------------------------------------------------------
    async def create_datasource(
        self, kb_id: str, ds_data: DataSourceCreate, tenant_id: str
    ) -> DataSourceEntity:
        """Create a data source under a KB. Caller commits."""
        source_type = (
            ds_data.source_type.value
            if hasattr(ds_data.source_type, "value")
            else ds_data.source_type
        )
        # reject source types that have no registered adapter (would fail every sync)
        from rag.datasource.registry import supported_source_types
        supported = supported_source_types()
        if source_type not in supported:
            raise ValueError(
                f"source_type '{source_type}' is not supported yet. "
                f"Supported types: {', '.join(sorted(supported))}."
            )
        try:
            datasource = DataSourceEntity.model_validate(
                ds_data,
                update={
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "source_type": source_type,
                },
            )
            self.session.add(datasource)
            await self.session.flush()
            await self.session.refresh(datasource)
            logger.info(
                f"Created data source {datasource.id} (key={datasource.datasource_key}) under kb {kb_id}"
            )
            return datasource
        except IntegrityError as e:
            if "Unique" in str(e.orig) or "Duplicate" in str(e.orig):
                raise ValueError(
                    f"Data source key '{ds_data.datasource_key}' already exists in this knowledge base."
                ) from e
            raise ValueError(f"Data source creation failed: {e.orig}") from e

    async def get_datasource(self, ds_id: str, tenant_id: str) -> Optional[DataSourceEntity]:
        result = await self.session.exec(
            select(DataSourceEntity).where(
                DataSourceEntity.id == ds_id,
                DataSourceEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    async def list_datasources(
        self,
        kb_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 20,
        query: Optional[str] = None,
    ) -> PagedResult[List[dict]]:
        conditions = [
            DataSourceEntity.tenant_id == tenant_id,
            DataSourceEntity.kb_id == kb_id,
        ]
        if query:
            q = query.lower()
            conditions.append(
                or_(
                    func.lower(DataSourceEntity.name).like(f"%{q}%"),
                    func.lower(DataSourceEntity.datasource_key).like(f"%{q}%"),
                )
            )
        where_condition = and_(*conditions)

        total = (await self.session.exec(
            select(func.count(DataSourceEntity.id)).where(where_condition)
        )).one_or_none() or 0

        offset = (page - 1) * size
        results = await self.session.exec(
            select(DataSourceEntity)
            .where(where_condition)
            .order_by(DataSourceEntity.created_at.desc(), DataSourceEntity.id.asc())
            .offset(offset)
            .limit(size)
        )
        items = [ds.model_dump(mode="json") for ds in results.all()]
        pages = (total + size - 1) // size if total > 0 else 0
        return PagedResult(items=items, total=total, pages=pages, page=page, size=size)

    async def count_by_kb(self, kb_id: str, tenant_id: str) -> int:
        """Number of data sources bound to a KB (used to gate agent tools)."""
        total = (await self.session.exec(
            select(func.count(DataSourceEntity.id)).where(
                DataSourceEntity.kb_id == kb_id,
                DataSourceEntity.tenant_id == tenant_id,
            )
        )).one_or_none() or 0
        return int(total)

    async def update_datasource(
        self, ds_id: str, update_data: DataSourceUpdate, tenant_id: str
    ) -> DataSourceEntity:
        datasource = await self.get_datasource(ds_id, tenant_id)
        if not datasource:
            raise ValueError(f"Data source '{ds_id}' does not exist.")

        # Use model_fields_set so an explicitly-provided null clears a field,
        # while an omitted field is left unchanged (PATCH semantics).
        fields_set = update_data.model_fields_set
        if "name" in fields_set and update_data.name is not None:
            datasource.name = update_data.name
        if "source_config" in fields_set and update_data.source_config is not None:
            datasource.source_config = update_data.source_config
        if "sync_schedule" in fields_set:
            # allow clearing the schedule (null) to disable scheduled syncs
            datasource.sync_schedule = update_data.sync_schedule
            datasource.next_sync_at = None  # recomputed by the dispatcher on next tick
        if "enabled" in fields_set and update_data.enabled is not None:
            datasource.enabled = update_data.enabled

        datasource.updated_at = _utcnow()
        self.session.add(datasource)
        await self.session.flush()
        await self.session.refresh(datasource)
        return datasource

    async def set_enabled(self, ds_id: str, enabled: bool, tenant_id: str) -> DataSourceEntity:
        datasource = await self.get_datasource(ds_id, tenant_id)
        if not datasource:
            raise ValueError(f"Data source '{ds_id}' does not exist.")
        datasource.enabled = enabled
        datasource.updated_at = _utcnow()
        self.session.add(datasource)
        await self.session.flush()
        await self.session.refresh(datasource)
        return datasource

    async def delete_datasource(self, ds_id: str, tenant_id: str) -> None:
        """Delete a data source and its documents + sync runs.

        Child rows are deleted explicitly (not relying on DB-level FK CASCADE)
        so this is correct regardless of the backend's FK enforcement.

        NOTE: cleanup of the ingested KbFileEntity rows / vectors is handled at
        the orchestration layer (RagService.delete_file) in the sync wiring
        (PR4), since this service has no access to the vector store.
        """
        datasource = await self.get_datasource(ds_id, tenant_id)
        if not datasource:
            raise ValueError(f"Data source '{ds_id}' does not exist.")

        await self.session.execute(
            delete(DataSourceDocumentEntity).where(
                DataSourceDocumentEntity.datasource_id == ds_id
            )
        )
        await self.session.execute(
            delete(DataSourceSyncRunEntity).where(
                DataSourceSyncRunEntity.datasource_id == ds_id
            )
        )
        await self.session.delete(datasource)
        await self.session.flush()
        logger.info(f"Deleted data source {ds_id} (with documents + sync runs)")

    # ------------------------------------------------------------------
    # Documents (manifest / "file list")
    # ------------------------------------------------------------------
    async def list_documents(
        self,
        ds_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 20,
        doc_status: Optional[str] = None,
        query: Optional[str] = None,
    ) -> PagedResult[List[dict]]:
        """List documents of a data source, with parse status joined from KbFileEntity."""
        conditions = [
            DataSourceDocumentEntity.tenant_id == tenant_id,
            DataSourceDocumentEntity.datasource_id == ds_id,
        ]
        if doc_status:
            conditions.append(DataSourceDocumentEntity.doc_status == doc_status)
        if query:
            q = query.lower()
            conditions.append(
                or_(
                    func.lower(func.coalesce(DataSourceDocumentEntity.title, "")).like(f"%{q}%"),
                    func.lower(DataSourceDocumentEntity.path).like(f"%{q}%"),
                )
            )
        where_condition = and_(*conditions)

        total = (await self.session.exec(
            select(func.count(DataSourceDocumentEntity.id)).where(where_condition)
        )).one_or_none() or 0

        offset = (page - 1) * size
        results = await self.session.exec(
            select(DataSourceDocumentEntity, KbFileEntity.status, KbFileEntity.failed_reason)
            .outerjoin(KbFileEntity, DataSourceDocumentEntity.file_id == KbFileEntity.id)
            .where(where_condition)
            .order_by(DataSourceDocumentEntity.updated_at.desc(), DataSourceDocumentEntity.id.asc())
            .offset(offset)
            .limit(size)
        )
        items = []
        for doc, parse_status, failed_reason in results.all():
            d = doc.model_dump(mode="json")
            d["parse_status"] = parse_status
            d["parse_failed_reason"] = failed_reason
            items.append(d)
        pages = (total + size - 1) // size if total > 0 else 0
        return PagedResult(items=items, total=total, pages=pages, page=page, size=size)

    async def list_document_file_ids(self, ds_id: str, tenant_id: str) -> List[str]:
        """All non-null KB file ids ingested by this data source (for cleanup)."""
        results = await self.session.exec(
            select(DataSourceDocumentEntity.file_id).where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
                DataSourceDocumentEntity.file_id.is_not(None),
            )
        )
        return [fid for fid in results.all() if fid]

    async def cancel_sync(self, ds_id: str, tenant_id: str) -> int:
        """Cancel an in-progress sync. Caller commits.

        Sets every not-yet-finished file (pending/parsing/persisting) to
        ``cancelled`` — running parse tasks cooperatively stop via
        ``should_cancel_file_task`` — flips their manifest rows to ``cancelled``,
        and marks the data source ``cancelled``. Already-succeeded files are kept.
        A later sync re-ingests the cancelled docs (see the sync diff).
        """
        now = _utcnow()
        unfinished = (FileStatus.pending, FileStatus.parsing, FileStatus.persisting)

        rows = (await self.session.exec(
            select(DataSourceDocumentEntity, KbFileEntity)
            .join(KbFileEntity, DataSourceDocumentEntity.file_id == KbFileEntity.id)
            .where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
                KbFileEntity.status.in_(unfinished),
            )
        )).all()
        for doc, file_entity in rows:
            file_entity.status = FileStatus.cancelled
            # Bump the version so any already-queued/running task for this file
            # self-cancels via should_cancel_file_task even if its status gets
            # flipped back to parsing before the cancel check runs.
            file_entity.file_version = (file_entity.file_version or 0) + 1
            file_entity.updated_at = now
            doc.doc_status = DataSourceDocStatus.cancelled
            doc.updated_at = now
            self.session.add(file_entity)
            self.session.add(doc)

        # docs that were discovered/fetching but never produced a file row
        pending_docs = (await self.session.exec(
            select(DataSourceDocumentEntity).where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
                DataSourceDocumentEntity.doc_status.in_(
                    (DataSourceDocStatus.discovered, DataSourceDocStatus.fetching, DataSourceDocStatus.ingesting)
                ),
            )
        )).all()
        for doc in pending_docs:
            doc.doc_status = DataSourceDocStatus.cancelled
            doc.updated_at = now
            self.session.add(doc)

        datasource = await self.get_datasource(ds_id, tenant_id)
        if datasource:
            datasource.status = DataSourceStatus.cancelled
            datasource.updated_at = now
            self.session.add(datasource)

        await self.session.flush()
        return len(rows)

    async def reparse_documents(
        self, ds_id: str, tenant_id: str, only_unfinished: bool = True,
    ) -> List[Tuple[str, int]]:
        """Re-kick the parse pipeline for already-fetched files (no re-fetch).

        Resets each target file to ``pending`` with a fresh ``file_version`` and
        flips its manifest row back to ``ingesting``. Returns ``[(file_id,
        version)]`` for the CALLER to enqueue AFTER commit (so the worker never
        reads an uncommitted row). Only documents that already have an ingested
        file are targeted; docs that never fetched need a full sync instead.
        """
        stmt = (
            select(DataSourceDocumentEntity, KbFileEntity)
            .join(KbFileEntity, DataSourceDocumentEntity.file_id == KbFileEntity.id)
            .where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
        )
        if only_unfinished:
            stmt = stmt.where(KbFileEntity.status != FileStatus.succeeded)

        rows = (await self.session.exec(stmt)).all()
        now = _utcnow()
        version = int(time.time())
        out: List[Tuple[str, int]] = []
        for doc, file_entity in rows:
            file_entity.status = FileStatus.pending
            file_entity.file_version = version
            file_entity.failed_reason = None
            file_entity.updated_at = now
            doc.doc_status = DataSourceDocStatus.ingesting
            doc.last_error = None
            doc.updated_at = now
            self.session.add(file_entity)
            self.session.add(doc)
            out.append((file_entity.id, version))

        if out:
            datasource = await self.get_datasource(ds_id, tenant_id)
            if datasource:
                datasource.status = DataSourceStatus.ingesting
                datasource.last_error = None
                datasource.updated_at = now
                self.session.add(datasource)

        await self.session.flush()
        return out

    # ------------------------------------------------------------------
    # catalog tool — metadata-layer search over the manifest (no body reads)
    # ------------------------------------------------------------------
    @staticmethod
    def _catalog_row(r: DataSourceDocumentEntity, score: Optional[float]) -> dict:
        return {
            "doc_id": r.doc_id,
            "title": r.title,
            "path": r.path,
            "product": r.product,
            "section": r.section,
            "lang": r.lang,
            "source_url": r.source_url,
            "score": score,
        }

    async def catalog_search(
        self, kb_id: str, tenant_id: str, query: Optional[str] = None,
        product: Optional[str] = None, section: Optional[str] = None,
        lang: Optional[str] = None, limit: int = 20,
    ) -> List[dict]:
        """Find/browse documents by metadata across the KB's data sources.

        Runs entirely on the manifest (no body reads). With ``query`` it ranks by
        fuzzy match over title > path > summary; without it, it lists/browses.
        """
        conditions = [
            DataSourceDocumentEntity.kb_id == kb_id,
            DataSourceDocumentEntity.tenant_id == tenant_id,
            DataSourceDocumentEntity.doc_status != DataSourceDocStatus.deleted,
        ]
        if product:
            conditions.append(DataSourceDocumentEntity.product == product)
        if section:
            conditions.append(DataSourceDocumentEntity.section == section)
        if lang:
            conditions.append(DataSourceDocumentEntity.lang == lang)
        where = and_(*conditions)

        if not query:
            rows = (await self.session.exec(
                select(DataSourceDocumentEntity)
                .where(where)
                .order_by(
                    DataSourceDocumentEntity.product,
                    DataSourceDocumentEntity.section,
                    DataSourceDocumentEntity.title,
                )
                .limit(limit)
            )).all()
            return [self._catalog_row(r, None) for r in rows]

        # query → fuzzy rank over a bounded candidate set
        CAP = 5000
        candidates = (await self.session.exec(
            select(DataSourceDocumentEntity)
            .where(where)
            .order_by(DataSourceDocumentEntity.updated_at.desc())
            .limit(CAP)
        )).all()

        from rapidfuzz import fuzz
        q = query.lower()
        scored = []
        for r in candidates:
            title = (r.title or "").lower()
            path = (r.path or "").lower()
            summary = (r.summary or "").lower()
            s = max(
                fuzz.token_set_ratio(q, title),          # title weighted highest
                0.8 * fuzz.partial_ratio(q, path),
                0.6 * fuzz.token_set_ratio(q, summary),
            )
            scored.append((s, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._catalog_row(r, round(s / 100, 4)) for s, r in scored[:limit]]

    async def get_document_row(
        self, ds_id: str, doc_id: str, tenant_id: str
    ) -> Optional[DataSourceDocumentEntity]:
        """Fetch a single manifest row entity (for in-session updates by the worker)."""
        result = await self.session.exec(
            select(DataSourceDocumentEntity).where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.doc_id == doc_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    async def get_document_row_by_source_id(
        self, ds_id: str, source_id: str, tenant_id: str
    ) -> Optional[DataSourceDocumentEntity]:
        """Fetch a manifest row by its stable upstream identity."""
        result = await self.session.exec(
            select(DataSourceDocumentEntity).where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.source_id == source_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    async def get_document(
        self, ds_id: str, doc_id: str, tenant_id: str
    ) -> Optional[dict]:
        result = await self.session.exec(
            select(DataSourceDocumentEntity, KbFileEntity.status, KbFileEntity.failed_reason)
            .outerjoin(KbFileEntity, DataSourceDocumentEntity.file_id == KbFileEntity.id)
            .where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.doc_id == doc_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
        )
        row = result.first()
        if not row:
            return None
        doc, parse_status, failed_reason = row
        d = doc.model_dump(mode="json")
        d["parse_status"] = parse_status
        d["parse_failed_reason"] = failed_reason
        return d

    # ------------------------------------------------------------------
    # Sync runs (history)
    # ------------------------------------------------------------------
    async def list_sync_runs(
        self, ds_id: str, tenant_id: str, page: int = 1, size: int = 20
    ) -> PagedResult[List[dict]]:
        where_condition = and_(
            DataSourceSyncRunEntity.tenant_id == tenant_id,
            DataSourceSyncRunEntity.datasource_id == ds_id,
        )
        total = (await self.session.exec(
            select(func.count(DataSourceSyncRunEntity.id)).where(where_condition)
        )).one_or_none() or 0
        offset = (page - 1) * size
        results = await self.session.exec(
            select(DataSourceSyncRunEntity)
            .where(where_condition)
            .order_by(DataSourceSyncRunEntity.started_at.desc())
            .offset(offset)
            .limit(size)
        )
        items = [r.model_dump(mode="json") for r in results.all()]
        pages = (total + size - 1) // size if total > 0 else 0
        return PagedResult(items=items, total=total, pages=pages, page=page, size=size)

    async def get_sync_run(self, run_id: str, tenant_id: str) -> Optional[DataSourceSyncRunEntity]:
        result = await self.session.exec(
            select(DataSourceSyncRunEntity).where(
                DataSourceSyncRunEntity.id == run_id,
                DataSourceSyncRunEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    # ------------------------------------------------------------------
    # Aggregate sync status (live, computed from documents)
    # ------------------------------------------------------------------
    async def get_sync_status(self, ds_id: str, tenant_id: str) -> dict:
        """Return the data source's current sync status + per-document breakdown.

        Combines the stored aggregate ``status`` (phase A) with a live count of
        document parse states (phase B), so the UI can show
        "synced, N parsing / M failed".
        """
        datasource = await self.get_datasource(ds_id, tenant_id)
        if not datasource:
            raise ValueError(f"Data source '{ds_id}' does not exist.")

        # per-doc_status counts
        rows = await self.session.exec(
            select(DataSourceDocumentEntity.doc_status, func.count(DataSourceDocumentEntity.id))
            .where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
            .group_by(DataSourceDocumentEntity.doc_status)
        )
        doc_status_counts = {status: int(count) for status, count in rows.all()}

        synced = doc_status_counts.get(DataSourceDocStatus.synced, 0)
        failed = doc_status_counts.get(DataSourceDocStatus.failed, 0)
        ingesting = doc_status_counts.get(DataSourceDocStatus.ingesting, 0)
        total_docs = sum(doc_status_counts.values())

        return {
            "datasource_id": ds_id,
            "status": datasource.status,
            "enabled": datasource.enabled,
            "doc_count": datasource.doc_count,
            "last_sync_at": datasource.last_sync_at.isoformat() + "Z"
            if datasource.last_sync_at else None,
            "last_sync_finished_at": datasource.last_sync_finished_at.isoformat() + "Z"
            if datasource.last_sync_finished_at else None,
            "last_sync_duration_ms": datasource.last_sync_duration_ms,
            "last_sync_report": datasource.last_sync_report,
            "last_error": datasource.last_error,
            "next_sync_at": datasource.next_sync_at.isoformat() + "Z"
            if datasource.next_sync_at else None,
            "doc_status_counts": doc_status_counts,
            "total_documents": total_docs,
            "synced": synced,
            "ingesting": ingesting,
            "failed": failed,
        }

    async def refresh_aggregate_status(self, ds_id: str, tenant_id: str) -> DataSourceEntity:
        """Recompute ``DataSourceEntity.status`` from document states (phase B).

        Called after files settle (on_file_settled, wired in PR4) or on read.
        Does not touch the syncing/failed phase-A states set by the worker.
        """
        datasource = await self.get_datasource(ds_id, tenant_id)
        if not datasource:
            raise ValueError(f"Data source '{ds_id}' does not exist.")

        # phase-A / user-owned terminal states aren't recomputed from docs here
        if datasource.status in (DataSourceStatus.syncing, DataSourceStatus.failed, DataSourceStatus.cancelled):
            return datasource

        rows = await self.session.exec(
            select(DataSourceDocumentEntity.doc_status, func.count(DataSourceDocumentEntity.id))
            .where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
            .group_by(DataSourceDocumentEntity.doc_status)
        )
        counts = {status: int(count) for status, count in rows.all()}
        ingesting = counts.get(DataSourceDocStatus.ingesting, 0) + counts.get(DataSourceDocStatus.fetching, 0)
        failed = counts.get(DataSourceDocStatus.failed, 0)

        if ingesting > 0:
            datasource.status = DataSourceStatus.ingesting
        elif failed > 0:
            datasource.status = DataSourceStatus.partial
        else:
            datasource.status = DataSourceStatus.succeeded

        datasource.updated_at = _utcnow()
        self.session.add(datasource)
        await self.session.flush()
        await self.session.refresh(datasource)
        return datasource

    # ------------------------------------------------------------------
    # Sync orchestration helpers (used by the sync worker)
    # ------------------------------------------------------------------
    async def begin_sync(
        self, ds_id: str, tenant_id: str, trigger: str = SyncTrigger.manual,
        triggered_by: Optional[str] = None,
    ):
        """Atomically claim a data source for syncing and open a sync run.

        Returns ``(datasource, run)`` on success, or ``(None, None)`` if the data
        source is already syncing (concurrent/duplicate trigger) or missing. The
        claim is a conditional UPDATE so only one of N racing tasks wins.

        If the API endpoint already set status to ``syncing`` before enqueuing
        (for immediate UI feedback), we accept that as a valid claim and proceed.
        """
        now = _utcnow()
        datasource = await self.get_datasource(ds_id, tenant_id)
        if datasource is None:
            logger.warning(
                f"[datasource-sync] begin_sync: datasource not found ds_id={ds_id} tenant={tenant_id}"
            )
            return None, None

        # If the API already set status to syncing, accept it — the API already
        # guarded against concurrent triggers. Otherwise, try an atomic claim.
        if datasource.status == DataSourceStatus.syncing:
            logger.info(
                f"[datasource-sync] begin_sync: status already syncing (set by API) "
                f"ds_id={ds_id} tenant={tenant_id}, proceeding"
            )
        else:
            # atomic compare-and-set: flip to syncing only if not already syncing
            result = await self.session.execute(
                update(DataSourceEntity)
                .where(
                    DataSourceEntity.id == ds_id,
                    DataSourceEntity.tenant_id == tenant_id,
                    DataSourceEntity.status != DataSourceStatus.syncing,
                )
                .values(status=DataSourceStatus.syncing, last_sync_at=now, last_error=None, updated_at=now)
            )
            if result.rowcount == 0:
                # already syncing or does not exist — do not start a second run
                logger.warning(
                    f"[datasource-sync] begin_sync: atomic claim FAILED for ds_id={ds_id} "
                    f"tenant={tenant_id} — already syncing (race condition)"
                )
                return None, None
            logger.info(
                f"[datasource-sync] begin_sync: atomic claim SUCCESS for ds_id={ds_id} tenant={tenant_id}"
            )
            # Refresh after the UPDATE
            datasource = await self.get_datasource(ds_id, tenant_id)

        run = DataSourceSyncRunEntity(
            tenant_id=tenant_id,
            datasource_id=ds_id,
            kb_id=datasource.kb_id,
            trigger=trigger,
            triggered_by=triggered_by,
            status=SyncRunStatus.running,
            started_at=now,
        )
        self.session.add(run)
        await self.session.flush()
        await self.session.refresh(run)
        return datasource, run

    async def get_manifest_map(self, ds_id: str, tenant_id: str) -> dict:
        """source_id -> manifest row for source-level incremental diffing."""
        results = await self.session.exec(
            select(DataSourceDocumentEntity).where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
        )
        return {row.source_id: row for row in results.all()}

    async def upsert_document(
        self, ds_id: str, kb_id: str, tenant_id: str, source_doc,
        doc_id: str, doc_status: str = DataSourceDocStatus.ingesting,
        existing: Optional[DataSourceDocumentEntity] = None,
        changed: bool = True,
    ) -> DataSourceDocumentEntity:
        """Create or update a manifest row from a SourceDocument. Caller commits."""
        now = _utcnow()
        row = existing
        if row is None:
            row = DataSourceDocumentEntity(
                tenant_id=tenant_id,
                datasource_id=ds_id,
                kb_id=kb_id,
                source_id=source_doc.source_id,
                doc_id=doc_id,
                file_id=doc_id,
                first_seen_at=now,
            )
        row.source_id = source_doc.source_id
        row.doc_id = doc_id
        row.path = source_doc.path
        row.file_id = doc_id
        row.source_url = source_doc.source_url
        row.fetch_url = source_doc.fetch_url
        row.title = source_doc.title
        row.section = source_doc.section
        row.product = source_doc.product
        row.summary = source_doc.summary
        row.lang = source_doc.lang
        row.content_hash = source_doc.content_hash
        row.byte_size = source_doc.byte_size
        row.source_meta = source_doc.source_meta or {}
        row.doc_status = doc_status
        row.last_error = None
        row.last_fetched_at = now
        if changed:
            row.last_changed_at = now
        row.updated_at = now
        self.session.add(row)
        await self.session.flush()
        return row

    async def mark_document_failed(
        self, ds_id: str, kb_id: str, tenant_id: str, source_id: str,
        doc_id: str, error: str,
        existing: Optional[DataSourceDocumentEntity] = None,
        path: Optional[str] = None,
    ) -> DataSourceDocumentEntity:
        """Record a fetch/ingest failure for a document. Caller commits."""
        now = _utcnow()
        row = existing
        if row is None:
            row = DataSourceDocumentEntity(
                tenant_id=tenant_id, datasource_id=ds_id, kb_id=kb_id,
                source_id=source_id, doc_id=doc_id, file_id=doc_id,
                path=path or source_id, first_seen_at=now,
            )
        row.source_id = source_id
        row.doc_id = doc_id
        row.file_id = doc_id
        row.doc_status = DataSourceDocStatus.failed
        row.last_error = error
        row.last_fetched_at = now
        row.updated_at = now
        self.session.add(row)
        await self.session.flush()
        return row

    async def delete_document_row(self, row: DataSourceDocumentEntity) -> None:
        await self.session.delete(row)
        await self.session.flush()

    async def reset_manifest(self, ds_id: str, tenant_id: str) -> int:
        """Delete ALL manifest rows for a datasource.

        After this, the next sync will see every document as 'added' and
        re-ingest from scratch. Use this to recover from a stale manifest
        (e.g. after manual file deletions).

        Returns the number of rows deleted.
        """
        stmt = delete(DataSourceDocumentEntity).where(
            DataSourceDocumentEntity.datasource_id == ds_id,
            DataSourceDocumentEntity.tenant_id == tenant_id,
        )
        result = await self.session.exec(stmt)
        await self.session.flush()
        deleted = result.rowcount
        logger.info(
            f"[datasource-sync] Reset manifest for ds_id={ds_id}: "
            f"deleted {deleted} row(s)"
        )
        return deleted

    async def delete_document_rows_by_file_id(self, kb_id: str, file_id: str, tenant_id: str) -> int:
        """Delete all manifest rows for a given file_id.

        Called when a user manually deletes a file from the KB so the sync
        manifest stays in sync — otherwise the next sync sees the doc as
        unchanged and skips re-ingestion.

        Returns the number of rows deleted.
        """
        stmt = delete(DataSourceDocumentEntity).where(
            DataSourceDocumentEntity.kb_id == kb_id,
            DataSourceDocumentEntity.file_id == file_id,
            DataSourceDocumentEntity.tenant_id == tenant_id,
        )
        result = await self.session.exec(stmt)
        await self.session.flush()
        deleted = result.rowcount
        if deleted:
            logger.info(
                f"[datasource-sync] Deleted {deleted} manifest row(s) "
                f"for kb_id={kb_id} file_id={file_id}"
            )
        return deleted

    async def finalize_sync(
        self, ds_id: str, tenant_id: str, run_id: str, counts: dict,
        report: dict, error: Optional[str] = None, final_status: Optional[str] = None,
    ) -> None:
        """Close the sync run (phase A) and set the data source aggregate state. Caller commits."""
        now = _utcnow()
        datasource = await self.get_datasource(ds_id, tenant_id)
        run = await self.get_sync_run(run_id, tenant_id)
        if run is None or datasource is None:
            return

        started = run.started_at or now
        duration_ms = int((now - started).total_seconds() * 1000)

        n_added = counts.get("added", 0)
        n_updated = counts.get("updated", 0)
        n_deleted = counts.get("deleted", 0)
        n_unchanged = counts.get("unchanged", 0)
        n_failed = counts.get("failed", 0)
        n_discovered = counts.get("discovered", 0)

        run.finished_at = now
        run.duration_ms = duration_ms
        run.n_discovered = n_discovered
        run.n_added = n_added
        run.n_updated = n_updated
        run.n_deleted = n_deleted
        run.n_unchanged = n_unchanged
        run.n_failed = n_failed
        run.report = report
        run.error = error
        if error:
            run.status = SyncRunStatus.failed
        elif n_failed > 0:
            run.status = SyncRunStatus.partial
        else:
            run.status = SyncRunStatus.succeeded
        self.session.add(run)

        # If the user cancelled at any point during this run, keep the cancelled
        # state — don't let finalize flip it back to ingesting/succeeded.
        if final_status is None and datasource.status == DataSourceStatus.cancelled:
            final_status = DataSourceStatus.cancelled

        # data source aggregate (two-phase): just-enqueued docs are still parsing
        if final_status:
            datasource.status = final_status  # e.g. cancelled (caller-owned terminal state)
        elif error:
            datasource.status = DataSourceStatus.failed
        elif (n_added + n_updated) > 0:
            datasource.status = DataSourceStatus.ingesting
        elif n_failed > 0:
            datasource.status = DataSourceStatus.partial
        else:
            datasource.status = DataSourceStatus.succeeded

        total_docs = (await self.session.exec(
            select(func.count(DataSourceDocumentEntity.id)).where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
        )).one_or_none() or 0

        datasource.doc_count = int(total_docs)
        datasource.last_sync_finished_at = now
        datasource.last_sync_duration_ms = duration_ms
        datasource.last_sync_report = report
        datasource.last_error = error
        datasource.updated_at = now
        self.session.add(datasource)
        await self.session.flush()
        logger.info(
            f"[datasource-sync] finalize_sync: ds_id={ds_id} run_id={run_id} "
            f"status={datasource.status} added={n_added} updated={n_updated} "
            f"deleted={n_deleted} unchanged={n_unchanged} failed={n_failed} "
            f"error={error}"
        )

    async def reconcile_document_statuses(self, ds_id: str, tenant_id: str) -> int:
        """Phase B: advance ingesting docs to synced/failed based on KbFileEntity.status.

        Read-time reconciliation — avoids coupling the core file pipeline to data
        sources. Returns the number of rows transitioned.
        """
        results = await self.session.exec(
            select(DataSourceDocumentEntity, KbFileEntity.status)
            .join(KbFileEntity, DataSourceDocumentEntity.file_id == KbFileEntity.id)
            .where(
                DataSourceDocumentEntity.datasource_id == ds_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
                DataSourceDocumentEntity.doc_status == DataSourceDocStatus.ingesting,
            )
        )
        changed = 0
        now = _utcnow()
        for row, file_status in results.all():
            if file_status == FileStatus.succeeded:
                row.doc_status = DataSourceDocStatus.synced
            elif file_status == FileStatus.failed:
                row.doc_status = DataSourceDocStatus.failed
            else:
                continue
            row.updated_at = now
            self.session.add(row)
            changed += 1
        if changed:
            await self.session.flush()
        return changed
