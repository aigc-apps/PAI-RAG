### DataSource configuration & management API ###
#
# Mounted under /v1/config/knowledgebases, so all paths are nested as
#   /{kb_id}/datasources[/{ds_id}][/...]
#
# A data source belongs to exactly one KB. This router exposes CRUD plus the
# management-state surfaces: the document list ("file list"), sync runs
# (history) and the aggregate sync status. The actual /sync trigger that
# enqueues the Celery sync task is wired in PR4.

from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

from db.db_context import get_db_session
from db.models.knowledgebase.datasource import (
    DataSourceEntity,
    DataSourceCreate,
    DataSourceUpdate,
)
from common.knowledgebase.types import DataSourceStatus
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException, handle_api_exceptions
from service.injection import (
    get_datasource_service,
    get_file_service,
    get_knowledgebase_service,
    get_rag_service,
    get_tenant_id,
)
from service.knowledgebase.datasource_service import DataSourceService
from service.knowledgebase.file_service import FileService
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from service.knowledgebase.rag_service import RagService

datasource_router = APIRouter()


async def _get_owned_datasource(
    kb_id: str,
    ds_id: str,
    tenant_id: str,
    datasource_service: DataSourceService,
) -> DataSourceEntity:
    """Fetch a data source and assert it belongs to the given KB + tenant."""
    datasource = await datasource_service.get_datasource(ds_id=ds_id, tenant_id=tenant_id)
    if not datasource or datasource.kb_id != kb_id:
        raise ApiException.not_found(resource_id=ds_id, resource_type="DataSource")
    return datasource


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
@datasource_router.post("/{kb_id}/datasources", response_model=ResponseModel[DataSourceEntity])
@handle_api_exceptions(action="create data source")
async def create_datasource(
    kb_id: str,
    ds_data: DataSourceCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
    knowledgebase_service: KnowledgebaseService = Depends(get_knowledgebase_service),
):
    kb = await knowledgebase_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not kb:
        raise ApiException.not_found(resource_id=kb_id, resource_type="Knowledgebase")

    datasource = await datasource_service.create_datasource(
        kb_id=kb_id, ds_data=ds_data, tenant_id=tenant_id
    )
    await session.commit()
    await session.refresh(datasource)
    return success_response(data=datasource, message="Data source created successfully.")


@datasource_router.get("/{kb_id}/datasources", response_model=ResponseModel[dict])
@handle_api_exceptions(action="list data sources")
async def list_datasources(
    kb_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, le=1000),
    query: Optional[str] = Query(default=None),
    tenant_id: str = Depends(get_tenant_id),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    result = await datasource_service.list_datasources(
        kb_id=kb_id, tenant_id=tenant_id, page=page, size=size, query=query
    )
    return success_response(data=result, message="List data sources success.")


@datasource_router.get("/{kb_id}/datasources/{ds_id}", response_model=ResponseModel[DataSourceEntity])
@handle_api_exceptions(action="get data source")
async def get_datasource(
    kb_id: str,
    ds_id: str,
    tenant_id: str = Depends(get_tenant_id),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    datasource = await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    return success_response(data=datasource, message="Get data source success.")


@datasource_router.put("/{kb_id}/datasources/{ds_id}", response_model=ResponseModel[DataSourceEntity])
@handle_api_exceptions(action="update data source")
async def update_datasource(
    kb_id: str,
    ds_id: str,
    update_data: DataSourceUpdate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    datasource = await datasource_service.update_datasource(
        ds_id=ds_id, update_data=update_data, tenant_id=tenant_id
    )
    await session.commit()
    await session.refresh(datasource)
    return success_response(data=datasource, message="Update data source success.")


@datasource_router.delete("/{kb_id}/datasources/{ds_id}", response_model=ResponseModel[dict])
@handle_api_exceptions(action="delete data source")
async def delete_datasource(
    kb_id: str,
    ds_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
    rag_service: RagService = Depends(get_rag_service),
    file_service: FileService = Depends(get_file_service),
):
    datasource = await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    # Refuse to delete while a sync is in flight: the running worker keeps
    # committing new KB files after our snapshot, which would be orphaned.
    # Caller should cancel (or wait) first.
    if datasource.status == DataSourceStatus.syncing:
        raise ApiException(
            code=409,
            message="Data source is syncing; cancel or wait for it to finish before deleting.",
        )
    # Delete each ingested KB file (chunks + vectors), committing PER FILE.
    # delete_file removes the external vector immediately (non-transactional), so we
    # must commit each file's DB deletion as we go — a single batch transaction would,
    # on a later failure, roll back earlier files' DB rows while their vectors are
    # already gone (orphaned, un-recallable). Per-file commit keeps DB and vector
    # store consistent; on real failure we stop and the data source is kept, so a
    # retry resumes (delete_file is idempotent for already-gone files).
    file_ids = await datasource_service.list_document_file_ids(ds_id=ds_id, tenant_id=tenant_id)
    failed_file_id = None
    failed_err = None
    for file_id in file_ids:
        # Skip files that are already gone (idempotent retry). Check existence
        # EXPLICITLY rather than pattern-matching exception strings — "not found"
        # also appears in config errors ("VectorDB config not found", "Embedding
        # model not found", "Vector table name not found"), and swallowing those
        # would skip real cleanup and orphan KB files/vectors.
        existing = await file_service.get_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
        if existing is None:
            continue
        try:
            await rag_service.delete_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
            await session.commit()
        except Exception as e:  # noqa: BLE001
            await session.rollback()
            logger.warning(f"Failed to delete file {file_id} of datasource {ds_id}: {e}")
            failed_file_id, failed_err = file_id, str(e)
            break  # stop on first real failure; deleted-so-far are durably committed
    if failed_file_id:
        raise ApiException(
            code=500,
            message=(
                "Failed to delete an ingested file; data source kept. Already-deleted "
                "files are removed. Please retry to resume."
            ),
            data={"failed_file_id": failed_file_id, "error": failed_err},
        )
    await datasource_service.delete_datasource(ds_id=ds_id, tenant_id=tenant_id)
    await session.commit()
    return success_response(data={"id": ds_id}, message="Delete data source success.")


@datasource_router.post("/{kb_id}/datasources/{ds_id}/enable", response_model=ResponseModel[DataSourceEntity])
@handle_api_exceptions(action="enable data source")
async def enable_datasource(
    kb_id: str,
    ds_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    datasource = await datasource_service.set_enabled(ds_id=ds_id, enabled=True, tenant_id=tenant_id)
    await session.commit()
    await session.refresh(datasource)
    return success_response(data=datasource, message="Data source enabled.")


@datasource_router.post("/{kb_id}/datasources/{ds_id}/disable", response_model=ResponseModel[DataSourceEntity])
@handle_api_exceptions(action="disable data source")
async def disable_datasource(
    kb_id: str,
    ds_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    datasource = await datasource_service.set_enabled(ds_id=ds_id, enabled=False, tenant_id=tenant_id)
    await session.commit()
    await session.refresh(datasource)
    return success_response(data=datasource, message="Data source disabled.")


# ---------------------------------------------------------------------------
# Sync trigger
# ---------------------------------------------------------------------------
@datasource_router.post("/{kb_id}/datasources/{ds_id}/sync", response_model=ResponseModel[dict])
@handle_api_exceptions(action="trigger data source sync")
async def sync_datasource(
    kb_id: str,
    ds_id: str,
    tenant_id: str = Depends(get_tenant_id),
    user_id: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    datasource = await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    # Best-effort guard against duplicate clicks; the worker's atomic claim
    # (begin_sync) is the real concurrency protection if this races.
    if datasource.status == DataSourceStatus.syncing:
        return success_response(
            data={"datasource_id": ds_id, "status": "already_syncing"},
            message="A sync is already in progress for this data source.",
        )
    # Set status to syncing BEFORE enqueuing the Celery task so the frontend
    # polling sees "syncing" immediately and shows a progress bar. Without this,
    # if the worker is down, /sync-status recomputes the aggregate from doc
    # statuses and flips back to "succeeded" within the first poll, making the
    # UI flash "done" before anything actually runs.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    datasource.status = DataSourceStatus.syncing
    datasource.last_sync_at = now
    datasource.last_error = None
    datasource.updated_at = now
    session.add(datasource)
    await session.commit()
    await session.refresh(datasource)
    # The Celery task opens its own sync run (phase A) and ingests via the
    # existing file pipeline. Client polls /sync-status and /sync-runs.
    import app.worker as background_worker
    task = background_worker.sync_datasource.delay(
        datasource_id=ds_id, tenant_id=tenant_id, trigger="manual", triggered_by=user_id,
    )
    logger.info(
        f"[datasource-sync] Enqueued Celery task {task.id} for datasource={ds_id} "
        f"kb={kb_id} tenant={tenant_id} source_type={datasource.source_type}"
    )
    return success_response(
        data={"datasource_id": ds_id, "status": "accepted", "task_id": task.id},
        message="Data source sync triggered.",
    )


@datasource_router.post("/{kb_id}/datasources/{ds_id}/cancel", response_model=ResponseModel[dict])
@handle_api_exceptions(action="cancel data source sync")
async def cancel_datasource_sync(
    kb_id: str,
    ds_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    """Cancel an in-progress sync. Unfinished files are marked 'cancelled' and
    running parse tasks stop cooperatively; a later sync re-ingests them."""
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    cancelled = await datasource_service.cancel_sync(ds_id=ds_id, tenant_id=tenant_id)
    await session.commit()
    return success_response(
        data={"datasource_id": ds_id, "cancelled": cancelled},
        message=f"Cancelled {cancelled} in-flight document(s).",
    )


@datasource_router.post("/{kb_id}/datasources/{ds_id}/reset", response_model=ResponseModel[dict])
@handle_api_exceptions(action="reset data source sync progress")
async def reset_datasource_sync_progress(
    kb_id: str,
    ds_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    """Reset sync progress — clear the document manifest so the next sync
    re-discovers and re-ingests ALL documents from scratch.

    Use this when:
    - Documents were manually deleted from the KB and the manifest is stale
    - You want to force a full re-sync regardless of content changes
    """
    datasource = await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    if datasource.status == DataSourceStatus.syncing:
        return success_response(
            data={"datasource_id": ds_id, "status": "already_syncing"},
            message="A sync is in progress; try reset after it finishes.",
        )
    deleted = await datasource_service.reset_manifest(ds_id=ds_id, tenant_id=tenant_id)
    await session.commit()
    return success_response(
        data={"datasource_id": ds_id, "deleted_manifest_rows": deleted},
        message=f"Sync progress reset. {deleted} manifest row(s) cleared. "
                f"Trigger a sync to re-ingest all documents.",
    )


@datasource_router.post("/{kb_id}/datasources/{ds_id}/reparse", response_model=ResponseModel[dict])
@handle_api_exceptions(action="reparse data source documents")
async def reparse_datasource(
    kb_id: str,
    ds_id: str,
    scope: str = Query(default="unfinished", description="'unfinished' (default) or 'all'"),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    """Re-run parsing for already-fetched files (no re-fetch). Use to recover
    documents stuck in 'ingesting'/'failed'."""
    datasource = await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    if datasource.status == DataSourceStatus.syncing:
        return success_response(
            data={"datasource_id": ds_id, "status": "already_syncing"},
            message="A sync is in progress; try reparse after it finishes.",
        )
    pairs = await datasource_service.reparse_documents(
        ds_id=ds_id, tenant_id=tenant_id, only_unfinished=(scope != "all"),
    )
    # Commit BEFORE enqueuing so the worker never reads an uncommitted file row.
    await session.commit()
    import app.worker as background_worker
    for file_id, version in pairs:
        background_worker.enqueue_file_tasks.delay(
            file_id, version, is_attachment=False, tenant_id=tenant_id,
        )
    return success_response(
        data={"datasource_id": ds_id, "reparsed": len(pairs)},
        message=f"Re-parsing {len(pairs)} document(s).",
    )


# ---------------------------------------------------------------------------
# catalog tool — metadata search across the KB's data sources
# ---------------------------------------------------------------------------
@datasource_router.get("/{kb_id}/catalog", response_model=ResponseModel[dict])
@handle_api_exceptions(action="catalog search")
async def catalog_search(
    kb_id: str,
    query: Optional[str] = Query(default=None),
    product: Optional[str] = Query(default=None),
    section: Optional[str] = Query(default=None),
    lang: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    tenant_id: str = Depends(get_tenant_id),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    """Find/browse documents by metadata across the KB's data sources (no body reads)."""
    results = await datasource_service.catalog_search(
        kb_id=kb_id, tenant_id=tenant_id, query=query,
        product=product, section=section, lang=lang, limit=limit,
    )
    return success_response(data={"results": results, "total": len(results)}, message="Catalog search success.")


@datasource_router.get("/{kb_id}/keyword", response_model=ResponseModel[dict])
@handle_api_exceptions(action="keyword search")
async def keyword_search(
    kb_id: str,
    pattern: str = Query(...),
    doc_id: Optional[str] = Query(default=None),
    path_prefix: Optional[str] = Query(default=None),
    datasource: Optional[str] = Query(default=None),
    context: int = Query(default=2, ge=0, le=10),
    limit: int = Query(default=20, ge=1, le=200),
    tenant_id: str = Depends(get_tenant_id),
    rag_service: RagService = Depends(get_rag_service),
):
    """Literal keyword grep over the whole KB (line numbers + context).

    ``doc_id`` / ``path_prefix`` / ``datasource`` narrow to specific
    data-source documents when supplied.
    """
    out = await rag_service.keyword_search(
        kb_id=kb_id, tenant_id=tenant_id, pattern=pattern,
        doc_id=doc_id, path_prefix=path_prefix, datasource=datasource,
        context=context, limit=limit,
    )
    return success_response(data=out, message="Keyword search success.")


# ---------------------------------------------------------------------------
# Management state / observability
# ---------------------------------------------------------------------------
@datasource_router.get("/{kb_id}/datasources/{ds_id}/documents", response_model=ResponseModel[dict])
@handle_api_exceptions(action="list data source documents")
async def list_datasource_documents(
    kb_id: str,
    ds_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, le=1000),
    doc_status: Optional[str] = Query(default=None),
    query: Optional[str] = Query(default=None),
    tenant_id: str = Depends(get_tenant_id),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    result = await datasource_service.list_documents(
        ds_id=ds_id, tenant_id=tenant_id, page=page, size=size,
        doc_status=doc_status, query=query,
    )
    return success_response(data=result, message="List data source documents success.")


@datasource_router.get("/{kb_id}/datasources/{ds_id}/document", response_model=ResponseModel[dict])
@handle_api_exceptions(action="get data source document")
async def get_datasource_document(
    kb_id: str,
    ds_id: str,
    doc_id: str = Query(..., description="doc_id (may contain slashes)"),
    tenant_id: str = Depends(get_tenant_id),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    doc = await datasource_service.get_document(ds_id=ds_id, doc_id=doc_id, tenant_id=tenant_id)
    if not doc:
        raise ApiException.not_found(resource_id=doc_id, resource_type="DataSourceDocument")
    return success_response(data=doc, message="Get data source document success.")


@datasource_router.get("/{kb_id}/datasources/{ds_id}/sync-runs", response_model=ResponseModel[dict])
@handle_api_exceptions(action="list sync runs")
async def list_sync_runs(
    kb_id: str,
    ds_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    result = await datasource_service.list_sync_runs(
        ds_id=ds_id, tenant_id=tenant_id, page=page, size=size
    )
    return success_response(data=result, message="List sync runs success.")


@datasource_router.get("/{kb_id}/datasources/{ds_id}/sync-runs/{run_id}", response_model=ResponseModel[dict])
@handle_api_exceptions(action="get sync run")
async def get_sync_run(
    kb_id: str,
    ds_id: str,
    run_id: str,
    tenant_id: str = Depends(get_tenant_id),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    run = await datasource_service.get_sync_run(run_id=run_id, tenant_id=tenant_id)
    if not run or run.datasource_id != ds_id:
        raise ApiException.not_found(resource_id=run_id, resource_type="DataSourceSyncRun")
    return success_response(data=run, message="Get sync run success.")


@datasource_router.get("/{kb_id}/datasources/{ds_id}/sync-status", response_model=ResponseModel[dict])
@handle_api_exceptions(action="get data source sync status")
async def get_sync_status(
    kb_id: str,
    ds_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    datasource_service: DataSourceService = Depends(get_datasource_service),
):
    await _get_owned_datasource(kb_id, ds_id, tenant_id, datasource_service)
    # Phase B lazy reconciliation: advance ingesting docs to synced/failed from
    # KbFileEntity.status, then recompute the aggregate, before reporting.
    await datasource_service.reconcile_document_statuses(ds_id=ds_id, tenant_id=tenant_id)
    await datasource_service.refresh_aggregate_status(ds_id=ds_id, tenant_id=tenant_id)
    await session.commit()
    status = await datasource_service.get_sync_status(ds_id=ds_id, tenant_id=tenant_id)
    return success_response(data=status, message="Get data source sync status success.")
