"""Data source incremental sync orchestrator.

One run = one incremental sync (spec §4). Flow:

  1. begin: mark the data source ``syncing`` and open a sync run (phase A).
  2. discover: list current documents (network).
  3. diff vs manifest: added = new ids, deleted = missing ids, the intersection
     is fetched to compare ``content_hash`` (no ETag on these sources).
  4. fetch changed bodies (concurrently) → emit → ingest via the EXISTING KB
     pipeline (write to file_store, create/update KbFileEntity, enqueue_file_tasks).
  5. delete removed docs via RagService.delete_file.
  6. finalize: write run counters + report, set the aggregate status.

No new parse/chunk/embed code — ingestion reuses the upload pipeline entirely.

Dependency seams (``adapter``/``enqueue_fn``/``file_writer``/``rag_service_factory``)
let tests drive the whole flow without network, Celery or a vector store.
"""

import io
import os
import time
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from sqlmodel import select

from db.db_context import create_db_session
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.datasource import DataSourceDocumentEntity
from common.knowledgebase.types import FileStatus, DataSourceDocStatus, DataSourceStatus, SyncTrigger
from service.knowledgebase.datasource_service import DataSourceService
from rag.datasource.registry import get_adapter

_MAX_REPORTED_ERRORS = 50


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _is_missing_file_error(e: Exception) -> bool:
    """True when delete_file failed only because the file is already gone (idempotent)."""
    return "does not exist" in str(e).lower() or "not found" in str(e).lower()


async def _is_cancelled(datasource_id: str, tenant_id: str) -> bool:
    """Re-read the data source's status to detect a concurrent user cancel."""
    async with create_db_session() as session:
        ds = await DataSourceService(session).get_datasource(datasource_id, tenant_id)
        return bool(ds and ds.status == DataSourceStatus.cancelled)


# -- default (production) dependency implementations ------------------------
async def _default_file_writer(content: bytes, file_name: str, dest_path: str, tenant_id: str) -> str:
    from pairag.file.store.file_store_helper import file_store
    result = await file_store.write_async(
        file=io.BytesIO(content), file_name=file_name, file_path=dest_path, tenant_id=tenant_id
    )
    return result.file_path


def _default_enqueue(file_id: str, file_version: int, tenant_id: str) -> None:
    from app.worker import enqueue_file_tasks
    enqueue_file_tasks.delay(file_id, file_version, is_attachment=False, tenant_id=tenant_id)


async def _default_rag_service(session):
    from service.injection import get_rag_service
    return await get_rag_service(session)


def _fetch_bodies(adapter, docs, workers: int) -> dict:
    """Fetch bodies for discovered docs. Returns path -> (body|None, error|None)."""
    out = {}
    if workers <= 1 or len(docs) <= 1:
        for d in docs:
            try:
                out[d.path] = (adapter.fetch(d), None)
            except Exception as e:  # noqa: BLE001
                out[d.path] = (None, e)
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(adapter.fetch, d): d for d in docs}
        for fu in as_completed(futs):
            d = futs[fu]
            try:
                out[d.path] = (fu.result(), None)
            except Exception as e:  # noqa: BLE001
                out[d.path] = (None, e)
    return out


async def _ingest_document(
    session, kb_id: str, datasource_key: str, datasource_id: str, tenant_id: str,
    source_doc, existing_file_id: Optional[str], file_writer,
) -> Tuple[str, int]:
    """Write a document into the KB ingestion pipeline. Returns (file_id, file_version)."""
    content_bytes = source_doc.content.encode("utf-8")
    # Namespace the file name by data source so paths that collide across sources
    # (e.g. "index.md") stay distinct and traceable in the shared KB file list.
    # The human title lives in file_metadata["title"]; doc_id stays the canonical key.
    file_name = f"{datasource_key}/{source_doc.path}"
    dest_path = f"{kb_id}/docs/{datasource_key}/{source_doc.path}"
    stored_path = await file_writer(content_bytes, file_name, dest_path, tenant_id)

    version = int(time.time())
    md5 = hashlib.md5(content_bytes).hexdigest()

    # The viewable link: the source page for remote sources; for local sources
    # (no source_url) generate one from the file store.
    file_url = source_doc.source_url
    if not file_url:
        try:
            from pairag.file.store.file_store_helper import file_store
            file_url = await file_store.get_url_async(file_path=stored_path, tenant_id=tenant_id)
        except Exception:
            file_url = None

    # file_metadata is merged into every chunk's metadata, so it drives search
    # filtering, citations and the file/chunk UI. Keep it to display/citation
    # fields (operational bytes/version live on the entity columns). Drop nulls.
    meta = {
        "title": source_doc.title,
        "file_url": file_url,
        "source_url": source_doc.source_url,
        "source_site": source_doc.source_site,
        "summary": source_doc.summary,
        "product": source_doc.product,
        "section": source_doc.section,
        "lang": source_doc.lang,
        # identifiers / sync bookkeeping (used by tools + incremental sync)
        "datasource_id": datasource_id,
        "datasource_key": datasource_key,
        "source_doc_id": source_doc.doc_id,
        "fetched_from": source_doc.fetched_from,
        "content_hash": source_doc.content_hash,
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    entity = None
    if existing_file_id:
        res = await session.exec(
            select(KbFileEntity).where(
                KbFileEntity.id == existing_file_id, KbFileEntity.tenant_id == tenant_id
            )
        )
        entity = res.first()

    if entity is None:
        entity = KbFileEntity(
            id=existing_file_id or uuid.uuid4().hex,
            tenant_id=tenant_id,
            kb_id=kb_id,
            message_id=f"ds-{datasource_id}",  # stable: (kb_id, message_id, file_name) unique per doc
            file_name=file_name,
            file_path=stored_path,
            file_extension=os.path.splitext(source_doc.path)[1].lower() or ".md",
            file_size=len(content_bytes),
            file_md5=md5,
            file_source=file_url,
            file_metadata=meta,
            file_version=version,
            status=FileStatus.pending,
            active=True,
            file_content="",
            file_content_length=0,
        )
    else:
        entity.file_name = file_name
        entity.file_path = stored_path
        entity.file_size = len(content_bytes)
        entity.file_md5 = md5
        entity.file_source = file_url
        entity.file_metadata = meta
        entity.file_version = version
        entity.status = FileStatus.pending
        entity.updated_at = _utcnow()

    session.add(entity)
    await session.flush()
    return entity.id, version


async def run_sync(
    datasource_id: str,
    tenant_id: str,
    trigger: str = SyncTrigger.manual,
    triggered_by: Optional[str] = None,
    *,
    adapter=None,
    enqueue_fn=None,
    file_writer=None,
    rag_service_factory=None,
    batch_size: int = 50,
    fetch_workers: int = 6,
) -> dict:
    """Run one incremental sync for a data source. Returns the change counts."""
    enqueue_fn = enqueue_fn or _default_enqueue
    file_writer = file_writer or _default_file_writer
    rag_service_factory = rag_service_factory or _default_rag_service

    # -- phase A setup (atomic claim; bail if already syncing) -------------
    async with create_db_session() as session:
        svc = DataSourceService(session)
        ds, run = await svc.begin_sync(datasource_id, tenant_id, trigger, triggered_by)
        if ds is None or run is None:
            await session.rollback()
            logger.warning(
                f"[datasource-sync] {datasource_id} is already syncing; skipping duplicate run."
            )
            return {"skipped": True, "reason": "already_syncing"}
        ds_info = {
            "kb_id": ds.kb_id,
            "datasource_key": ds.datasource_key,
            "source_type": ds.source_type,
            "source_config": ds.source_config,
        }
        run_id = run.id
        await session.commit()

    counts = {"discovered": 0, "added": 0, "updated": 0, "deleted": 0, "unchanged": 0, "failed": 0}
    report = {"errors": []}
    kb_id = ds_info["kb_id"]
    datasource_key = ds_info["datasource_key"]

    def _record_error(doc_id: str, error: str):
        if len(report["errors"]) < _MAX_REPORTED_ERRORS:
            report["errors"].append({"doc_id": doc_id, "error": error})

    try:
        if adapter is None:
            adapter = get_adapter(ds_info["source_type"], datasource_key, ds_info["source_config"])

        discovered = adapter.discover()
        counts["discovered"] = len(discovered)
        disc_by_id = {adapter.make_doc_id(d.path): d for d in discovered}

        # load manifest as plain info to avoid cross-session entity reuse
        async with create_db_session() as session:
            svc = DataSourceService(session)
            manifest = await svc.get_manifest_map(datasource_id, tenant_id)
            manifest_info = {
                doc_id: {"content_hash": row.content_hash, "file_id": row.file_id, "doc_status": row.doc_status}
                for doc_id, row in manifest.items()
            }

        current_ids = set(disc_by_id)
        existing_ids = set(manifest_info)
        added_ids = current_ids - existing_ids
        maybe_ids = current_ids & existing_ids
        deleted_ids = existing_ids - current_ids

        # (DiscoveredDoc, is_existing) — added + intersection (intersection needs fetch to diff)
        to_process: List[Tuple] = (
            [(disc_by_id[i], False) for i in added_ids]
            + [(disc_by_id[i], True) for i in maybe_ids]
        )
        logger.info(
            f"[datasource-sync] {datasource_id}: discovered={len(discovered)} "
            f"added={len(added_ids)} maybe_changed={len(maybe_ids)} deleted={len(deleted_ids)}"
        )

        # -- phase A: fetch + ingest changed docs in batches ---------------
        was_cancelled = False
        for batch in _chunks(to_process, batch_size):
            # cooperative cancel: stop before fetching a batch
            if await _is_cancelled(datasource_id, tenant_id):
                was_cancelled = True
                logger.info(f"[datasource-sync] {datasource_id}: cancelled by user; stopping fetch.")
                break
            fetched = _fetch_bodies(adapter, [d for d, _ in batch], fetch_workers)
            # Re-check after the (blocking) fetch: if cancelled meanwhile, drop this
            # batch entirely — do NOT ingest/commit/enqueue it.
            if await _is_cancelled(datasource_id, tenant_id):
                was_cancelled = True
                logger.info(f"[datasource-sync] {datasource_id}: cancelled during fetch; dropping batch.")
                break
            # Collect (file_id, version) and enqueue ONLY AFTER the session
            # commits — otherwise the Celery worker (separate connection) reads
            # the KbFileEntity before it is committed and fails "File not found".
            to_enqueue: List[Tuple[str, int]] = []
            async with create_db_session() as session:
                svc = DataSourceService(session)
                for d, is_existing in batch:
                    doc_id = adapter.make_doc_id(d.path)
                    body, err = fetched.get(d.path, (None, RuntimeError("no fetch result")))
                    existing_row = (
                        await svc.get_document_row(datasource_id, doc_id, tenant_id)
                        if is_existing else None
                    )
                    if err is not None:
                        await svc.mark_document_failed(
                            datasource_id, kb_id, tenant_id, doc_id, str(err),
                            existing=existing_row, path=d.path,
                        )
                        counts["failed"] += 1
                        _record_error(doc_id, str(err))
                        continue

                    source_doc = adapter.emit(d, body)
                    prev = manifest_info.get(doc_id, {})
                    # Skip only when content is unchanged AND the doc is already
                    # fully synced — cancelled/failed/incomplete docs are re-ingested
                    # even if their content hash is identical.
                    if (
                        is_existing
                        and prev.get("content_hash") == source_doc.content_hash
                        and prev.get("doc_status") == DataSourceDocStatus.synced
                    ):
                        counts["unchanged"] += 1
                        continue
                    try:
                        existing_file_id = existing_row.file_id if existing_row else None
                        file_id, version = await _ingest_document(
                            session, kb_id, datasource_key, datasource_id, tenant_id,
                            source_doc, existing_file_id, file_writer,
                        )
                        await svc.upsert_document(
                            datasource_id, kb_id, tenant_id, source_doc,
                            file_id=file_id, doc_status=DataSourceDocStatus.ingesting,
                            existing=existing_row,
                        )
                        to_enqueue.append((file_id, version))
                        if is_existing:
                            counts["updated"] += 1
                        else:
                            counts["added"] += 1
                    except Exception as ie:  # noqa: BLE001
                        logger.warning(f"[datasource-sync] ingest failed for {doc_id}: {ie}")
                        await svc.mark_document_failed(
                            datasource_id, kb_id, tenant_id, doc_id, str(ie),
                            existing=existing_row, path=d.path,
                        )
                        counts["failed"] += 1
                        _record_error(doc_id, str(ie))
                await session.commit()

            # Final cancel check before enqueue: if cancelled in the tiny window
            # after commit, don't enqueue. Sweep the just-committed pending files to
            # cancelled so they don't sit forever unparsed.
            if to_enqueue and await _is_cancelled(datasource_id, tenant_id):
                was_cancelled = True
                async with create_db_session() as session:
                    await DataSourceService(session).cancel_sync(datasource_id, tenant_id)
                    await session.commit()
                logger.info(f"[datasource-sync] {datasource_id}: cancelled before enqueue; swept batch.")
                break

            # Files + manifest rows are committed now — safe to enqueue parsing.
            for file_id, version in to_enqueue:
                enqueue_fn(file_id, version, tenant_id)

        # -- deletions -----------------------------------------------------
        # Never delete when discovery was incomplete (e.g. a sphinx crawl page
        # failed transiently) — a missing page would otherwise look like a
        # source-side removal and wrongly purge live KB content.
        if deleted_ids and getattr(adapter, "discovery_partial", False):
            logger.warning(
                f"[datasource-sync] {datasource_id}: discovery incomplete; "
                f"skipping {len(deleted_ids)} candidate deletion(s) this run."
            )
            report["deletions_skipped_partial_discovery"] = len(deleted_ids)
            deleted_ids = set()

        if deleted_ids:
            async with create_db_session() as session:
                svc = DataSourceService(session)
                rag = await rag_service_factory(session)
                for doc_id in deleted_ids:
                    row = await svc.get_document_row(datasource_id, doc_id, tenant_id)
                    if row is None:
                        continue
                    delete_err = None
                    if row.file_id:
                        try:
                            await rag.delete_file(kb_id=kb_id, file_id=row.file_id, tenant_id=tenant_id)
                        except Exception as de:  # noqa: BLE001
                            if not _is_missing_file_error(de):
                                delete_err = de
                    if delete_err is None:
                        # vectors/chunks/file gone (or never existed) — safe to drop manifest row
                        await svc.delete_document_row(row)
                        counts["deleted"] += 1
                    else:
                        # KEEP the manifest row so the deletion is retried on the next
                        # sync; never orphan vectors/chunks that are still searchable.
                        logger.warning(f"[datasource-sync] delete_file failed for {doc_id}: {delete_err}")
                        await svc.mark_document_failed(
                            datasource_id, kb_id, tenant_id, doc_id,
                            f"delete failed: {delete_err}", existing=row, path=row.path,
                        )
                        counts["failed"] += 1
                        _record_error(doc_id, f"delete failed: {delete_err}")
                await session.commit()

        # -- finalize ------------------------------------------------------
        report["summary"] = (
            f"+{counts['added']} new / ~{counts['updated']} updated / "
            f"-{counts['deleted']} deleted / ={counts['unchanged']} unchanged / "
            f"{counts['failed']} failed"
        )
        if was_cancelled:
            report["cancelled"] = True
        async with create_db_session() as session:
            svc = DataSourceService(session)
            await svc.finalize_sync(
                datasource_id, tenant_id, run_id, counts, report, error=None,
                final_status=DataSourceStatus.cancelled if was_cancelled else None,
            )
            await session.commit()
        logger.info(f"[datasource-sync] {datasource_id} done: {report['summary']}")
        return counts

    except Exception as ex:  # noqa: BLE001
        logger.exception(f"[datasource-sync] {datasource_id} failed: {ex}")
        report["error"] = str(ex)
        try:
            async with create_db_session() as session:
                svc = DataSourceService(session)
                await svc.finalize_sync(datasource_id, tenant_id, run_id, counts, report, error=str(ex))
                await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception("[datasource-sync] failed to finalize errored run")
        return counts
