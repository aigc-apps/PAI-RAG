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

# Max bytes of source document body stored in KbFileEntity.file_content (DB).
# Covers 99%+ of markdown docs; grep/read tools hit this DB column directly
# instead of OSS, avoiding network I/O on every request.
MAX_FILE_CONTENT_BYTES = 200 * 1024


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_doc_id() -> str:
    """Generate the one local identifier shared by doc_id and file_id."""
    return f"doc_{uuid.uuid4().hex[:8]}"


def _local_doc_id(existing: Optional[DataSourceDocumentEntity]) -> str:
    if existing is None:
        return _new_doc_id()
    if not existing.doc_id or existing.file_id != existing.doc_id:
        raise ValueError(
            f"Manifest identity mismatch for source_id={existing.source_id}: "
            f"doc_id={existing.doc_id!r}, file_id={existing.file_id!r}"
        )
    return existing.doc_id


def _diff_source_ids(discovered: dict, manifest: dict) -> Tuple[set, set, set]:
    """Return added, existing, and deleted upstream identities."""
    current_ids = set(discovered)
    existing_ids = set(manifest)
    return (
        current_ids - existing_ids,
        current_ids & existing_ids,
        existing_ids - current_ids,
    )


def _is_unchanged(previous: dict, source_doc, is_existing: bool) -> bool:
    """Only fully synced content at the same source path is unchanged."""
    return bool(
        is_existing
        and previous.get("content_hash") == source_doc.content_hash
        and previous.get("doc_status") == DataSourceDocStatus.synced
        and previous.get("path") == source_doc.path
    )


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
    source_doc, doc_id: str, file_writer,
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
        "source_id": source_doc.source_id,
        "source_doc_id": doc_id,
        "fetched_from": source_doc.fetched_from,
        "content_hash": source_doc.content_hash,
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    # Look up by doc_id first (the canonical key). If the doc_id format
    # changed (e.g. after a make_doc_id refactor), fall back to the unique
    # business key (kb_id, message_id, file_name) to find the existing row.
    res = await session.exec(
        select(KbFileEntity).where(
            KbFileEntity.id == doc_id, KbFileEntity.tenant_id == tenant_id
        )
    )
    entity = res.first()

    if entity is None:
        message_id = f"ds-{datasource_id}"
        res2 = await session.exec(
            select(KbFileEntity).where(
                KbFileEntity.kb_id == kb_id,
                KbFileEntity.message_id == message_id,
                KbFileEntity.file_name == file_name,
                KbFileEntity.tenant_id == tenant_id,
            )
        )
        entity = res2.first()
        if entity is not None:
            logger.info(
                f"[datasource-sync] Found existing file by unique key "
                f"(kb={kb_id}, msg={message_id}, name={file_name}), "
                f"reusing id={entity.id} instead of new doc_id={doc_id}"
            )
            doc_id = entity.id  # Keep the existing id to avoid PK conflict

    if entity is None:
        entity = KbFileEntity(
            id=doc_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            message_id=f"ds-{datasource_id}",  # stable namespace for source files
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
            file_content=source_doc.content[:MAX_FILE_CONTENT_BYTES],
            file_content_length=min(len(source_doc.content.encode("utf-8")), MAX_FILE_CONTENT_BYTES),
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
        entity.file_content = source_doc.content[:MAX_FILE_CONTENT_BYTES]
        entity.file_content_length = min(len(content_bytes), MAX_FILE_CONTENT_BYTES)
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
    logger.info(
        f"[datasource-sync] START datasource_id={datasource_id} tenant={tenant_id} "
        f"trigger={trigger} triggered_by={triggered_by}"
    )
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
        logger.info(
            f"[datasource-sync] Phase A: claimed datasource={datasource_id} "
            f"run_id={run_id} kb_id={ds_info['kb_id']} "
            f"source_type={ds_info['source_type']} key={ds_info['datasource_key']}"
        )
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
            logger.info(
                f"[datasource-sync] Creating adapter for source_type={ds_info['source_type']} "
                f"key={datasource_key} config_keys={list(ds_info['source_config'].keys()) if ds_info['source_config'] else 'None'}"
            )
            adapter = get_adapter(ds_info["source_type"], datasource_key, ds_info["source_config"])

        logger.info(
            f"[datasource-sync] {datasource_id}: starting discovery with adapter "
            f"type={type(adapter).__name__}"
        )
        discovered = adapter.discover()
        counts["discovered"] = len(discovered)
        logger.info(
            f"[datasource-sync] {datasource_id}: discovery returned {len(discovered)} docs, "
            f"discovery_partial={getattr(adapter, 'discovery_partial', False)}"
        )
        if discovered:
            sample_paths = [d.path for d in discovered[:5]]
            logger.info(
                f"[datasource-sync] {datasource_id}: first {len(sample_paths)} paths: {sample_paths}"
            )
        disc_by_id = {}
        for doc in discovered:
            source_id = adapter.get_source_id(doc)
            if source_id in disc_by_id:
                raise ValueError(
                    f"Adapter returned duplicate source_id={source_id!r} "
                    f"for paths {disc_by_id[source_id].path!r} and {doc.path!r}."
                )
            disc_by_id[source_id] = doc

        # load manifest as plain info to avoid cross-session entity reuse
        async with create_db_session() as session:
            svc = DataSourceService(session)
            manifest = await svc.get_manifest_map(datasource_id, tenant_id)
            manifest_info = {
                source_id: {
                    "content_hash": row.content_hash,
                    "doc_id": row.doc_id,
                    "file_id": row.file_id,
                    "doc_status": row.doc_status,
                    "path": row.path,
                }
                for source_id, row in manifest.items()
            }
        logger.info(
            f"[datasource-sync] {datasource_id}: manifest has {len(manifest_info)} existing docs"
        )

        added_ids, maybe_ids, deleted_ids = _diff_source_ids(disc_by_id, manifest_info)

        # (DiscoveredDoc, is_existing) — added + intersection (intersection needs fetch to diff)
        to_process: List[Tuple] = (
            [(disc_by_id[i], False) for i in added_ids]
            + [(disc_by_id[i], True) for i in maybe_ids]
        )
        logger.info(
            f"[datasource-sync] {datasource_id}: DIFF result — "
            f"discovered={len(discovered)} added={len(added_ids)} "
            f"maybe_changed={len(maybe_ids)} deleted={len(deleted_ids)}"
        )
        if len(added_ids) == 0 and len(maybe_ids) == 0 and len(deleted_ids) == 0:
            logger.warning(
                f"[datasource-sync] {datasource_id}: NO CHANGES detected! "
                f"discovered_ids_sample={list(disc_by_id.keys())[:5]} "
                f"existing_ids_sample={list(manifest_info.keys())[:5]}"
            )

        # -- phase A: fetch + ingest changed docs in batches ---------------
        was_cancelled = False
        logger.info(
            f"[datasource-sync] {datasource_id}: starting fetch+ingest for {len(to_process)} docs "
            f"(batch_size={batch_size}, fetch_workers={fetch_workers})"
        )
        for batch_idx, batch in enumerate(_chunks(to_process, batch_size)):
            # cooperative cancel: stop before fetching a batch
            if await _is_cancelled(datasource_id, tenant_id):
                was_cancelled = True
                logger.info(f"[datasource-sync] {datasource_id}: cancelled by user; stopping fetch.")
                break
            logger.info(
                f"[datasource-sync] {datasource_id}: batch {batch_idx} fetching {len(batch)} docs..."
            )
            fetched = _fetch_bodies(adapter, [d for d, _ in batch], fetch_workers)
            fetch_ok = sum(1 for _, (body, err) in fetched.items() if err is None)
            fetch_fail = sum(1 for _, (body, err) in fetched.items() if err is not None)
            logger.info(
                f"[datasource-sync] {datasource_id}: batch {batch_idx} fetch done: "
                f"ok={fetch_ok} failed={fetch_fail}"
            )
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
                batch_ingested = 0
                batch_skipped = 0
                batch_ingest_failed = 0
                for d, is_existing in batch:
                    source_id = adapter.get_source_id(d)
                    body, err = fetched.get(d.path, (None, RuntimeError("no fetch result")))
                    existing_row = (
                        await svc.get_document_row_by_source_id(datasource_id, source_id, tenant_id)
                        if is_existing else None
                    )
                    doc_id = _local_doc_id(existing_row)
                    if err is not None:
                        logger.warning(
                            f"[datasource-sync] {datasource_id}: fetch failed for doc_id={doc_id} "
                            f"path={d.path}: {err}"
                        )
                        await svc.mark_document_failed(
                            datasource_id, kb_id, tenant_id, source_id, doc_id, str(err),
                            existing=existing_row, path=d.path,
                        )
                        counts["failed"] += 1
                        batch_ingest_failed += 1
                        _record_error(doc_id, str(err))
                        continue

                    source_doc = adapter.emit(d, body)
                    prev = manifest_info.get(source_id, {})
                    # Skip only when content is unchanged AND the doc is already
                    # fully synced — cancelled/failed/incomplete docs are re-ingested
                    # even if their content hash is identical.
                    if _is_unchanged(prev, source_doc, is_existing):
                        counts["unchanged"] += 1
                        batch_skipped += 1
                        continue
                    try:
                        # Isolate each document so a failed flush rolls back to a
                        # savepoint before we record its failure in this session.
                        async with session.begin_nested():
                            file_id, version = await _ingest_document(
                                session, kb_id, datasource_key, datasource_id, tenant_id,
                                source_doc, doc_id, file_writer,
                            )
                            await svc.upsert_document(
                                datasource_id, kb_id, tenant_id, source_doc,
                                doc_id=doc_id, doc_status=DataSourceDocStatus.ingesting,
                                existing=existing_row,
                            )
                        to_enqueue.append((file_id, version))
                        batch_ingested += 1
                        if is_existing:
                            counts["updated"] += 1
                        else:
                            counts["added"] += 1
                        logger.debug(
                            f"[datasource-sync] {datasource_id}: ingested doc_id={doc_id} "
                            f"file_id={file_id} version={version} "
                            f"({'updated' if is_existing else 'new'})"
                        )
                    except Exception as ie:  # noqa: BLE001
                        logger.warning(f"[datasource-sync] ingest failed for {doc_id}: {ie}")
                        await svc.mark_document_failed(
                            datasource_id, kb_id, tenant_id, source_id, doc_id, str(ie),
                            existing=existing_row, path=d.path,
                        )
                        counts["failed"] += 1
                        batch_ingest_failed += 1
                        _record_error(doc_id, str(ie))
                await session.commit()
                logger.info(
                    f"[datasource-sync] {datasource_id}: batch {batch_idx} committed — "
                    f"ingested={batch_ingested} skipped={batch_skipped} failed={batch_ingest_failed} "
                    f"to_enqueue={len(to_enqueue)}"
                )

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
            logger.info(
                f"[datasource-sync] {datasource_id}: batch {batch_idx} enqueuing "
                f"{len(to_enqueue)} file tasks for parsing..."
            )
            for file_id, version in to_enqueue:
                enqueue_fn(file_id, version, tenant_id)
            logger.info(
                f"[datasource-sync] {datasource_id}: batch {batch_idx} enqueued successfully"
            )

        # -- deletions -----------------------------------------------------
        logger.info(
            f"[datasource-sync] {datasource_id}: processing {len(deleted_ids)} deletions..."
        )
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
                for source_id in deleted_ids:
                    row = await svc.get_document_row_by_source_id(datasource_id, source_id, tenant_id)
                    if row is None:
                        continue
                    doc_id = row.doc_id
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
                            datasource_id, kb_id, tenant_id, source_id, doc_id,
                            f"delete failed: {delete_err}", existing=row, path=row.path,
                        )
                        counts["failed"] += 1
                        _record_error(doc_id, f"delete failed: {delete_err}")
                await session.commit()

        # -- finalize ------------------------------------------------------
        logger.info(
            f"[datasource-sync] {datasource_id}: FINALIZING — "
            f"added={counts['added']} updated={counts['updated']} "
            f"deleted={counts['deleted']} unchanged={counts['unchanged']} "
            f"failed={counts['failed']} cancelled={was_cancelled}"
        )
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
        logger.exception(
            f"[datasource-sync] {datasource_id} FAILED with exception: {ex}"
        )
        report["error"] = str(ex)
        try:
            async with create_db_session() as session:
                svc = DataSourceService(session)
                await svc.finalize_sync(datasource_id, tenant_id, run_id, counts, report, error=str(ex))
                await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception("[datasource-sync] failed to finalize errored run")
        return counts
