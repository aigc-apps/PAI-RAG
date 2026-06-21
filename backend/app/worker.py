import traceback
import dotenv
dotenv.load_dotenv()

from common.knowledgebase.types import FileStatus
from db.models.knowledgebase.file import KbFileEntity

# Fix for macOS fork issues (like with ChromaDB)
# this forces the application to use spawn instead of fork
import os
os.environ["FORKED_BY_MULTIPROCESSING"] = "1"
if os.name != "nt":
    from billiard import context
    context._force_start_method("spawn")

from celery import Celery
from celery.signals import worker_shutdown
import os
import asyncio
from typing import List
from db.redis_conn import REDIS_URL, REDIS_CLUSTER_MODE, REDIS_CLUSTER_NODES
from utils.format_logging import format_logging
from loguru import logger

format_logging()
logger.info("Worker starting up...")

DEFAULT_BROKER = REDIS_URL

# Log the Redis configuration for debugging
logger.info(f"Redis Cluster Mode: {REDIS_CLUSTER_MODE}")
logger.info(f"Redis URL scheme: {REDIS_URL.split('://')[0] if '://' in REDIS_URL else 'unknown'}")

# For Redis Cluster, disable result backend as it has compatibility issues with kombu
# Tasks in this worker are fire-and-forget, so result backend is not needed
result_backend = None if REDIS_CLUSTER_MODE else (os.environ.get("PAIRAG_BROKER") or DEFAULT_BROKER)

app = Celery(
    "PAIRAG_WORKER",
    broker=os.environ.get("PAIRAG_BROKER") or DEFAULT_BROKER,
    backend=result_backend,
)

# Configure Celery for Redis Cluster if needed
if REDIS_CLUSTER_MODE:
    cluster_config = {
        'task_ignore_result': True,
    }
    # Add startup nodes for cluster discovery
    if REDIS_CLUSTER_NODES and len(REDIS_CLUSTER_NODES) > 0:
        cluster_config['broker_transport_options'] = {
            'startup_nodes': [
                {'host': host, 'port': port}
                for host, port in REDIS_CLUSTER_NODES
            ],
        }
    app.conf.update(**cluster_config)


# Periodic GC of expired, unreferenced /v1/files rows. Beat is optional —
# when not running, the task can still be invoked on demand (e.g. via flower)
# or disabled entirely by setting PAIRAG_FILE_GC_INTERVAL_SECONDS=0.
_FILE_GC_INTERVAL = int(os.environ.get("PAIRAG_FILE_GC_INTERVAL_SECONDS", "3600"))
if _FILE_GC_INTERVAL > 0:
    app.conf.beat_schedule = {
        **getattr(app.conf, "beat_schedule", {}),
        "pairag-file-gc": {
            "task": "file_resource_gc_sweep",
            "schedule": _FILE_GC_INTERVAL,
        },
    }

# Periodic dispatcher that enqueues syncs for due data sources (per their
# sync_schedule). Set PAIRAG_DATASOURCE_SYNC_DISPATCH_INTERVAL_SECONDS=0 to disable.
_DS_DISPATCH_INTERVAL = int(os.environ.get("PAIRAG_DATASOURCE_SYNC_DISPATCH_INTERVAL_SECONDS", "300"))
if _DS_DISPATCH_INTERVAL > 0:
    app.conf.beat_schedule = {
        **getattr(app.conf, "beat_schedule", {}),
        "pairag-datasource-sync-dispatch": {
            "task": "datasource_sync_dispatch",
            "schedule": _DS_DISPATCH_INTERVAL,
        },
    }


@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    from pairag.file.store.file_store_helper import file_store
    loop = asyncio.get_event_loop()
    loop.run_until_complete(file_store.cleanup())
    logger.info("[WORKER] File store cleaned up.")


async def enqueue_file_tasks_async(file_id: str, file_version: int, is_attachment: bool = False, tenant_id: str = None) -> None:
    from rag.kb_file_client import kb_file_client
    from rag.offline_db_helper import (
        clear_useless_file_resources_async,
        delete_file_tasks_by_file_id_async,
        read_file_from_db,
        save_file_task_async,
        update_file_status_async,
        update_file_content_async
    )
    from rag.split.file_split import split_file_tasks

    logger.info(f"[WORKER] Enqueueing file {file_id} for tenant {tenant_id} in background.")

    try:
        # Read FIRST and bail before touching status, so a cancelled/superseded
        # file is never flipped back to parsing (cancel race).
        file_entity: KbFileEntity = await read_file_from_db(file_id=file_id, tenant_id=tenant_id)
        if not file_entity:
            logger.warning(f"[WORKER] file {file_id} not found. Process file completed.")
            return

        if file_entity.file_version != file_version:
            logger.warning(f"[WORKER] file {file_id} has been updated/cancelled (version changed). Skipping.")
            return

        if file_entity.status in (FileStatus.cancelled, FileStatus.failed):
            logger.warning(f"[WORKER] file {file_id} is {file_entity.status}; skipping enqueue.")
            return

        await update_file_status_async(file_id=file_id, status=FileStatus.parsing, tenant_id=tenant_id)

        await delete_file_tasks_by_file_id_async(file_id=file_id, kb_id=file_entity.kb_id, tenant_id=tenant_id)
        # Split file into small file tasks
        part_count = 0
        num_tasks = 0

        current_task = None
        for file_task in split_file_tasks(file_entity=file_entity):
            if current_task:
                process_file_task.delay(task_id=current_task.id, is_attachment=is_attachment, tenant_id=tenant_id)
                logger.info(f"[WORKER] Enqueued file {file_id} part {current_task.file_part} with task {current_task.id} successfully.")

            num_tasks += 1
            part_count = file_task.file_part
            current_task = await save_file_task_async(task_entity=file_task, tenant_id=tenant_id)

        # directly process the file if there is only one task
        if current_task and num_tasks == 1:
            logger.info(f"[WORKER] Processing file {file_id} part {current_task.file_part} with task {current_task.id}.")
            await kb_file_client.process_file_async(task_id=current_task.id, is_attachment=is_attachment, tenant_id=tenant_id)
            logger.info(f"[WORKER] Processed file {file_id} part {current_task.file_part} with task {current_task.id} successfully.")
        else:
            process_file_task.delay(task_id=current_task.id, is_attachment=is_attachment, tenant_id=tenant_id)
            logger.info(f"[WORKER] Enqueued file {file_id} part {current_task.file_part} with task {current_task.id} successfully.")

        chunk_ids_to_delete = await clear_useless_file_resources_async(file_id=file_id, kb_id=file_entity.kb_id, part_count=part_count, tenant_id=tenant_id)
        if chunk_ids_to_delete:
            await kb_file_client.adelete_chunks_from_vectordb(kb_id=file_entity.kb_id, node_ids=chunk_ids_to_delete, tenant_id=tenant_id)
        if num_tasks == 0:
            await update_file_status_async(file_id=file_id, status=FileStatus.succeeded, tenant_id=tenant_id)
            logger.info("No tasks enqueued. Mark file as completed.")
    except Exception as ex:
        logger.error(f"[WORKER] Enqueueing file {file_id} failed, error: {traceback.format_exc()}")
        await update_file_status_async(file_id=file_id, status=FileStatus.failed, failed_reason=str(ex), tenant_id=tenant_id)

@app.task(name="enqueue_file_tasks")
def enqueue_file_tasks(file_id: str, file_version: int, is_attachment: bool = False, tenant_id: str = None):
    loop = asyncio.get_event_loop()
    logger.info(f"[WORKER] Enqueueing file {file_id} for tenant {tenant_id} in background.")
    loop.run_until_complete(enqueue_file_tasks_async(file_id=file_id, file_version=file_version, is_attachment=is_attachment, tenant_id=tenant_id))
    logger.info(f"[WORKER] Enqueueing file {file_id} completed.")


async def process_file_resource_async(file_id: str, tenant_id: str):
    """Background extraction for the new /v1/files resource.

    Reads bytes from file_store by file_id, runs the stateless extractor,
    writes the preview into pai_file_text_content, and flips pai_file.status
    to succeeded/failed. No coupling to KbFileEntity.
    """
    from db.db_context import create_db_session
    from service.file.file_resource_service import FileResourceService
    from service.file.content_extractor import (
        EXTRACTOR_VERSION,
        chunk_text,
        extract_text_from_bytes,
        should_chunk,
    )
    from pairag.file.store.file_store_helper import file_store

    async with create_db_session() as session:
        svc = FileResourceService(session)
        entity = await svc.get_file(file_id=file_id, tenant_id=tenant_id)
        if not entity:
            logger.warning(f"[WORKER] FileEntity {file_id} not found; skipping.")
            return
        await svc.mark_status(
            file_id=file_id, tenant_id=tenant_id, status=FileStatus.parsing
        )

    try:
        stream = await file_store.read_async(
            file_path=entity.file_path, tenant_id=tenant_id
        )
        raw = stream.read() if hasattr(stream, "read") else stream
        result = extract_text_from_bytes(
            raw,
            entity.file_extension or "",
            file_name=entity.file_name,
            tenant_id=tenant_id,
        )
        async with create_db_session() as session:
            svc = FileResourceService(session)
            if result is not None:
                content, truncated_at_extract = result
                await svc.write_text_content(
                    file_id=file_id,
                    tenant_id=tenant_id,
                    content=content,
                    extractor_version=EXTRACTOR_VERSION,
                    truncated_at_extract=truncated_at_extract,
                )
                # Only build chunks when the extract is big enough to warrant
                # search. Small files skip chunking; the agent inlines their
                # full text via /text directly.
                if should_chunk(len(content)):
                    chunks = chunk_text(content)
                    written = await svc.replace_chunks(
                        file_id=file_id, tenant_id=tenant_id, chunks=chunks
                    )
                    logger.info(
                        f"[WORKER] {file_id} chunked: {written} chunks "
                        f"(total_length={len(content)})"
                    )
            await svc.mark_status(
                file_id=file_id, tenant_id=tenant_id, status=FileStatus.succeeded
            )
    except Exception as ex:
        logger.error(
            f"[WORKER] process_file_resource {file_id} failed: {traceback.format_exc()}"
        )
        async with create_db_session() as session:
            svc = FileResourceService(session)
            await svc.mark_status(
                file_id=file_id,
                tenant_id=tenant_id,
                status=FileStatus.failed,
                failed_reason=str(ex),
            )


@app.task(name="process_file_resource_task")
def process_file_resource_task(file_id: str, tenant_id: str = None):
    loop = asyncio.get_event_loop()
    logger.info(f"[WORKER] Processing file resource {file_id} for tenant {tenant_id}.")
    loop.run_until_complete(
        process_file_resource_async(file_id=file_id, tenant_id=tenant_id)
    )
    logger.info(f"[WORKER] Processed file resource {file_id} successfully.")


async def file_resource_gc_sweep_async(batch_size: int = 200):
    """Hard-delete expired unreferenced files. One batch per tick.

    Safe to run on any schedule; the query is O(log N) with the `expires_at`
    index and each file delete is independent.
    """
    from db.db_context import create_db_session
    from service.file.file_resource_service import FileResourceService

    async with create_db_session() as session:
        svc = FileResourceService(session)
        candidates = await svc.sweep_expired_candidates(limit=batch_size)
        logger.info(f"[GC] file_resource sweep found {len(candidates)} candidates")
        deleted = 0
        for c in candidates:
            try:
                ok = await svc.hard_delete(file_id=c.id, tenant_id=c.tenant_id)
                if ok:
                    deleted += 1
            except Exception:
                logger.warning(
                    f"[GC] hard_delete failed for {c.id} (tenant={c.tenant_id}): "
                    f"{traceback.format_exc()}"
                )
        logger.info(f"[GC] file_resource sweep deleted {deleted} files")
        return deleted


@app.task(name="file_resource_gc_sweep")
def file_resource_gc_sweep():
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(file_resource_gc_sweep_async())


# Enqueue file for processing, split into multiple tasks for large excels.
@app.task(name="process_file_task")
def process_file_task(task_id: str, is_attachment: bool = False, tenant_id: str = None):
    from rag.kb_file_client import kb_file_client
    loop = asyncio.get_event_loop()
    logger.info(f"Processing file {task_id}.")
    loop.run_until_complete(kb_file_client.process_file_async(task_id=task_id, is_attachment=is_attachment, tenant_id=tenant_id))
    logger.info(f"Processed file {task_id} completed.")


@app.task(name="datasource_sync_dispatch")
def datasource_sync_dispatch():
    from rag.datasource.scheduler import dispatch_due_datasources
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(dispatch_due_datasources())


@app.task(name="sync_datasource")
def sync_datasource(datasource_id: str, tenant_id: str = None, trigger: str = "manual", triggered_by: str = None):
    from rag.datasource.sync_worker import run_sync
    loop = asyncio.get_event_loop()
    logger.info(f"[WORKER] Syncing data source {datasource_id} for tenant {tenant_id} (trigger={trigger}).")
    loop.run_until_complete(run_sync(
        datasource_id=datasource_id, tenant_id=tenant_id, trigger=trigger, triggered_by=triggered_by,
    ))
    logger.info(f"[WORKER] Synced data source {datasource_id} completed.")


@app.task(name="download_model")
def download_model(
    id: str,
    model_name: str,
    model_type: str="embedding"):
    from utils.modelscope_utils import download_model_to_directory
    from rag.offline_db_helper import set_embedding_model_ready
    logger.info(f"Downloading {model_type} model {id} {model_name}.")
    if model_type == "embedding":
        loop = asyncio.get_event_loop()
        try:
            download_model_to_directory(model_name)
        except Exception:
            logger.error(f"Failed to download embedding model, error: {traceback.format_exc()}")
        loop.run_until_complete(set_embedding_model_ready(id=id))
        logger.info(f"Downloaded embedding model {id} {model_name} successfully.")
    else:
        logger.info(f"Model {model_name} is not a embedding model,skip processing.")


@app.task(name="execute_evaluation_task")
def execute_evaluation_task(dataset_id: str, experiment_id: str, exp_run_ids: List[str], is_evaluate_single_sample:bool=False, tenant_id: str = None):
    from rag.evaluation_tool import eval_client

    logger.info(f"execute_evaluation_task exp_run_ids {exp_run_ids} dataset_id {dataset_id}.")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(eval_client.create_evaluation_task(dataset_id, experiment_id, exp_run_ids, is_evaluate_single_sample, tenant_id=tenant_id))
    logger.info(f"execute_evaluation_task successfully.")
