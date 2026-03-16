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
    await update_file_status_async(file_id=file_id, status=FileStatus.parsing, tenant_id=tenant_id)

    try:
        file_entity: KbFileEntity = await read_file_from_db(file_id=file_id, tenant_id=tenant_id)
        if not file_entity:
            logger.warning(f"[WORKER] file {file_id} not found. Process file completed.")
            return

        if file_entity.file_version != file_version:
            logger.warning(f"[WORKER] file {file_id} has been updated. Process file completed.")
            return

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


async def process_attachments_content_async(file_id: str, file_extension: str, tenant_id: str = None):
    from rag.offline_db_helper import update_file_content_async, update_file_status_async
    try:
        await update_file_content_async(file_id=file_id, is_attachment=True, tenant_id=tenant_id)
        await update_file_status_async(file_id=file_id, status=FileStatus.succeeded, is_attachment=True, tenant_id=tenant_id)
    except Exception as ex:
        logger.error(f"[WORKER] Process attachments content {file_id} failed, error: {traceback.format_exc()}")
        await update_file_status_async(file_id=file_id, status=FileStatus.failed, is_attachment=True, failed_reason=str(ex), tenant_id=tenant_id)


@app.task(name="enqueue_attachments_file_tasks")
def enqueue_attachments_file_tasks(file_id: str, file_version: int, file_extension: str, is_attachment: bool = False, tenant_id: str = None):
    loop = asyncio.get_event_loop()
    if file_extension in [".xlsx", ".xls", ".csv", ".jpg", ".png", ".jpeg", ".jsonl", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"]:
        loop.run_until_complete(process_attachments_content_async(file_id=file_id, file_extension=file_extension, tenant_id=tenant_id))
    else:
        loop.run_until_complete(enqueue_file_tasks_async(file_id=file_id, file_version=file_version, is_attachment=is_attachment, tenant_id=tenant_id))
    logger.info(f"[WORKER] Enqueueing file {file_id} completed.")

# Enqueue file for processing, split into multiple tasks for large excels.
@app.task(name="process_file_task")
def process_file_task(task_id: str, is_attachment: bool = False, tenant_id: str = None):
    from rag.kb_file_client import kb_file_client
    loop = asyncio.get_event_loop()
    logger.info(f"Processing file {task_id}.")
    loop.run_until_complete(kb_file_client.process_file_async(task_id=task_id, is_attachment=is_attachment, tenant_id=tenant_id))
    logger.info(f"Processed file {task_id} completed.")


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
