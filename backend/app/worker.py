import traceback
import dotenv
dotenv.load_dotenv()

from common.knowledgebase.types import FileStatus
from db.models.knowledgebase.file import KbFileEntity

from rag.split.file_split import split_file_tasks
from rag.chunk_helper import clear_useless_file_resources_async, delete_file_tasks_by_file_id_async, read_file_from_db, save_file_task_async, set_embedding_model_ready, update_file_status_async, update_file_content_async
from utils.modelscope_utils import download_model_to_directory
# Fix for macOS fork issues (like with ChromaDB)
# this forces the application to use spawn instead of fork
import os
os.environ["FORKED_BY_MULTIPROCESSING"] = "1"
if os.name != "nt":
    from billiard import context
    context._force_start_method("spawn")

from celery import Celery
import os
from rag.kb_file_client import kb_file_client
from rag.evaluation_tool import eval_client
import asyncio
from loguru import logger
from typing import List


DEFAULT_BROKER = "redis://localhost:6379/0"


app = Celery(
    "PAIRAG_WORKER",
    broker=os.environ.get("PAIRAG_BROKER") or DEFAULT_BROKER,
    backend=os.environ.get("PAIRAG_BROKER") or DEFAULT_BROKER,
)

async def enqueue_file_tasks_async(file_id: str, file_version: int, is_attachment: bool = False) -> None:
    logger.info(f"[WORKER] Enqueueing file {file_id} in background.")
    await update_file_status_async(file_id=file_id, status=FileStatus.parsing, failed_reason=str(traceback.format_exc()))

    try:

        file_entity: KbFileEntity = await read_file_from_db(file_id=file_id)
        if not file_entity:
            logger.warning(f"[WORKER] file {file_id} not found. Process file completed.")
            return

        if file_entity.file_version != file_version:
            logger.warning(f"[WORKER] file {file_id} has been updated. Process file completed.")
            return

        await delete_file_tasks_by_file_id_async(file_id=file_id, kb_id=file_entity.kb_id)
        # Split file into small file tasks
        part_count = 0
        num_tasks = 0
        for file_task in split_file_tasks(file_entity=file_entity):
            num_tasks += 1
            part_count = file_task.file_part
            file_task = await save_file_task_async(task_entity=file_task)
            process_file_task.delay(task_id=file_task.id, is_attachment=is_attachment)
            logger.info(f"[WORKER] Enqueued file {file_id} part {file_task.file_part} with task {file_task.id} successfully.")

        chunk_ids_to_delete = await clear_useless_file_resources_async(file_id=file_id, kb_id=file_entity.kb_id, part_count=part_count)
        await kb_file_client.adelete_chunks_from_vectordb(kb_id=file_entity.kb_id, node_ids=chunk_ids_to_delete)
        if num_tasks == 0:
            await update_file_status_async(file_id=file_id, status=FileStatus.succeeded, failed_reason=None)
            logger.info("No tasks enqueued. Mark file as completed.")
    except Exception:
        logger.error(f"[WORKER] Enqueueing file {file_id} failed, error: {traceback.format_exc()}")
        await update_file_status_async(file_id=file_id, status=FileStatus.failed, failed_reason=str(traceback.format_exc()))

@app.task(name="enqueue_file_tasks")
def enqueue_file_tasks(file_id: str, file_version: int, is_attachment: bool = False):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(enqueue_file_tasks_async(file_id=file_id, file_version=file_version, is_attachment=is_attachment))
    logger.info(f"[WORKER] Enqueueing file {file_id} completed.")

@app.task(name="enqueue_attachments_file_tasks")
def enqueue_attachments_file_tasks(file_id: str, file_version: int, file_extension: str, is_attachment: bool = False):
    loop = asyncio.get_event_loop()
    if file_extension in [".xlsx"]:
        loop.run_until_complete(update_file_content_async(file_id=file_id, is_attachment=is_attachment))
        loop.run_until_complete(update_file_status_async(file_id=file_id, status=FileStatus.succeeded, is_attachment=is_attachment))
    else:
        loop.run_until_complete(enqueue_file_tasks_async(file_id=file_id, file_version=file_version, is_attachment=is_attachment))
    logger.info(f"[WORKER] Enqueueing file {file_id} completed.")

# Enqueue file for processing, split into multiple tasks for large excels.
@app.task(name="process_file_task")
def process_file_task(task_id: str, is_attachment: bool = False):
    loop = asyncio.get_event_loop()
    logger.info(f"Processing file {task_id}.")
    loop.run_until_complete(kb_file_client.process_file_async(task_id=task_id, is_attachment=is_attachment))
    logger.info(f"Processed file {task_id} completed.")


@app.task(name="download_model")
def download_model(
    id: str,
    model_name: str,
    model_type: str="embedding"):
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
def execute_evaluation_task(dataset_id: str, experiment_id: str, exp_run_ids: List[str], is_evaluate_single_sample:bool=False):
    logger.info(f"execute_evaluation_task exp_run_ids {exp_run_ids} dataset_id {dataset_id}.")
    asyncio.run(eval_client.create_evaluation_task(dataset_id, experiment_id, exp_run_ids, is_evaluate_single_sample))
    logger.info(f"execute_evaluation_task successfully.")


@app.task(name="evaluate_sample_result")
def evaluate_sample_result(experiment_id: str, exp_run_id:str, sample_id:str, trace_id:str, evaluator_config_id:str, execution_metadata:str, output:str):
    logger.info(f"evaluate_sample_result sample_id: {sample_id}.")
    asyncio.run(eval_client.evaluate_sample_result(experiment_id, exp_run_id, sample_id, trace_id, evaluator_config_id, execution_metadata, output))
    logger.info(f"evaluate_sample_result sample_id: {sample_id} successfully.")
