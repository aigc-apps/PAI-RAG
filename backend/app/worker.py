import traceback
import dotenv
dotenv.load_dotenv()

from rag.chunk_helper import set_embedding_model_ready
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
from rag.knowledgebase_tool import kb_client
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


@app.task(name="process_file")
def process_file(file_id: str, is_attachment: bool = False):
    loop = asyncio.get_event_loop()
    logger.info(f"Processing file {file_id}.")
    loop.run_until_complete(kb_client.process_file_async(file_id, is_attachment))
    logger.info(f"Processed file {file_id} successfully.")


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
def execute_evaluation_task(eval_id: str, experiment_id: str, exp_run_ids: List[str]):
    loop = asyncio.get_event_loop()
    logger.info(f"execute_evaluation_task exp_run_ids {exp_run_ids} eval_id {eval_id}.")
    loop.run_until_complete(eval_client.create_evaluation_task(eval_id, experiment_id, exp_run_ids))
    logger.info(f"execute_evaluation_task successfully.")
