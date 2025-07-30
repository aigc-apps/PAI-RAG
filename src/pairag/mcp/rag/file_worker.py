import traceback
import dotenv

from pairag.mcp.providers.chunk_helper import set_embedding_model_ready
from pairag.utils.modelscope_utils import download_model_to_directory
dotenv.load_dotenv()
# Fix for macOS fork issues (like with ChromaDB)
# this forces the application to use spawn instead of fork
import os
os.environ["FORKED_BY_MULTIPROCESSING"] = "1"
if os.name != "nt":
    from billiard import context
    context._force_start_method("spawn")

from celery import Celery
import os
from pairag.mcp.tools.knowledgebase.knowledgebase_tool import kb_client
from pairag.mcp.providers.config_change_manager import config_change_manager
import asyncio
from loguru import logger



DEFAULT_BROKER = "redis://localhost:6379/0"


app = Celery(
    "PAIRAG_WORKER",
    broker=os.environ.get("PAIRAG_BROKER") or DEFAULT_BROKER,
    backend=os.environ.get("PAIRAG_BROKER") or DEFAULT_BROKER,
)

async def init_worker():
    if not config_change_manager.initialized:
        config_change_manager.worker_mode = True
        await config_change_manager.init_configuration()
        asyncio.create_task(config_change_manager.monitor_changes_async())
        logger.info("FileWorker initialized.")
    else:
        logger.info("FileWorker already initialized.")

@app.task(name="process_file")
def process_file(file_id: str):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_worker())
    logger.info(f"Processing file {file_id}.")
    loop.run_until_complete(kb_client.process_file_async(file_id))
    logger.info(f"Processed file {file_id} successfully.")

@app.task(name="download_model")
def download_model(
    model_id: str,
    model_name: str,
    model_type: str="embedding"):
    if model_type == "embedding":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(init_worker())
        try:
            download_model_to_directory(model_name)
        except Exception:
            logger.error(f"Failed to download embedding model, error: {traceback.format_exc()}")
        loop.run_until_complete(set_embedding_model_ready(model_id=model_id))
        logger.info(f"Downloaded embedding model {model_id} {model_name} successfully.")
    else:
        logger.info(f"Model {model_name} is not a embedding model,skip processing.")
