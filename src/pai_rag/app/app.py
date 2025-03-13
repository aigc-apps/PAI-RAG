# init trace
import os
import asyncio
import threading
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pai_rag.utils.format_logging import format_logging
from pai_rag.app.api.service import configure_app
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pai_rag.knowledgebase.rag_job_manager import job_manager
from pai_rag.core.service_daemon import startup_event
from loguru import logger

format_logging()

DEFAULT_BACKGROUND_WORKER_NUM = os.environ.get("DEFAULT_BACKGROUND_WORKER_NUM", 4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    stop_event = threading.Event()

    background_thread = threading.Thread(target=job_manager.execute_job_with_workers, args=(stop_event,DEFAULT_BACKGROUND_WORKER_NUM))
    background_thread.start()

    asyncio.create_task(startup_event())
    yield
    logger.info("Gracefully exit. Exiting background thread...")
    stop_event.set()
    background_thread.join()

    logger.info("Application shutting down...")



os.environ["PAI_RAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

app = FastAPI(lifespan=lifespan)

ModelScopeDownloader().load_rag_models()

configure_app(app)
