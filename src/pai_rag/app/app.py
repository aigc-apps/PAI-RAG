# init trace
import os
import asyncio
import threading
from fastapi import FastAPI

# setup models
os.environ["PAI_RAG_MODEL_DIR"] = DEFAULT_MODEL_DIR
from pai_rag.utils.download_models import ModelScopeDownloader
ModelScopeDownloader().load_rag_models()


from contextlib import asynccontextmanager
from pai_rag.utils.format_logging import format_logging
from pai_rag.app.api.service import configure_app
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pai_rag.knowledgebase.rag_job_manager import job_manager
from pai_rag.core.service_daemon import startup_event
from loguru import logger

format_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    daemon_thread = threading.Thread(target=job_manager.execute_job, daemon=True)
    daemon_thread.start()

    asyncio.create_task(startup_event())
    yield

    logger.info("Application shutting down...")




app = FastAPI(lifespan=lifespan)
configure_app(app)
