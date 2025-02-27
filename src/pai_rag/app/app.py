# init trace
import os
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pai_rag.utils.format_logging import format_logging
from pai_rag.app.api.service import configure_app
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pai_rag.core.service_daemon import periodic_check_config
from loguru import logger

format_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    asyncio.create_task(periodic_check_config())
    yield
    logger.info("Application shutting down...")



os.environ["PAI_RAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

app = FastAPI(lifespan=lifespan)

ModelScopeDownloader().load_rag_models()

configure_app(app)
