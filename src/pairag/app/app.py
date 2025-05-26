# init trace
import os
import asyncio
import threading
from fastapi import FastAPI

# setup models

from pairag.api.middleware import add_middlewares
from pairag.utils.constants import DEFAULT_MODEL_DIR
os.environ["pairag_MODEL_DIR"] = DEFAULT_MODEL_DIR
from pairag.utils.download_models import ModelScopeDownloader
ModelScopeDownloader().load_rag_models()


from contextlib import asynccontextmanager
from pairag.utils.format_logging import format_logging
from pairag.core.chat_service import chat_service
from pairag.data_pipeline.job.rag_job_manager import job_manager
from pairag.core.service_daemon import startup_event
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


def configure(app: FastAPI):
    from pairag.api.api_router_v1 import router_v1
    from pairag.api.chat_completions import router_openai
    from pairag.api.exception_handler import add_exception_handler
    from pairag.api.middleware import add_middlewares
    from pairag.web.webui import configure_webapp

    app.include_router(router_v1, prefix="/api/v1", tags=["api_v1"])
    app.include_router(router_openai, prefix="/v1", tags=["chat_completions"])

    chat_service.initialize()
    add_middlewares(app)
    add_exception_handler(app)
    configure_webapp(app)


app = FastAPI(lifespan=lifespan)
configure(app)
