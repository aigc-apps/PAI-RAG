# init trace
from dotenv import load_dotenv

from pairag.mcp.tools.knowledgebase.local_chroma_service import LocalChromaService

load_dotenv()

import os
import asyncio
import threading
from fastapi import FastAPI
# setup models
from pairag.utils.constants import DEFAULT_MODEL_DIR
os.environ["PAIRAG_MODEL_DIR"] = DEFAULT_MODEL_DIR
from pairag.utils.download_models import ModelScopeDownloader
ModelScopeDownloader().load_rag_models()


from contextlib import asynccontextmanager
from pairag.utils.format_logging import format_logging
from pairag.core.chat_service import chat_service
from pairag.data_pipeline.job.rag_job_manager import job_manager
from pairag.core.service_daemon import startup_event
from pairag.app.feature_flags import is_feature_enabled, FeatureFlags
from loguru import logger

format_logging()



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    if is_feature_enabled(FeatureFlags.MCP):
        from pairag.app.init import init_dependencies
        await init_dependencies(init_mcp_tools=True)

    daemon_thread = threading.Thread(target=job_manager.execute_job, daemon=True)
    daemon_thread.start()

    chroma_service = LocalChromaService()
    chroma_service.start()

    asyncio.create_task(startup_event())
    yield

    chroma_service.stop()
    logger.info("Application shutting down...")


def configure(app: FastAPI):
    from pairag.api.v1_api import v1_router
    from pairag.api.chat_api import openai_router, chat_router
    from pairag.api.exception_handler import add_exception_handler
    from pairag.api.middleware import add_middlewares
    from pairag.web.webui import configure_webapp

    app.include_router(v1_router, prefix="/api/v1", tags=["api_v1"])
    app.include_router(openai_router, prefix="/v1", tags=["chat_completions"])
    app.include_router(chat_router, prefix="/chat", tags=["chat_api"])
    if is_feature_enabled(FeatureFlags.MCP):
        from pairag.api.agent.routers import add_chat_router, add_config_router
        add_config_router(app)
        add_chat_router(app)

    chat_service.initialize()
    add_middlewares(app)
    add_exception_handler(app)
    configure_webapp(app)


app = FastAPI(lifespan=lifespan)
configure(app)
