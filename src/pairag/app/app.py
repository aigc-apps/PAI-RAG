# init trace
from dotenv import load_dotenv
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
        logger.info("Initializing databases for MCP.")
        from pairag.db.db_context import init_db
        from pairag.mcp.providers.mcp_tool_provider import mcp_provider
        from pairag.mcp.providers.llm_provider import llm_provider

        await init_db()
        logger.info("Initialized databases for MCP.")
        await mcp_provider.refresh()
        logger.info("Initialized mcp tools.")
        await llm_provider.refresh()
        logger.info("Initialized llm models.")

    daemon_thread = threading.Thread(target=job_manager.execute_job, daemon=True)
    daemon_thread.start()

    asyncio.create_task(startup_event())
    yield

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
        from pairag.api.config_api import config_router
        app.include_router(config_router, prefix="/v1/config", tags=["config_api"])

    chat_service.initialize()
    add_middlewares(app)
    add_exception_handler(app)
    configure_webapp(app)


app = FastAPI(lifespan=lifespan)
configure(app)
