# init trace
from dotenv import load_dotenv

from pairag.mcp.tools.knowledgebase.local_chroma_service import LocalChromaService

load_dotenv()

import os
import asyncio
from fastapi import FastAPI
# setup models
from pairag.utils.constants import DEFAULT_MODEL_DIR
os.environ["PAIRAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

from pairag.mcp.providers.config_change_manager import config_change_manager
from contextlib import asynccontextmanager
from pairag.utils.format_logging import format_logging
from loguru import logger

format_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    chroma_service = LocalChromaService()
    chroma_service.start()
    await config_change_manager.init_configuration()
    asyncio.create_task(config_change_manager.monitor_changes_async())
    yield

    chroma_service.stop()
    logger.info("Application shutting down...")


def configure(app: FastAPI):
    from pairag.api.v1_api import v1_router
    from pairag.api.chat_api import openai_router, chat_router
    from pairag.api.exception_handler import add_exception_handler
    from pairag.api.middleware import add_middlewares
    from pairag.web.webui import configure_webapp

    # app.include_router(v1_router, prefix="/api/v1", tags=["api_v1"])
    # app.include_router(openai_router, prefix="/v1", tags=["chat_completions"])
    # app.include_router(chat_router, prefix="/chat", tags=["chat_api"])
    from pairag.api.agent.routers import add_chat_router, add_config_router
    add_config_router(app)
    add_chat_router(app)

    # chat_service.initialize()
    # add_middlewares(app)
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    add_exception_handler(app)
    # configure_webapp(app)


app = FastAPI(lifespan=lifespan)
configure(app)
