# init trace
from dotenv import load_dotenv

from rag.vector_store.local_chroma_service import LocalChromaService

load_dotenv()

import os
import asyncio
from fastapi import FastAPI
# setup models
from utils.constants import DEFAULT_MODEL_DIR
os.environ["PAIRAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

from config.providers.config_change_manager import config_change_manager
from contextlib import asynccontextmanager
from utils.format_logging import format_logging
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
    from api.v1.routers import add_chat_router, add_config_router
    add_config_router(app)
    add_chat_router(app)

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )


app = FastAPI(lifespan=lifespan)
configure(app)
