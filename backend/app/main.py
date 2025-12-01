# init trace
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
from fastapi import FastAPI
import threading
from db.sqlite_store import sync_sqlite_store_task, stop_event, sync_sqlite_store

# setup models
from utils.constants import DEFAULT_MODEL_DIR
os.environ["PAIRAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

import api.v1.mcp_server_middleware as mcp_middleware
from rag.vector_store.local_chroma_service import LocalChromaService
from app.log_middleware import CustomLoggingMiddleware
from config.providers.config_change_manager import config_change_manager
from contextlib import asynccontextmanager
from utils.format_logging import format_logging
from utils.http_session import HttpSessionShared
import anyio
from loguru import logger

format_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")

    await HttpSessionShared.ensure_session()
    await config_change_manager.init_configuration()
    asyncio.create_task(config_change_manager.monitor_changes_async())

    sqlite_thread = None
    if os.getenv("DB_TYPE", "sqlite") != "postgresql":
        sqlite_thread = threading.Thread(target=sync_sqlite_store_task, daemon=False)
        sqlite_thread.start()

    chroma_service = LocalChromaService()
    chroma_service.start()

    async with anyio.create_task_group() as tg:
        mcp_middleware.mcp_task_group = tg
        yield

    chroma_service.stop()

    if sqlite_thread:
        stop_event.set()
        sync_sqlite_store()
        sqlite_thread.join(timeout=10)
    await HttpSessionShared.cleanup()
    logger.info("Application shutting down...")


def configure(app: FastAPI):
    from api.v1.routers import add_chat_router, add_config_router
    add_config_router(app)
    add_chat_router(app)

    # Initialize KbMcpServerMiddleware and add middleware
    # Pass the FastAPI app instance so middleware can mount sub-applications
    app.add_middleware(mcp_middleware.KbMcpServerMiddleware, fastapi_app=app)

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    app.add_middleware(CustomLoggingMiddleware)


app = FastAPI(lifespan=lifespan)
configure(app)
