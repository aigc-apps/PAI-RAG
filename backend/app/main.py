# init trace
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
from fastapi import FastAPI
import threading
from db.sqlite_store import sync_sqlite_store_task, stop_event, sync_sqlite_store
from fastapi.exceptions import RequestValidationError
# setup models
from utils.constants import DEFAULT_MODEL_DIR
os.environ["PAIRAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

from api.api_exception import ApiException, api_exception_handler
from api.request_validate_exception import validation_exception_handler
import api.v1.mcp_server_middleware as mcp_middleware
from rag.vector_store.local_chroma_service import LocalChromaService
from app.log_middleware import CustomLoggingMiddleware
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
    from db.db_context import init_db, get_db_session
    from service.tool.trace_service import TraceService
    from extensions.trace.base import init_instrument, TraceConfig
    from common.system_constants import DEFAULT_TENANT_ID
    from service.model.embedding_service import EmbeddingService

    await init_db()
    logger.info("Initialized database tables.")

    session_getter = get_db_session()
    session = await anext(session_getter)
    try:
        embedding_service = EmbeddingService(session)
        _ = await embedding_service.get_default_embedding(tenant_id=DEFAULT_TENANT_ID)
        trace_service = TraceService(session)
        trace_config = await trace_service.get_trace_config(tenant_id=DEFAULT_TENANT_ID)
        if trace_config:
            init_instrument(TraceConfig(
                endpoint=trace_config.endpoint,
                token=trace_config.token,
                service_name=trace_config.service_name,
                user_args=trace_config.user_args,
                enabled=trace_config.enabled))
            logger.info("Initialized trace config.")
    finally:
        await session.close()

    sqlite_thread = None
    if os.getenv("DB_TYPE", "sqlite") == "sqlite":
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
    app.add_exception_handler(ApiException, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

app = FastAPI(lifespan=lifespan)
configure(app)
