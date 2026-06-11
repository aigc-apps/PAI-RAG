# init trace
from dotenv import load_dotenv
import os

# don't trace for elasticsearch client
os.environ["OTEL_PYTHON_INSTRUMENTATION_ELASTICSEARCH_ENABLED"] = "false"
load_dotenv()

from fastapi import FastAPI
import threading
# setup models
from utils.constants import DEFAULT_MODEL_DIR
os.environ["PAIRAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

from contextlib import asynccontextmanager
from utils.format_logging import format_logging
import anyio
from loguru import logger

format_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from utils.http_session import HttpSessionShared

    logger.info("Application starting up...")

    await HttpSessionShared.ensure_session()
    from db.db_context import init_db, get_db_session
    from service.tool.trace_service import TraceService
    from extensions.trace.base import init_instrument, TraceConfig
    from common.system_constants import DEFAULT_TENANT_ID
    from service.model.embedding_service import EmbeddingService
    from service.tool.evaluation_service import EvaluationService
    from service.tool.trace_service import TraceService
    from rag.vector_store.local_chroma_service import LocalChromaService
    from db.sqlite_store import sync_sqlite_store_task, stop_event, sync_sqlite_store
    import api.v1.mcp_server_middleware as mcp_middleware

    await init_db()
    logger.info("Initialized database tables.")

    session_getter = get_db_session()
    session = await anext(session_getter)
    try:
        embedding_service = EmbeddingService(session)
        evaluation_service = EvaluationService(session)
        _ = await embedding_service.get_default_embedding(tenant_id=DEFAULT_TENANT_ID)
        _ = await evaluation_service.get_default_eval_dataset(tenant_id=DEFAULT_TENANT_ID)
        trace_service = TraceService(session)
        await trace_service.init_trace()
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
    from pairag.file.store.file_store_helper import file_store
    if hasattr(file_store, "cleanup"):
        await file_store.cleanup()
    await HttpSessionShared.cleanup()
    logger.info("Application shutting down...")


def create_app():
    from api.v1.routers import add_chat_router, add_config_router
    from api.api_exception import (
        ApiException,
        api_exception_handler,
        unhandled_exception_handler,
    )
    from fastapi.exceptions import RequestValidationError
    import api.v1.mcp_server_middleware as mcp_middleware
    from app.log_middleware import CustomLoggingMiddleware
    from app.i18n_middleware import I18nMiddleware
    from fastapi.middleware.cors import CORSMiddleware
    from api.request_validate_exception import validation_exception_handler
    from extensions.trace.base import setup_propagator

    app = FastAPI(lifespan=lifespan)
    add_config_router(app)
    add_chat_router(app)

    # Initialize KbMcpServerMiddleware and add middleware
    # Pass the FastAPI app instance so middleware can mount sub-applications
    app.add_middleware(mcp_middleware.KbMcpServerMiddleware, fastapi_app=app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    # Add I18n middleware to handle Accept-Language header
    app.add_middleware(I18nMiddleware)
    app.add_middleware(CustomLoggingMiddleware)
    setup_propagator(app)
    app.add_exception_handler(ApiException, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    # Last-resort handler: anything that escapes the specific handlers above
    # (e.g. a route that forgot `@handle_api_exceptions` and let an unexpected
    # error propagate) returns a structured JSON body instead of Starlette's
    # default HTML "Internal Server Error" page.
    app.add_exception_handler(Exception, unhandled_exception_handler)
    return app

app = create_app()
