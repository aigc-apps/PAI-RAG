# ruff: noqa: E402
from __future__ import annotations
import os
from contextlib import asynccontextmanager

from loguru import logger

# Load .env into os.environ before anything reads settings or secrets. This
# must run at import time, ahead of get_settings()/provider/sandbox code that
# resolves AGENTRUN_*/OPENAI_API_KEY via os.environ.get. pydantic-settings'
# env_file would only fill Settings fields, not os.environ, so we use
# load_dotenv() to cover both.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.engine import make_url
from app.config import get_settings
from app.deps import AppState, rebuild_app_state_from_config, reload_app_state
from app.db import make_engine, create_all, migrate
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.llm import LeanLLM
from app.providers import ModelCatalog, ProviderRouter
from app.routes.auth import router as auth_router
from app.routes.responses import router as responses_router
from app.routes.chat import router as chat_router
from app.routes.conversations import router as conversations_router
from app.routes.models import router as models_router
from app.routes.users import router as users_router
from app.routes.config import router as config_router
from app.routes.files import router as files_router
from app.routes.aliyun import router as aliyun_router
from app.routes.knowledge import router as knowledge_router
from app.routes.agents import router as agents_router
from app.knowledge import KnowledgeService
from app.jobs import JobQueue, register_knowledge_handlers
from app.search_engine import build_search_engine
from app.sync_pipeline import PipelineLimits
from app.agent_config import DEFAULT_DOCUMENT, apply_runtime_status
from app.agent_config_store import SqlAgentConfigStore
from agent.tools.defaults import build_default_registry
from agent.tools.skills import load_skills


def _new_database_config_seed(_settings=None):
    """Return a clean seed; local YAML must never initialize a new database."""
    return DEFAULT_DOCUMENT.model_copy(deep=True)


def _build_llm(settings) -> LeanLLM | None:
    """Legacy fallback client built from the openai_* settings.

    Only constructed when OPENAI_API_KEY is set; otherwise returns None and the
    ProviderRouter handles every request (resolving the configured default
    provider on demand). This keeps boot from hard-requiring OpenAI creds — a
    deployment that only configures, say, dashscope/anthropic boots fine."""
    if not settings.openai_api_key:
        return None
    return LeanLLM(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.default_model,
    )


def _init_tracing(app: FastAPI, settings) -> None:
    """Best-effort OpenTelemetry init. Imported at runtime (not module top) so the
    service does not load the OTel stack when tracing is not configured. Any
    initialization failure degrades to no tracing."""
    try:
        from extensions.trace.config import TraceConfig

        if not TraceConfig.from_env().enabled:
            logger.info("[trace] tracing disabled (no OTLP endpoint / LANGFUSE_* configured)")
            return
        from extensions.trace import init_tracing, instrument_fastapi
        if init_tracing(settings):
            instrument_fastapi(app)
    except Exception as e:  # extension or opentelemetry absent — run without tracing
        logger.info("[trace] tracing extension unavailable, continuing without it: {}", e)


def _log_database_backend(settings) -> None:
    backend = (
        "sqlite (memory)"
        if settings.store_backend == "memory"
        else make_url(settings.db_url).get_backend_name()
    )
    logger.info("[db] database backend = {}", backend)


async def _reload_config_from_state(app: FastAPI, settings) -> None:
    state = app.state.app_state
    if getattr(state, "config_store", None) is None:
        reload_app_state(state, settings)
        return
    stored = await state.config_store.load()
    state.config_revision = stored.revision
    if state.router is not None:
        state.router.reload(ModelCatalog(**stored.doc.models))
    rebuild_app_state_from_config(state, settings, stored.doc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _log_database_backend(settings)
    _init_tracing(app, settings)
    engine = None
    if settings.store_backend == "memory":
        store = InMemoryStore()
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
    else:
        # ensure the data dir exists for file-based sqlite
        if (
            settings.db_url.startswith("sqlite")
            and ":memory:" not in settings.db_url
        ):
            os.makedirs("./data", exist_ok=True)
        # Persistent DBs are schema-managed by Alembic. Migrate on boot unless
        # AUTO_MIGRATE=false hands that to the deploy pipeline.
        if settings.auto_migrate:
            await migrate(settings.db_url)
        engine = make_engine(settings.db_url)
        store = SqlStore(engine)
    config_store = SqlAgentConfigStore(
        engine,
        seed=_new_database_config_seed(settings),
    )
    stored_config = await config_store.load()
    agent_config = stored_config.doc
    # Router before KnowledgeService: the KB service resolves its embedder/reranker
    # through the router (ingest + query). One router instance, stored on AppState.
    catalog = ModelCatalog(**agent_config.models)
    provider_router = ProviderRouter(
        catalog,
        path=settings.models_path,
        embedding_concurrency=settings.sync_embedding_concurrency,
    )
    # Built before the registry so knowledge_search can bind to it; the same
    # instance is stored on AppState below and reused for the REST query routes.
    # Global vector-store selection lives in knowledgebase.vectordb (see
    # agent_config). ES degrades to local on outage (fallback_to_local) so a
    # transient ES failure doesn't hard-fail retrieval.
    vectordb = agent_config.knowledgebase.vectordb
    knowledge = KnowledgeService(
        engine,
        search_engine=build_search_engine(settings, engine, vectordb=vectordb),
        fallback_to_local=(vectordb.engine != "local"),
        router=provider_router,
        pipeline_limits=PipelineLimits(
            fetch_concurrency=settings.sync_fetch_concurrency,
            fetched_queue_size=settings.sync_fetch_queue_size,
            sql_batch_documents=settings.sync_sql_batch_documents,
            sql_batch_chunks=settings.sync_sql_batch_chunks,
            progress_interval_seconds=settings.sync_progress_interval_seconds,
        ),
    )
    # Wire the control-plane reloader lazily: it reads app.state.app_state on
    # call (set just below), so tools like enable_skill_for_agent can refresh the
    # live registry + agent_config from boot without a restart.
    registry = build_default_registry(
        settings,
        agent_config=agent_config,
        on_config_change=lambda: _reload_config_from_state(app, settings),
        knowledge_service=knowledge,
    )
    if settings.skills_dir:
        load_skills(settings.skills_dir, registry)
    runtime_agent_config = apply_runtime_status(agent_config, settings, provider_router)
    # Durable background job queue: KB ingest/sync run here, off the request path,
    # surviving restarts. Recover any jobs a prior process left mid-run, then start
    # the worker pool.
    jobs = JobQueue(
        engine,
        concurrency=settings.job_worker_concurrency,
        default_max_attempts=settings.job_max_attempts,
        heartbeat_seconds=settings.job_heartbeat_seconds,
        lease_seconds=settings.job_lease_seconds,
    )
    register_knowledge_handlers(jobs, knowledge)
    jobs.start()
    app.state.app_state = AppState(
        store=store, llm=_build_llm(settings), default_model=settings.default_model,
        registry=registry, router=provider_router,
        agent_config=runtime_agent_config,
        knowledge=knowledge,
        jobs=jobs,
        memory_enabled=settings.memory_enabled,
        memory_model=settings.memory_model,
        summary_enabled=settings.summary_enabled,
        summary_keep_recent=settings.summary_keep_recent,
        summary_batch=settings.summary_batch,
        project_context=settings.project_context,
        config_store=config_store,
        config_revision=stored_config.revision,
    )
    # Register spawn_subagent tools into the boot registry (boot builds the registry
    # directly, not via rebuild_app_state_from_config, so wire them here too).
    from app.subagent import wire_subagents
    wire_subagents(app.state.app_state)
    from app.context_tools import wire_context_tools
    wire_context_tools(app.state.app_state)
    try:
        yield
    finally:
        # Stop the worker pool cleanly on shutdown (no shutdown hook existed before).
        await jobs.stop()
        await provider_router.aclose()
        # Flush any buffered trace spans before the process exits.
        try:
            from extensions.trace import shutdown_tracing
            shutdown_tracing()
        except Exception:
            pass


app = FastAPI(title="Lean Agent Service", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _http_error_handler(_request: Request, exc: HTTPException):
    """Return errors in the OpenAI API shape: ``{"error": {message, type, ...}}``."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(_request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "param": None,
                "code": None,
            }
        },
    )


app.include_router(auth_router)
app.include_router(responses_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(models_router)
app.include_router(users_router)
app.include_router(config_router)
app.include_router(files_router)
app.include_router(aliyun_router)
app.include_router(knowledge_router)
app.include_router(agents_router)
