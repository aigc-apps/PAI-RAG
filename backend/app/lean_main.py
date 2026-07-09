from __future__ import annotations
import os
from contextlib import asynccontextmanager

# Load .env into os.environ before anything reads settings or secrets. This
# must run at import time, ahead of get_settings()/provider/sandbox code that
# resolves AGENTRUN_*/OPENAI_API_KEY via os.environ.get. pydantic-settings'
# env_file would only fill Settings fields, not os.environ, so we use
# load_dotenv() to cover both.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.config import get_settings
from app.deps import AppState, reload_app_state
from app.db import make_engine, create_all, migrate
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.llm import LeanLLM
from app.providers import ProviderRouter, load_catalog
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
from app.agent_config import apply_runtime_status, load_agent_config
from agent.soul import Soul
from agent.tools.defaults import build_default_registry
from agent.tools.skills import load_skills


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
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
    soul = Soul(name=settings.agent_name, role=settings.agent_role)
    agent_config = load_agent_config(settings.config_path)
    # Router before KnowledgeService: the KB service resolves its embedder/reranker
    # through the router (ingest + query). One router instance, stored on AppState.
    catalog = load_catalog(settings.models_path, settings)
    provider_router = ProviderRouter(catalog, path=settings.models_path)
    # Built before the registry so knowledge_search can bind to it; the same
    # instance is stored on AppState below and reused for the REST query routes.
    knowledge = KnowledgeService(
        engine,
        search_engine=build_search_engine(settings, engine),
        fallback_to_local=(settings.search_engine != "elasticsearch"),
        router=provider_router,
    )
    # Wire the control-plane reloader lazily: it reads app.state.app_state on
    # call (set just below), so tools like enable_skill_for_agent can refresh the
    # live registry + agent_config from boot without a restart.
    registry = build_default_registry(
        settings,
        agent_config=agent_config,
        on_config_change=lambda: reload_app_state(app.state.app_state, settings),
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
    )
    register_knowledge_handlers(jobs, knowledge)
    await jobs.recover_orphans()
    jobs.start()
    app.state.app_state = AppState(
        store=store, llm=_build_llm(settings), default_model=settings.default_model,
        soul=soul, registry=registry, router=provider_router,
        agent_config=runtime_agent_config,
        knowledge=knowledge,
        jobs=jobs,
        memory_enabled=settings.memory_enabled,
        memory_model=settings.memory_model,
        summary_enabled=settings.summary_enabled,
        summary_keep_recent=settings.summary_keep_recent,
        summary_batch=settings.summary_batch,
        project_context=settings.project_context,
    )
    try:
        yield
    finally:
        # Stop the worker pool cleanly on shutdown (no shutdown hook existed before).
        await jobs.stop()


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
