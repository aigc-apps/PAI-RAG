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
from app.db import make_engine, create_all
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.llm import LeanLLM
from app.providers import ProviderRouter, load_catalog
from app.routes.responses import router as responses_router
from app.routes.chat import router as chat_router
from app.routes.conversations import router as conversations_router
from app.routes.models import router as models_router
from app.routes.users import router as users_router
from app.routes.config import router as config_router
from app.routes.files import router as files_router
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
    if settings.store_backend == "memory":
        store = InMemoryStore()
    else:
        # ensure the data dir exists for file-based sqlite
        if (
            settings.db_url.startswith("sqlite")
            and ":memory:" not in settings.db_url
        ):
            os.makedirs("./data", exist_ok=True)
        # Build ONE engine and reuse it for both create_all and the store so a
        # `:memory:` DB (one connection = one DB) is shared, not duplicated.
        engine = make_engine(settings.db_url)
        await create_all(engine)
        store = SqlStore(engine)
    soul = Soul(name=settings.agent_name, role=settings.agent_role)
    agent_config = load_agent_config(settings.config_path)
    # Wire the control-plane reloader lazily: it reads app.state.app_state on
    # call (set just below), so tools like enable_skill_for_agent can refresh the
    # live registry + agent_config from boot without a restart.
    registry = build_default_registry(
        settings,
        agent_config=agent_config,
        on_config_change=lambda: reload_app_state(app.state.app_state, settings),
    )
    if settings.skills_dir:
        load_skills(settings.skills_dir, registry)
    catalog = load_catalog(settings.models_path, settings)
    provider_router = ProviderRouter(catalog, path=settings.models_path)
    runtime_agent_config = apply_runtime_status(agent_config, settings, provider_router)
    app.state.app_state = AppState(
        store=store, llm=_build_llm(settings), default_model=settings.default_model,
        soul=soul, registry=registry, router=provider_router,
        agent_config=runtime_agent_config,
        memory_enabled=settings.memory_enabled,
        memory_model=settings.memory_model,
        summary_enabled=settings.summary_enabled,
        summary_keep_recent=settings.summary_keep_recent,
        summary_batch=settings.summary_batch,
        project_context=settings.project_context,
    )
    yield


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


app.include_router(responses_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(models_router)
app.include_router(users_router)
app.include_router(config_router)
app.include_router(files_router)
