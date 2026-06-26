from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import get_settings
from app.deps import AppState
from app.db import make_engine, create_all
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.llm import LeanLLM
from app.routes.responses import router as responses_router
from app.routes.chat import router as chat_router
from app.routes.conversations import router as conversations_router
from agent.soul import Soul
from agent.tools.defaults import build_default_registry
from agent.tools.skills import load_skills


def _build_llm(settings) -> LeanLLM:
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
    registry = build_default_registry(settings)
    if settings.skills_dir:
        load_skills(settings.skills_dir, registry)
    app.state.app_state = AppState(
        store=store, llm=_build_llm(settings), default_model=settings.default_model,
        soul=soul, registry=registry,
    )
    yield


app = FastAPI(title="Lean Agent Service", lifespan=lifespan)
app.include_router(responses_router)
app.include_router(chat_router)
app.include_router(conversations_router)
