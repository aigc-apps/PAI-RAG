"""Agent profiles drive the run: per-agent tool filtering, instructions, persona
name, and pinned model — plus the user-facing GET /v1/agents roster."""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.tools.defaults import build_default_registry
from app.agent_config import AgentProfile, AgentToolsConfig
from app.builder import build_context, resolve_agent_model
from app.deps import AppState
from app.routes.agents import router as agents_router
from app.schemas import ResponsesRequest
from app.store.memory import InMemoryStore
from tests.authutil import apply_auth


class _Settings:
    search_provider = "none"


def _cfg(agents, default_agent="main", skills_root="/nonexistent"):
    return types.SimpleNamespace(
        agents=agents,
        default_agent=default_agent,
        capabilities=[],
        providers=[],
        skills=types.SimpleNamespace(root=skills_root),
    )


def test_profile_include_restricts_toolbox():
    async def run():
        reg = build_default_registry(_Settings())  # {current_datetime, web_fetch}
        cfg = _cfg([AgentProfile(id="main", name="Main", tools=AgentToolsConfig(include=["current_datetime"]))])
        ctx, _ = await build_context(
            ResponsesRequest(input="hi", agent_id="main"), InMemoryStore(),
            registry=reg, agent_config=cfg,
        )
        assert [t.name for t in ctx.tools.tools] == ["current_datetime"]
        assert "web_fetch" not in ctx.system_prompt

    asyncio.run(run())


def test_profile_exclude_subtracts_from_toolbox():
    async def run():
        reg = build_default_registry(_Settings())
        cfg = _cfg([AgentProfile(id="main", name="Main", tools=AgentToolsConfig(exclude=["web_fetch"]))])
        ctx, _ = await build_context(
            ResponsesRequest(input="hi"), InMemoryStore(), registry=reg, agent_config=cfg,
        )
        names = {t.name for t in ctx.tools.tools}
        assert "web_fetch" not in names
        assert "current_datetime" in names

    asyncio.run(run())


def test_profile_instructions_and_name_applied():
    async def run():
        reg = build_default_registry(_Settings())
        cfg = _cfg([AgentProfile(id="main", name="Helper", instructions="ALWAYS_SAY_MOO")])
        ctx, _ = await build_context(
            ResponsesRequest(input="hi"), InMemoryStore(), registry=reg, agent_config=cfg,
        )
        assert "Helper" in ctx.system_prompt          # persona name overrides the soul default
        assert "ALWAYS_SAY_MOO" in ctx.context_block  # profile instructions injected

    asyncio.run(run())


def test_agent_id_selects_the_right_profile():
    async def run():
        reg = build_default_registry(_Settings())
        cfg = _cfg(
            [
                AgentProfile(id="main", name="Main", tools=AgentToolsConfig(include=["current_datetime", "web_fetch"])),
                AgentProfile(id="fetcher", name="Fetcher", tools=AgentToolsConfig(include=["web_fetch"])),
            ],
        )
        ctx, _ = await build_context(
            ResponsesRequest(input="hi", agent_id="fetcher"), InMemoryStore(),
            registry=reg, agent_config=cfg,
        )
        assert [t.name for t in ctx.tools.tools] == ["web_fetch"]
        assert ctx.agent_id == "fetcher"

    asyncio.run(run())


def test_resolve_agent_model():
    cfg = _cfg([
        AgentProfile(id="main", name="Main", model="prov/big"),
        AgentProfile(id="lite", name="Lite", model=""),
    ])
    assert resolve_agent_model(cfg, ResponsesRequest(input="x", agent_id="main")) == "prov/big"
    assert resolve_agent_model(cfg, ResponsesRequest(input="x", agent_id="lite")) is None
    assert resolve_agent_model(None, ResponsesRequest(input="x")) is None


def test_no_agent_config_is_inert():
    async def run():
        reg = build_default_registry(_Settings())
        ctx, _ = await build_context(ResponsesRequest(input="hi"), InMemoryStore(), registry=reg)
        assert {t.name for t in ctx.tools.tools} == {"current_datetime", "web_fetch"}

    asyncio.run(run())


# --- GET /v1/agents -------------------------------------------------------- #
def _client(*, role="user", agent_config=None):
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(), llm=None, default_model="test/echo",
        agent_config=agent_config,
        soul=types.SimpleNamespace(name="MiniAgent"),
    )
    app.include_router(agents_router)
    apply_auth(app, user_id="u1", role=role)
    return TestClient(app)


def test_agents_endpoint_lists_profiles_for_regular_user():
    cfg = _cfg([
        AgentProfile(id="main", name="MiniAgent", description="general", model=""),
        AgentProfile(id="coder", name="Coder", description="writes code", model="prov/x"),
    ])
    c = _client(role="user", agent_config=cfg)
    r = c.get("/v1/agents")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["default_agent"] == "main"
    ids = [a["id"] for a in body["agents"]]
    assert ids == ["main", "coder"]
    assert body["agents"][1]["model"] == "prov/x"


def test_agents_endpoint_falls_back_to_default_when_no_config():
    c = _client(role="user", agent_config=None)
    r = c.get("/v1/agents")
    assert r.status_code == 200
    body = r.json()
    assert body["agents"] == [{"id": "main", "name": "MiniAgent", "description": "", "model": ""}]
    assert body["default_agent"] == "main"
