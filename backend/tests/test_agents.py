# ruff: noqa: E402
"""Agent profiles drive the run: per-agent tool filtering, instructions (the
agent's base system prompt), and pinned model — plus the user-facing GET
/v1/agents roster."""

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


def test_profile_instructions_are_the_stable_base_prompt():
    async def run():
        reg = build_default_registry(_Settings())
        instructions = "You are Helper, a meticulous code archaeologist. ALWAYS_SAY_MOO."
        cfg = _cfg([AgentProfile(id="main", name="Helper", instructions=instructions)])
        ctx, _ = await build_context(
            ResponsesRequest(input="hi"), InMemoryStore(), registry=reg, agent_config=cfg,
        )
        # The agent's instructions markdown IS the stable (cacheable) system prompt —
        # not the volatile per-turn context block.
        assert instructions in ctx.system_prompt
        assert "ALWAYS_SAY_MOO" not in ctx.context_block

    asyncio.run(run())


def test_blank_instructions_fall_back_to_default_persona():
    async def run():
        from agent.soul import DEFAULT_INSTRUCTIONS
        reg = build_default_registry(_Settings())
        cfg = _cfg([AgentProfile(id="main", name="Main", instructions="")])
        ctx, _ = await build_context(
            ResponsesRequest(input="hi"), InMemoryStore(), registry=reg, agent_config=cfg,
        )
        assert DEFAULT_INSTRUCTIONS.strip() in ctx.system_prompt

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


# --- POST /v1/agents/{id}/code-manifest/generate --------------------------- #
def _manifest_client(state, *, role="admin"):
    app = FastAPI()
    app.state.app_state = state
    app.include_router(agents_router)
    apply_auth(app, user_id="u1", role=role)
    return TestClient(app)


def test_code_manifest_generate_409_when_agent_code_disabled():
    state = AppState(
        store=InMemoryStore(), llm=None, default_model="test/echo",
        agent_config=_cfg([AgentProfile(id="main", name="Main")]),
        registry=types.SimpleNamespace(sandbox_provider=object()),
    )
    c = _manifest_client(state)
    r = c.post("/v1/agents/main/code-manifest/generate")
    assert r.status_code == 409, r.text
    assert "not enabled" in r.json()["error"]["message"]


def test_code_manifest_generate_404_for_unknown_agent():
    state = AppState(
        store=InMemoryStore(), llm=None, default_model="test/echo",
        agent_config=_cfg([AgentProfile(id="main", name="Main")]),
        registry=types.SimpleNamespace(sandbox_provider=object()),
    )
    r = _manifest_client(state).post("/v1/agents/missing/code-manifest/generate")
    assert r.status_code == 404, r.text


def test_code_manifest_generate_503_without_sandbox_provider():
    state = AppState(
        store=InMemoryStore(), llm=None, default_model="test/echo",
        agent_config=_cfg([
            AgentProfile(id="main", name="Main", code={"enabled": True}),
        ]),
    )
    r = _manifest_client(state).post("/v1/agents/main/code-manifest/generate")
    assert r.status_code == 503, r.text
    assert "sandbox" in r.json()["error"]["message"]


def test_code_manifest_generate_requires_admin():
    state = AppState(
        store=InMemoryStore(), llm=None, default_model="test/echo",
        agent_config=_cfg([AgentProfile(id="main", name="Main")]),
    )
    c = _manifest_client(state, role="user")
    r = c.post("/v1/agents/main/code-manifest/generate")
    assert r.status_code == 403


def test_code_manifest_generate_happy_path(monkeypatch):
    from agent.core.events import TextDelta
    import app.routes.agents as agents_mod

    registry = types.SimpleNamespace(sandbox_provider=object())
    # Fake model router: one model, trivial config + llm.
    model_cfg = types.SimpleNamespace(context_window=1000, max_output_tokens=100)
    router = types.SimpleNamespace(
        default_model_id="prov/m",
        get_config=lambda mid: model_cfg,
        get_llm=lambda mid: object(),
    )
    state = AppState(
        store=InMemoryStore(), llm=None, default_model="prov/m",
        agent_config=_cfg([
            AgentProfile(id="main", name="Main", code={"enabled": True}),
        ]),
        registry=registry, router=router,
    )

    captured = {}

    async def fake_build_context(request, store, **kwargs):
        captured["instruction"] = request.input
        return object(), None

    monkeypatch.setattr(agents_mod, "build_context", fake_build_context)

    class _FakeAgent:
        max_steps = 0

        async def run(self, ctx):
            async def gen():
                yield TextDelta(text="- repo-a — the API server\n")
            return gen()

    fake_agent = _FakeAgent()
    state.make_agent = lambda **kw: fake_agent  # type: ignore[method-assign]

    c = _manifest_client(state)
    r = c.post("/v1/agents/main/code-manifest/generate")
    assert r.status_code == 200, r.text
    assert r.json() == {"manifest": "- repo-a — the API server"}
    assert '${AGENT_CODE_PATH:-/opt/code}' in captured["instruction"]
    # Exploration loop was capped, and nothing was persisted (InMemoryStore untouched).
    assert fake_agent.max_steps == agents_mod._MANIFEST_MAX_STEPS
