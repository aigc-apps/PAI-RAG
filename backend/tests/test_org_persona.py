import sys, os, asyncio, types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent_config import (
    AgentConfigDocument,
    SoulConfig,
    build_soul,
    load_agent_config,
    save_agent_config,
    _merge_default,
)
from app.builder import build_context
from app.schemas import ResponsesRequest
from app.store.memory import InMemoryStore
from agent.soul import DEFAULT_SOUL


def _settings(name="MiniAgent", role="a general-purpose AI assistant"):
    return types.SimpleNamespace(agent_name=name, agent_role=role)


# --------------------------------------------------------------------------- #
# schema: the org persona is a first-class document section that round-trips
# --------------------------------------------------------------------------- #
def test_soul_config_survives_save_load(tmp_path):
    path = str(tmp_path / "config.yaml")
    doc = AgentConfigDocument()
    doc.soul = SoulConfig(name="Atlas", role="a research copilot",
                          principles=["cite sources", "show your work"])
    save_agent_config(path, doc)
    reloaded = load_agent_config(path)
    assert reloaded.soul.name == "Atlas"
    assert reloaded.soul.role == "a research copilot"
    assert reloaded.soul.principles == ["cite sources", "show your work"]


def test_merge_default_overlays_partial_soul():
    # A stored config that sets only soul.role must keep every other soul field
    # blank (= inherit DEFAULT_SOUL), not wipe the section.
    merged = _merge_default({"soul": {"role": "a billing specialist"}})
    assert merged.soul.role == "a billing specialist"
    assert merged.soul.name == ""          # untouched → inherits DEFAULT_SOUL
    assert merged.soul.identity == ""


# --------------------------------------------------------------------------- #
# build_soul: DEFAULT_SOUL  ←  doc.soul  ←  env (only when explicitly set)
# --------------------------------------------------------------------------- #
def test_build_soul_blank_doc_inherits_default():
    soul = build_soul(AgentConfigDocument(), _settings())
    assert soul.name == DEFAULT_SOUL.name
    assert soul.role == DEFAULT_SOUL.role


def test_build_soul_org_persona_overrides_default(monkeypatch):
    monkeypatch.delenv("AGENT_NAME", raising=False)
    monkeypatch.delenv("AGENT_ROLE", raising=False)
    doc = AgentConfigDocument(soul=SoulConfig(name="Atlas", role="a research copilot"))
    soul = build_soul(doc, _settings())
    assert soul.name == "Atlas"
    assert soul.role == "a research copilot"
    # unset fields still inherit DEFAULT_SOUL rather than blanking
    assert soul.identity == DEFAULT_SOUL.identity


def test_build_soul_env_overrides_org_persona_only_when_set(monkeypatch):
    doc = AgentConfigDocument(soul=SoulConfig(name="Atlas", role="a research copilot"))
    # env unset → org persona wins
    monkeypatch.delenv("AGENT_NAME", raising=False)
    assert build_soul(doc, _settings(name="MiniAgent")).name == "Atlas"
    # env set → deployment override wins for that field, role still from doc
    monkeypatch.setenv("AGENT_NAME", "EnvForced")
    soul = build_soul(doc, _settings(name="EnvForced"))
    assert soul.name == "EnvForced"
    assert soul.role == "a research copilot"


# --------------------------------------------------------------------------- #
# end to end: the base persona reaches the system prompt, and a per-agent
# persona still layers on top of it (precedence unchanged by this feature).
# --------------------------------------------------------------------------- #
def test_org_persona_reaches_prompt_and_agent_persona_overrides():
    async def run():
        doc = AgentConfigDocument(**{
            "soul": {"role": "an org-wide expert assistant"},
            "default_agent": "main",
            "agents": [{"id": "main", "name": "Main"}],
        })
        base_soul = build_soul(doc, _settings())
        req = ResponsesRequest(model="m", input="hi")
        ctx, _ = await build_context(req, InMemoryStore(), soul=base_soul, agent_config=doc)
        # org role shows up in the stable Identity block
        assert "an org-wide expert assistant" in ctx.system_prompt

        # now give the agent its own persona.role — it must win over the org base
        doc2 = AgentConfigDocument(**{
            "soul": {"role": "an org-wide expert assistant"},
            "default_agent": "main",
            "agents": [{"id": "main", "name": "Main",
                        "persona": {"role": "a narrow billing specialist"}}],
        })
        ctx2, _ = await build_context(req, InMemoryStore(), soul=base_soul, agent_config=doc2)
        assert "a narrow billing specialist" in ctx2.system_prompt
        assert "an org-wide expert assistant" not in ctx2.system_prompt

    asyncio.run(run())
