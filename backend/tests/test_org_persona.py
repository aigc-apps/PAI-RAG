import sys, os, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent_config import (
    AgentConfigDocument,
    AgentProfile,
    load_agent_config,
    save_agent_config,
    _merge_default,
)
from app.builder import build_context
from app.schemas import ResponsesRequest
from app.store.memory import InMemoryStore
from agent.soul import DEFAULT_INSTRUCTIONS


# --------------------------------------------------------------------------- #
# schema: the "Default Persona" template is a first-class scalar that round-trips
# --------------------------------------------------------------------------- #
def test_default_instructions_survives_save_load(tmp_path):
    path = str(tmp_path / "config.yaml")
    doc = AgentConfigDocument()
    doc.default_instructions = "# House voice\nYou are Atlas, a research copilot."
    save_agent_config(path, doc)
    reloaded = load_agent_config(path)
    assert reloaded.default_instructions == "# House voice\nYou are Atlas, a research copilot."


def test_merge_default_overlays_default_instructions():
    merged = _merge_default({"default_instructions": "You are a billing specialist."})
    assert merged.default_instructions == "You are a billing specialist."
    # A config that doesn't set it inherits the shipped, non-blank template.
    assert _merge_default({}).default_instructions == DEFAULT_INSTRUCTIONS
    # A stored BLANK value (older configs persisted "") also inherits the template
    # rather than blanking the "Default Persona" box.
    assert _merge_default({"default_instructions": ""}).default_instructions == DEFAULT_INSTRUCTIONS
    assert _merge_default({"default_instructions": "   "}).default_instructions == DEFAULT_INSTRUCTIONS


# --------------------------------------------------------------------------- #
# seeding: a new agent copies the template into its own instructions at creation
# (snapshot — a later template edit never reaches the agent).
# --------------------------------------------------------------------------- #
def test_new_agent_seeds_instructions_from_template_snapshot():
    doc = AgentConfigDocument(default_instructions="House voice.")
    # Mirror the frontend newAgentProfile(): snapshot-copy the template at creation.
    created = AgentProfile(id="agent-2", name="New agent 2",
                           instructions=doc.default_instructions or "")
    assert created.instructions == "House voice."
    # Editing the template afterwards must not touch the already-created agent.
    doc.default_instructions = "Changed."
    assert created.instructions == "House voice."


# --------------------------------------------------------------------------- #
# end to end: the agent's own instructions ARE the stable base system prompt;
# blank falls back to DEFAULT_INSTRUCTIONS (no org-persona merge anymore).
# --------------------------------------------------------------------------- #
def test_agent_instructions_reach_prompt_and_blank_falls_back():
    async def run():
        doc = AgentConfigDocument(**{
            "default_agent": "main",
            "agents": [{"id": "main", "name": "Main",
                        "instructions": "You are a narrow billing specialist."}],
        })
        req = ResponsesRequest(model="m", input="hi")
        ctx, _ = await build_context(req, InMemoryStore(), agent_config=doc)
        assert "You are a narrow billing specialist." in ctx.system_prompt

        # A blank agent falls back to the built-in default persona.
        blank = AgentConfigDocument(**{
            "default_agent": "main",
            "agents": [{"id": "main", "name": "Main", "instructions": ""}],
        })
        ctx2, _ = await build_context(req, InMemoryStore(), agent_config=blank)
        assert DEFAULT_INSTRUCTIONS.strip() in ctx2.system_prompt

    asyncio.run(run())
