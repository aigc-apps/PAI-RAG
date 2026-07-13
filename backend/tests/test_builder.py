# ruff: noqa: E402
# tests/app/test_builder.py
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.schemas import ResponsesRequest
from app.builder import build_context, items_to_messages
from app.agent_config import AgentConfigDocument
from app.store.memory import InMemoryStore
from app.store.base import Item, StoredResponse


def test_request_ignores_unknown_fields_and_parses_input():
    req = ResponsesRequest(
        model="m", input="hello", enable_agent=True, kb_ids=["k1"]
    )
    assert req.input == "hello" and req.model == "m"


def test_build_context_from_string_input():
    async def run():
        st = InMemoryStore()
        req = ResponsesRequest(
            model="m", input="hi there", instructions="be terse"
        )
        ctx, conv_id = await build_context(req, st)
        assert ctx.current_turn.role == "user"
        assert ctx.current_turn.content == "hi there"
        # instructions now live in the volatile context block, not the stable system prompt
        assert "be terse" in ctx.context_block
        assert "# Additional instructions" in ctx.context_block
        assert "be terse" not in ctx.system_prompt
        # The default persona (the base prompt) + the always-on Tools section.
        assert "# Tools" in ctx.system_prompt
        assert ctx.history == []
        assert conv_id is not None

    asyncio.run(run())


def test_build_context_injects_agent_knowledge_rerank_policy():
    async def run():
        doc = AgentConfigDocument(
            agents=[
                {
                    "id": "main",
                    "name": "Main",
                    "knowledge": {
                        "kb_ids": ["kb_a", "kb_b"],
                        "rerank": {
                            "enabled": True,
                            "model": "dashscope/rr",
                            "candidate_pool_size": 80,
                        },
                    },
                }
            ]
        )
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="turbox"),
            InMemoryStore(),
            agent_config=doc,
        )
        assert ctx.metadata["default_kb_ids"] == ["kb_a", "kb_b"]
        assert ctx.metadata["knowledge_rerank"] == {
            "enabled": True,
            "model": "dashscope/rr",
            "candidate_pool_size": 80,
        }

    asyncio.run(run())


def test_build_context_injects_matching_custom_skill(tmp_path):
    skill_dir = tmp_path / "writer"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "id: writer\n"
        "name: Writer\n"
        "description: Write articles.\n"
        "triggers:\n"
        "  keywords: [article]\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("Always produce an outline first.", encoding="utf-8")

    async def run():
        doc = AgentConfigDocument(**{
            "skills": {
                "root": str(tmp_path),
                "mount": {"mount_root": "/mnt/skills"},
            },
            "agents": [{
                "id": "main",
                "name": "Main",
                "skills": {"enabled": ["skill.writer"]},
            }],
            "capabilities": [{
                "id": "skill.writer",
                "kind": "skill",
                "name": "Writer",
                "enabled": True,
                "status": "ready",
            }],
        })
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="write an article about agents"),
            InMemoryStore(),
            agent_config=doc,
        )
        # Progressive disclosure: only the L1 catalog is injected. The full skill
        # body is NOT preloaded — the agent pulls it on demand via load_skill.
        assert "# Available Skills" in ctx.context_block
        assert "Writer" in ctx.context_block
        assert "Always produce an outline first." not in ctx.context_block
        assert ctx.agent_id == "main"
        assert ctx.skill_mounts == [{
            "id": "skill.writer",
            "version": "0.0.0",
            "source_path": str(skill_dir),
            "mount_path": "/mnt/skills/writer",
            "read_only": True,
            "nas": {},
        }]
        assert ctx.skill_fingerprint != "none"

    asyncio.run(run())


def test_build_context_catalogs_skill_md_only_skill_without_query_match(tmp_path):
    """The bug: a community SKILL.md-only skill (no trigger keywords) was invisible
    to the agent when the query didn't substring-match its English name/desc. The
    always-injected catalog must surface it regardless."""
    skill_dir = tmp_path / "architecture-diagram"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: architecture-diagram\n"
        "description: Create architecture diagrams as HTML+SVG files.\n---\n\nDraw boxes.\n",
        encoding="utf-8",
    )

    async def run():
        doc = AgentConfigDocument(**{
            "skills": {"root": str(tmp_path), "mount": {"mount_root": "/mnt/skills"}},
            "agents": [{
                "id": "main",
                "name": "Main",
                "skills": {"enabled": ["skill.architecture-diagram"]},
            }],
            "capabilities": [{
                "id": "skill.architecture-diagram",
                "kind": "skill",
                "name": "architecture-diagram",
                "enabled": True,
                "status": "ready",
            }],
        })
        # A cross-language query that matches nothing in the skill's English text.
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="你有哪些技能"),
            InMemoryStore(),
            agent_config=doc,
        )
        assert "# Available Skills" in ctx.context_block
        assert "architecture-diagram" in ctx.context_block
        assert "Create architecture diagrams" in ctx.context_block
        # No query match -> full instructions NOT injected, only the catalog line.
        assert "Draw boxes." not in ctx.context_block

    asyncio.run(run())


def test_build_context_resolves_previous_response_history():
    async def run():
        st = InMemoryStore()
        conv = await st.create_conversation()
        await st.append_items(
            conv.id,
            [
                Item(
                    type="message",
                    role="user",
                    content={"text": "q1"},
                    response_id="resp_1",
                ),
                Item(
                    type="message",
                    role="assistant",
                    content={"text": "a1"},
                    response_id="resp_1",
                ),
            ],
        )
        await st.save_response(
            StoredResponse(
                id="resp_1",
                conversation_id=conv.id,
                model="m",
                status="completed",
            )
        )
        req = ResponsesRequest(
            model="m", input="q2", previous_response_id="resp_1"
        )
        ctx, conv_id = await build_context(req, st)
        assert conv_id == conv.id
        assert [m.role for m in ctx.history] == ["user", "assistant"]
        assert ctx.history[0].content == "q1"
        assert ctx.current_turn.content == "q2"

    asyncio.run(run())


def test_items_to_messages_handles_function_call_and_output():
    msgs = items_to_messages(
        [
            Item(
                type="function_call",
                content={"call_id": "c1", "name": "get", "arguments": "{}"},
            ),
            Item(
                type="function_call_output",
                content={"call_id": "c1", "output": "42"},
            ),
        ]
    )
    assert msgs[0].role == "assistant" and msgs[0].tool_calls[0].id == "c1"
    assert (
        msgs[1].role == "tool"
        and msgs[1].tool_call_id == "c1"
        and msgs[1].content == "42"
    )


def test_build_context_from_list_input_takes_last_user_text():
    async def run():
        st = InMemoryStore()
        req = ResponsesRequest(model="m", input=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ])
        ctx, conv_id = await build_context(req, st)
        assert ctx.current_turn.role == "user"
        assert ctx.current_turn.content == "second"
        assert conv_id is not None

    asyncio.run(run())


def test_build_context_conflicting_ids_raises_value_error():
    async def run():
        st = InMemoryStore()
        c1 = await st.create_conversation()
        c2 = await st.create_conversation()
        await st.save_response(
            StoredResponse(
                id="resp_x",
                conversation_id=c1.id,
                model="m",
                status="completed",
            )
        )
        req = ResponsesRequest(
            model="m",
            input="q",
            previous_response_id="resp_x",
            conversation=c2.id,
        )
        import pytest

        with pytest.raises(ValueError):
            await build_context(req, st)

    asyncio.run(run())


def test_build_context_renders_default_persona_into_system_prompt():
    async def run():
        from agent.soul import DEFAULT_INSTRUCTIONS
        st = InMemoryStore()
        req = ResponsesRequest(model="m", input="hi")
        ctx, _ = await build_context(req, st)
        # With no agent config, the built-in default persona is the base prompt.
        assert DEFAULT_INSTRUCTIONS.strip() in ctx.system_prompt
        # no tools wired in this plan
        assert "no tools" in ctx.system_prompt.lower()

    asyncio.run(run())
