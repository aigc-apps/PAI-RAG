# ruff: noqa: E402
"""End-to-end test for the agent-facing knowledge_search tool.

Exercises the online query path the way the agent hits it: build a live
KnowledgeService, ingest docs, set a ToolScope (as the run loop does per turn),
invoke the tool fn, and assert formatting + permission scoping.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger

from agent.tools.builtin.knowledge import make_knowledge_search_tool
from agent.tools.scope import ToolScope, reset_current_tool_scope, set_current_tool_scope
from app.db import create_all, make_engine
from app.knowledge import KnowledgeService
from app.store.base import User


def test_knowledge_search_description_matches_prompt_trigger_contract():
    tool = make_knowledge_search_tool(object())

    assert "Before answering" in tool.description
    assert "call knowledge_search first" in tool.description
    assert "error messages or codes" in tool.description
    assert "troubleshooting" in tool.description
    assert "Do not search" in tool.description
    assert "greetings" in tool.description
    assert "identity questions" in tool.description
    assert "product or service names" in tool.description
    assert "TurboX license_check 失败" in tool.description


ALICE = User(id="u_alice", email="alice@x.io", role="user")
BOB = User(id="u_bob", email="bob@x.io", role="user")


async def _seed():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    svc = KnowledgeService(engine)

    # A public KB anyone can query.
    pub = await svc.create_kb(user=ALICE, name="Public Docs", visibility="public")
    await svc.import_text_document(
        pub.id, user=ALICE, title="安装指南",
        content="安装 PAI 平台需要先配置环境变量，然后运行安装脚本完成部署。",
        uri="docs/install",
    )
    # A private KB only Alice can query.
    priv = await svc.create_kb(user=ALICE, name="Alice Secret", visibility="private")
    await svc.import_text_document(
        priv.id, user=ALICE, title="机密调优",
        content="机密的性能调优参数与内部基准数据。",
        uri="docs/secret-tuning",
    )
    return svc, pub, priv


def test_knowledge_search_returns_ranked_snippets_with_sources():
    async def scenario():
        svc, pub, _priv = await _seed()
        tool = make_knowledge_search_tool(svc)
        out = await tool.fn(query="安装 PAI")
        return out

    token = set_current_tool_scope(ToolScope(user_id=ALICE.id, metadata={"role": "user"}))
    try:
        out = asyncio.run(scenario())
    finally:
        reset_current_tool_scope(token)

    assert "安装指南" in out
    assert "docs/install" in out  # source cited
    assert "score" in out
    assert "Document: 安装指南" in out
    assert "document_id:" in out and "chunk_id:" in out
    assert 'knowledge_read(chunk_id=…, mode="locate")' in out
    assert "[1]" not in out
    assert "Cite sources by their [n]" not in out


def test_knowledge_search_respects_kb_visibility():
    """Bob (not the owner) must not retrieve passages from Alice's private KB,
    even though the private doc is the strongest lexical match for the query."""
    async def scenario():
        svc, _pub, _priv = await _seed()
        tool = make_knowledge_search_tool(svc)
        bob_view = None
        alice_view = None

        tok_b = set_current_tool_scope(ToolScope(user_id=BOB.id, metadata={"role": "user"}))
        try:
            bob_view = await tool.fn(query="机密 调优")
        finally:
            reset_current_tool_scope(tok_b)

        tok_a = set_current_tool_scope(ToolScope(user_id=ALICE.id, metadata={"role": "user"}))
        try:
            alice_view = await tool.fn(query="机密 调优")
        finally:
            reset_current_tool_scope(tok_a)
        return bob_view, alice_view

    bob_view, alice_view = asyncio.run(scenario())
    assert "机密调优" not in bob_view          # private doc hidden from Bob
    assert "机密调优" in alice_view            # owner can retrieve it


def test_knowledge_search_explicit_kb_ids_scope():
    async def scenario():
        svc, pub, priv = await _seed()
        tool = make_knowledge_search_tool(svc)
        tok = set_current_tool_scope(ToolScope(user_id=ALICE.id, metadata={"role": "user"}))
        try:
            # restrict to the public KB → the private tuning doc must not appear
            out = await tool.fn(query="调优", kb_ids=[pub.id])
        finally:
            reset_current_tool_scope(tok)
        return out

    out = asyncio.run(scenario())
    assert "机密调优" not in out


def test_knowledge_search_soft_default_narrows_to_agent_kbs():
    """With an agent-scoped default_kb_ids and no explicit kb_ids, the search is
    confined to that base — the other accessible KB must not appear."""
    async def scenario():
        svc, pub, priv = await _seed()
        tool = make_knowledge_search_tool(svc)
        # Agent is scoped to the public KB only; Alice can reach both.
        tok = set_current_tool_scope(
            ToolScope(
                user_id=ALICE.id,
                metadata={"role": "user", "default_kb_ids": [pub.id]},
            )
        )
        try:
            out = await tool.fn(query="调优")  # no explicit kb_ids
        finally:
            reset_current_tool_scope(tok)
        return out

    out = asyncio.run(scenario())
    assert "机密调优" not in out  # private tuning doc excluded by the soft default


def test_knowledge_search_forwards_agent_rerank_and_logs_resolved_kbs():
    sentinel_query = "SENSITIVE_QUERY_MUST_NOT_APPEAR_IN_LOGS_7f3d"

    async def scenario():
        svc, pub, priv = await _seed()
        calls = []
        original_search = svc.search

        async def recording_search(**kwargs):
            calls.append(kwargs)
            return await original_search(**kwargs)

        svc.search = recording_search
        tool = make_knowledge_search_tool(svc)
        messages: list[str] = []
        sink = logger.add(messages.append, format="{message}")
        token = set_current_tool_scope(
            ToolScope(
                user_id=ALICE.id,
                metadata={
                    "role": "user",
                    "default_kb_ids": [pub.id, priv.id],
                    "knowledge_rerank": {
                        "enabled": True,
                        "model": "dashscope/rr",
                        "candidate_pool_size": 80,
                    },
                },
            )
        )
        try:
            await tool.fn(query=sentinel_query, top_k=10, mode="keyword")
        finally:
            reset_current_tool_scope(token)
            logger.remove(sink)
        return pub, priv, calls, messages

    pub, priv, calls, messages = asyncio.run(scenario())
    assert calls[-1]["kb_ids"] == [pub.id, priv.id]
    assert calls[-1]["rerank_config"]["model"] == "dashscope/rr"
    line = next(message for message in messages if "resolved_kb_ids" in message)
    assert "Calling tool knowledge_search with args:" in line
    assert pub.id in line and priv.id in line
    assert "'top_k': 10" in line
    assert "'mode': 'keyword'" in line
    assert all(sentinel_query not in message for message in messages)


def test_knowledge_search_defaults_to_ten_results():
    async def scenario():
        svc, _pub, _priv = await _seed()
        calls = []
        original_search = svc.search

        async def recording_search(**kwargs):
            calls.append(kwargs)
            return await original_search(**kwargs)

        svc.search = recording_search
        tool = make_knowledge_search_tool(svc)
        await tool.fn(query="安装 PAI")
        return calls[-1]["top_k"]

    token = set_current_tool_scope(ToolScope(user_id=ALICE.id, metadata={"role": "user"}))
    try:
        top_k = asyncio.run(scenario())
    finally:
        reset_current_tool_scope(token)

    assert top_k == 10


def test_knowledge_search_soft_default_cannot_leak_forbidden_kb():
    """A soft default pointing at a KB the user can't access must not leak it —
    the per-KB permission check still applies, so Bob gets nothing from it."""
    async def scenario():
        svc, _pub, priv = await _seed()
        tool = make_knowledge_search_tool(svc)
        # The Agent configuration defaults to Alice's private KB, but the
        # caller is Bob, who has no access to it.
        tok = set_current_tool_scope(
            ToolScope(
                user_id=BOB.id,
                metadata={"role": "user", "default_kb_ids": [priv.id]},
            )
        )
        try:
            out = await tool.fn(query="机密 调优")
        finally:
            reset_current_tool_scope(tok)
        return out

    out = asyncio.run(scenario())
    assert "机密调优" not in out  # no leak: forbidden KB filtered per-request


def test_knowledge_search_empty_default_searches_all_accessible():
    """Empty default_kb_ids preserves today's behavior: all accessible KBs."""
    async def scenario():
        svc, _pub, _priv = await _seed()
        tool = make_knowledge_search_tool(svc)
        tok = set_current_tool_scope(
            ToolScope(
                user_id=ALICE.id,
                metadata={"role": "user", "default_kb_ids": []},
            )
        )
        try:
            out = await tool.fn(query="机密 调优")
        finally:
            reset_current_tool_scope(tok)
        return out

    out = asyncio.run(scenario())
    assert "机密调优" in out  # owner still reaches the private KB when unscoped


def test_knowledge_search_empty_and_no_kb_messages():
    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine)
        tool = make_knowledge_search_tool(svc)

        tok = set_current_tool_scope(ToolScope(user_id="nobody", metadata={"role": "user"}))
        try:
            no_kb = await tool.fn(query="anything")
            blank = await tool.fn(query="   ")
        finally:
            reset_current_tool_scope(tok)
        return no_kb, blank

    no_kb, blank = asyncio.run(scenario())
    assert "No knowledge bases" in no_kb
    assert "non-empty" in blank


def test_knowledge_search_failure_redacts_query_from_logs_and_output():
    sentinel = "SENSITIVE_SEARCH_QUERY_8c31"

    class FailingService:
        async def list_kbs(self, *, user):
            raise RuntimeError(f"backend rejected query={sentinel}")

    async def scenario():
        tool = make_knowledge_search_tool(FailingService())
        messages: list[str] = []
        sink = logger.add(messages.append, format="{message}")
        token = set_current_tool_scope(
            ToolScope(user_id=ALICE.id, metadata={"role": "user"})
        )
        try:
            output = await tool.fn(query=sentinel)
        finally:
            reset_current_tool_scope(token)
            logger.remove(sink)
        return output, messages

    output, messages = asyncio.run(scenario())
    assert output == "knowledge_search failed due to an internal error."
    assert sentinel not in output
    assert messages
    assert all(sentinel not in message for message in messages)
    assert any(
        "operation=knowledge_search" in message and "error_type=RuntimeError" in message
        for message in messages
    )
