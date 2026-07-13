"""Subagent orchestration: the clean child context (firewall), the runner, and the
spawn_subagent tool. Uses a scripted LLM so the loop is deterministic and offline
(same pattern as test_agent_hitl / test_llm_agent_integration). Parallel fan-out
lives in the agent loop, not a batch tool — see test_agent_parallel_dispatch."""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai.types.completion_usage import CompletionUsage

from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.scope import ToolScope, set_current_tool_scope, reset_current_tool_scope
from agent.tools.builtin.spawn_subagent import (
    SPAWN_TOOL_NAMES, make_spawn_subagent_tool,
)
from app.agent_config import AgentConfigDocument, AgentProfile, AgentToolsConfig, AgentKnowledgeConfig
from app.builder import build_subagent_context
from app.deps import AppState
from app.subagent import SubagentRunner
from common.llm.models import TextChunk


def _text_chunk(text):
    return TextChunk(
        delta=text,
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class _EchoLLM:
    """Emits one fixed text chunk (no tool calls) per astream() call and completes —
    so any child subagent turn ends with that text as its summary. Shared across
    concurrent children; the call counter increments atomically (no await before it)."""
    context_window = 0
    max_tokens = 0

    def __init__(self, text):
        self._text = text
        self.calls = 0

    async def astream(self, messages, tools=None, **kwargs):
        self.calls += 1
        text = self._text

        async def g():
            yield _text_chunk(text)

        return g()


def _noop_tool(name):
    async def fn(**kwargs):
        return "ok"
    return Tool(name=name, description="d",
                parameters={"type": "object", "properties": {}}, fn=fn)


def _registry(*names):
    reg = ToolRegistry()
    for n in names:
        reg.register(_noop_tool(n))
    return reg


def _state(*, registry, agent_config, llm):
    return AppState(store=None, llm=llm, default_model="m",
                    registry=registry, agent_config=agent_config, router=None)


# ---- build_subagent_context ---------------------------------------------------

def test_child_context_is_clean_and_scoped():
    reg = _registry("knowledge_search", "knowledge_find", "spawn_subagent")
    profile = AgentProfile(
        id="researcher", name="R", instructions="be sharp",
        knowledge=AgentKnowledgeConfig(kb_ids=["kb1"]),
        tools=AgentToolsConfig(include=["knowledge_search", "knowledge_find", "spawn_subagent"]),
    )
    scope = ToolScope(user_id="u1", conversation_id="c1", metadata={"aliyun_sandbox_env": {"X": "1"}})
    ctx = build_subagent_context(profile=profile, task="find X", parent_scope=scope,
                                 registry=reg, agent_config=None, depth=1)

    # Firewall: empty history, no volatile memory/summary block.
    assert ctx.history == []
    assert ctx.context_block == ""
    # The task is the live user turn.
    assert ctx.current_turn.content == "find X"
    # Scope inherited from the parent; child agent id stamped.
    assert ctx.user_id == "u1" and ctx.conversation_id == "c1"
    assert ctx.agent_id == "researcher"
    assert ctx.metadata["aliyun_sandbox_env"] == {"X": "1"}   # sandbox creds carried
    assert ctx.metadata["subagent_depth"] == 1
    assert ctx.metadata["default_kb_ids"] == ["kb1"]          # child's own KB soft-default
    # Depth cap: the spawn tools are stripped so a subagent can never nest.
    names = [t.name for t in ctx.tools.tools]
    assert "knowledge_search" in names and "knowledge_find" in names
    assert all(n not in names for n in SPAWN_TOOL_NAMES)
    # The subagent protocol is in the system prompt.
    assert "Subagent protocol" in ctx.system_prompt


# ---- SubagentRunner -----------------------------------------------------------

def test_runner_explore_worker_returns_summary():
    async def go():
        reg = _registry("knowledge_search", "grep_file", "shell", "current_datetime")
        cfg = AgentConfigDocument(agents=[])
        llm = _EchoLLM("FOUND: the auth flow lives in app/auth.py")
        runner = SubagentRunner(_state(registry=reg, agent_config=cfg, llm=llm))
        res = await runner.run(agent_id="explore", task="where is auth handled?",
                               scope=ToolScope(user_id="u1"), depth=0)
        assert res.ok
        assert res.summary == "FOUND: the auth flow lives in app/auth.py"
        assert res.finish_reason == "stop"
        assert llm.calls == 1
    asyncio.run(go())


def test_runner_unknown_agent_id():
    async def go():
        cfg = AgentConfigDocument(agents=[AgentProfile(id="main", name="M")])
        runner = SubagentRunner(_state(registry=_registry(), agent_config=cfg, llm=_EchoLLM("x")))
        res = await runner.run(agent_id="ghost", task="t", scope=ToolScope(), depth=0)
        assert not res.ok
        assert "unknown agent_id 'ghost'" in res.error
        assert "explore" in res.error  # the built-in is always offered
    asyncio.run(go())


def test_runner_refuses_to_nest():
    async def go():
        cfg = AgentConfigDocument(agents=[])
        runner = SubagentRunner(_state(registry=_registry(), agent_config=cfg, llm=_EchoLLM("x")))
        res = await runner.run(agent_id="explore", task="t", scope=ToolScope(), depth=1)
        assert not res.ok
        assert "nested subagents are not allowed" in res.error
    asyncio.run(go())


def test_runner_empty_task():
    async def go():
        cfg = AgentConfigDocument(agents=[])
        runner = SubagentRunner(_state(registry=_registry(), agent_config=cfg, llm=_EchoLLM("x")))
        res = await runner.run(agent_id="explore", task="   ", scope=ToolScope(), depth=0)
        assert not res.ok and "empty" in res.error
    asyncio.run(go())


# ---- spawn_subagent tool ------------------------------------------------------

def _run_with_scope(coro_fn, scope):
    async def go():
        token = set_current_tool_scope(scope)
        try:
            return await coro_fn()
        finally:
            reset_current_tool_scope(token)
    return asyncio.run(go())


def test_spawn_subagent_tool_returns_summary():
    reg = _registry("knowledge_search")
    cfg = AgentConfigDocument(agents=[])
    runner = SubagentRunner(_state(registry=reg, agent_config=cfg, llm=_EchoLLM("done: X")))
    tool = make_spawn_subagent_tool(runner)
    out = _run_with_scope(lambda: tool.fn(agent_id="explore", task="find X"),
                          ToolScope(user_id="u1"))
    assert out == "done: X"


def test_spawn_subagent_tool_reports_failure_as_string():
    reg = _registry()
    cfg = AgentConfigDocument(agents=[])
    runner = SubagentRunner(_state(registry=reg, agent_config=cfg, llm=_EchoLLM("x")))
    tool = make_spawn_subagent_tool(runner)
    out = _run_with_scope(lambda: tool.fn(agent_id="ghost", task="t"), ToolScope())
    assert out.startswith("[subagent 'ghost' failed]")


def test_spawn_tool_names_is_single():
    # Parallel fan-out is handled by the agent loop dispatching multiple calls
    # concurrently — there is no spawn_subagents twin tool to strip.
    assert SPAWN_TOOL_NAMES == ("spawn_subagent",)
