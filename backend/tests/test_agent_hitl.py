"""Human-in-the-loop halt: a tool that emits an ``interrupt`` notice stops the
agent turn after the current tool batch, instead of re-entering the LLM."""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from agent.agent import Agent, _HITL_PAUSE_TEXT
from agent.context import AgentContext, RunVars
from agent.message import Message
from agent.tools.base import Tool, ToolBox
from agent.tools.artifacts import emit_tool_notice
from agent.core.events import TextDelta, RunCompleted, ToolResult
from common.llm.models import TextChunk


def _tool_call_chunk(call_id, name):
    tc = ChoiceDeltaToolCall(
        index=0, id=call_id, type="function",
        function=ChoiceDeltaToolCallFunction(name=name, arguments="{}"),
    )
    return TextChunk(delta="", tool_calls=[tc],
                     usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))


class _ScriptedLLM:
    """Yields one pre-baked chunk batch per astream() call; counts the calls so a
    test can assert the loop did (or did not) re-enter the model."""

    context_window = 0
    max_tokens = 0

    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = 0

    async def astream(self, messages, tools=None, **kwargs):
        batch = self._batches[self.calls]
        self.calls += 1

        async def g():
            for c in batch:
                yield c
        return g()


def _pause_tool():
    async def fn(**kwargs):
        emit_tool_notice({"kind": "test_hitl", "interrupt": True})
        return "needs authorization"
    return Tool(name="pause_tool", description="d",
                parameters={"type": "object", "properties": {}}, fn=fn)


def _plain_tool():
    async def fn(**kwargs):
        return "ok"
    return Tool(name="plain_tool", description="d",
                parameters={"type": "object", "properties": {}}, fn=fn)


def _ctx(tools):
    return AgentContext(system_prompt="be brief", history=[],
                        current_turn=Message(role="user", content="hi"),
                        attachments=[], hints=[], tools=tools, run_vars=RunVars())


def test_interrupt_notice_halts_turn_without_reentering_llm():
    async def run():
        llm = _ScriptedLLM([[_tool_call_chunk("c1", "pause_tool")]])
        agent = Agent(llm=llm)
        events = [e async for e in await agent.run(_ctx(ToolBox([_pause_tool()])))]

        # The model was asked exactly once — the loop did NOT re-enter it.
        assert llm.calls == 1
        # The tool result carried the interrupt notice.
        assert any(isinstance(e, ToolResult) and (e.notice or {}).get("interrupt")
                   for e in events)
        # A deterministic, persistable pause line was streamed.
        assert any(isinstance(e, TextDelta) and e.text == _HITL_PAUSE_TEXT
                   for e in events)
        completed = [e for e in events if isinstance(e, RunCompleted)]
        assert completed and completed[-1].finish_reason == "awaiting_user"
    asyncio.run(run())


def test_plain_tool_result_reenters_llm_as_usual():
    async def run():
        llm = _ScriptedLLM([
            [_tool_call_chunk("c1", "plain_tool")],
            [TextChunk(delta="done",
                       usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))],
        ])
        agent = Agent(llm=llm)
        events = [e async for e in await agent.run(_ctx(ToolBox([_plain_tool()])))]

        # No interrupt → the loop re-entered the model for a second turn.
        assert llm.calls == 2
        assert not any(isinstance(e, TextDelta) and e.text == _HITL_PAUSE_TEXT
                       for e in events)
        completed = [e for e in events if isinstance(e, RunCompleted)]
        assert completed and completed[-1].finish_reason == "stop"
    asyncio.run(run())
