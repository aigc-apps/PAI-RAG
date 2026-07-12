"""The agent loop dispatches several independent tool calls from ONE assistant turn
concurrently (this is how parallel subagent fan-out works without a batch tool),
while keeping message/event order in the model's original call order. Single-call
and return_direct turns stay sequential — covered by test_agent_hitl."""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from agent.agent import Agent
from agent.context import AgentContext, RunVars
from agent.message import Message
from agent.tools.base import Tool, ToolBox
from agent.core.events import RunCompleted, ToolResult
from common.llm.models import TextChunk


def _multi_tool_chunk(*calls):
    """One streamed chunk carrying several tool calls (distinct index + id) — i.e.
    the model asked for all of them in a single turn."""
    tcs = [
        ChoiceDeltaToolCall(
            index=i, id=call_id, type="function",
            function=ChoiceDeltaToolCallFunction(name=name, arguments="{}"),
        )
        for i, (call_id, name) in enumerate(calls)
    ]
    return TextChunk(delta="", tool_calls=tcs,
                     usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))


def _text_chunk(text):
    return TextChunk(delta=text,
                     usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))


class _ScriptedLLM:
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


class _ConcurrencyProbe:
    """Shared across sibling tools: each tool enters, bumps the live counter, waits
    a beat so a sequential dispatcher couldn't overlap them, then leaves. peak==2
    proves the two calls actually overlapped."""

    def __init__(self):
        self.live = 0
        self.peak = 0

    def tool(self, name):
        async def fn(**kwargs):
            self.live += 1
            self.peak = max(self.peak, self.live)
            await asyncio.sleep(0.05)
            self.live -= 1
            return f"{name}-done"
        return Tool(name=name, description="d",
                    parameters={"type": "object", "properties": {}}, fn=fn)


def _ctx(tools):
    return AgentContext(system_prompt="be brief", history=[],
                        current_turn=Message(role="user", content="go"),
                        attachments=[], hints=[], tools=tools, run_vars=RunVars())


def test_multiple_tool_calls_in_one_turn_run_concurrently_and_in_order():
    async def run():
        probe = _ConcurrencyProbe()
        toolbox = ToolBox([probe.tool("alpha"), probe.tool("beta")])
        # Turn 1: model asks for both tools at once. Turn 2: it answers, ending the run.
        llm = _ScriptedLLM([
            [_multi_tool_chunk(("c1", "alpha"), ("c2", "beta"))],
            [_text_chunk("both done")],
        ])
        agent = Agent(llm=llm)
        events = [e async for e in await agent.run(_ctx(toolbox))]

        # Both tools overlapped in time → dispatched in parallel, not one-at-a-time.
        assert probe.peak == 2

        # ToolResult events are still emitted in the model's original call order.
        order = [e.call_id for e in events if isinstance(e, ToolResult)]
        assert order == ["c1", "c2"]
        assert all(e.ok for e in events if isinstance(e, ToolResult))

        completed = [e for e in events if isinstance(e, RunCompleted)]
        assert completed and completed[-1].finish_reason == "stop"
    asyncio.run(run())


def test_single_tool_call_still_dispatches_sequentially():
    async def run():
        probe = _ConcurrencyProbe()
        toolbox = ToolBox([probe.tool("solo")])
        llm = _ScriptedLLM([
            [_multi_tool_chunk(("c1", "solo"))],
            [_text_chunk("done")],
        ])
        agent = Agent(llm=llm)
        events = [e async for e in await agent.run(_ctx(toolbox))]
        assert probe.peak == 1  # only one call — nothing to overlap
        assert [e.call_id for e in events if isinstance(e, ToolResult)] == ["c1"]
    asyncio.run(run())
