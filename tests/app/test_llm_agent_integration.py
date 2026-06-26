# tests/app/test_llm_agent_integration.py
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.llm import LeanLLM
from agent.agent import Agent
from agent.context import AgentContext, RunVars
from agent.message import Message
from agent.tools.base import ToolBox
from agent.core.events import RunStarted, TextDelta, RunCompleted
from openai.types.completion_usage import CompletionUsage


class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, delta): self.delta = delta


class _FakeChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks): self._chunks = chunks
    def __aiter__(self):
        async def g():
            for c in self._chunks:
                yield c
        return g()


class _FakeCompletions:
    def __init__(self, chunks): self._chunks = chunks
    async def create(self, **kwargs): return _FakeStream(self._chunks)


def _fake_client(chunks):
    return type("C", (), {"chat": type("Ch", (), {"completions": _FakeCompletions(chunks)})()})()


def test_agent_run_over_lean_llm_emits_events():
    async def run():
        chunks = [
            _FakeChunk([_FakeChoice(_FakeDelta(content="Hello "))]),
            _FakeChunk([_FakeChoice(_FakeDelta(content="world"))]),
            _FakeChunk([_FakeChoice(_FakeDelta())],
                       usage=CompletionUsage(prompt_tokens=2, completion_tokens=2, total_tokens=4)),
        ]
        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = _fake_client(chunks)
        agent = Agent(llm=llm)
        ctx = AgentContext(system_prompt="be brief", history=[],
                           current_turn=Message(role="user", content="hi"),
                           attachments=[], hints=[], tools=ToolBox([]), run_vars=RunVars())
        events = [e async for e in await agent.run(ctx)]
        assert any(isinstance(e, RunStarted) for e in events)
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Hello world"
        assert any(isinstance(e, RunCompleted) for e in events)
    asyncio.run(run())
