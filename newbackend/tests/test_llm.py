# tests/app/test_llm.py
import sys, os, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.llm import LeanLLM
from common.llm.models import TextChunk, ReasoningChunk, ErrorChunk
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _FakeStream:
    """Async-iterable stand-in for the openai streaming response."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.completions = _FakeCompletions(chunks)
        self.chat = type("C", (), {"completions": self.completions})()


def test_astream_emits_text_and_usage():
    async def run():
        chunks = [
            _FakeChunk([_FakeChoice(_FakeDelta(content="Hell"))]),
            _FakeChunk([_FakeChoice(_FakeDelta(content="o"))]),
            _FakeChunk(
                [_FakeChoice(_FakeDelta())],
                usage=CompletionUsage(
                    prompt_tokens=3, completion_tokens=5, total_tokens=8
                ),
            ),
        ]
        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = _FakeClient(chunks)
        out = [
            c
            async for c in await llm.astream(
                messages=[{"role": "user", "content": "hi"}], tools=[]
            )
        ]
        text = "".join(c.delta for c in out)
        assert text == "Hello"
        assert any(c.usage and c.usage.total_tokens == 8 for c in out)

    asyncio.run(run())


def test_astream_coalesces_tool_calls():
    async def run():
        chunks = [
            _FakeChunk(
                [
                    _FakeChoice(
                        _FakeDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=ChoiceDeltaToolCallFunction(
                                        name="get", arguments='{"a"'
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            _FakeChunk(
                [
                    _FakeChoice(
                        _FakeDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id=None,
                                    type="function",
                                    function=ChoiceDeltaToolCallFunction(
                                        name=None, arguments=":1}"
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            _FakeChunk(
                [_FakeChoice(_FakeDelta())],
                usage=CompletionUsage(
                    prompt_tokens=3, completion_tokens=5, total_tokens=8
                ),
            ),
        ]
        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = _FakeClient(chunks)
        out = [c async for c in await llm.astream(messages=[], tools=[])]
        final_calls = [c.tool_calls for c in out if c.tool_calls][-1]
        assert final_calls[0].function.name == "get"
        assert final_calls[0].function.arguments == '{"a":1}'

    asyncio.run(run())


def test_astream_error_yields_error_chunk():
    async def run():
        class _BoomCompletions:
            async def create(self, **kwargs):
                raise RuntimeError("boom")

        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = type(
            "C2",
            (),
            {"chat": type("C", (), {"completions": _BoomCompletions()})()},
        )()
        out = [c async for c in await llm.astream(messages=[], tools=[])]
        assert len(out) == 1 and isinstance(out[0], ErrorChunk)
        assert out[0].error_type == "llm"

    asyncio.run(run())


def test_astream_reasoning_content_field():
    async def run():
        chunks = [
            _FakeChunk(
                [_FakeChoice(_FakeDelta(reasoning_content="thinking"))]
            ),
            _FakeChunk(
                [_FakeChoice(_FakeDelta())],
                usage=CompletionUsage(
                    prompt_tokens=3, completion_tokens=5, total_tokens=8
                ),
            ),
        ]
        llm = LeanLLM(
            base_url="x", api_key="x", model="m", enable_thinking=True
        )
        llm.client = _FakeClient(chunks)
        out = [c async for c in await llm.astream(messages=[], tools=[])]
        reasoning = [c for c in out if isinstance(c, ReasoningChunk)]
        assert len(reasoning) == 1
        assert reasoning[0].reasoning_delta == "thinking"

    asyncio.run(run())


def test_astream_think_tags_split():
    async def run():
        chunks = [
            _FakeChunk([_FakeChoice(_FakeDelta(content="<think>"))]),
            _FakeChunk([_FakeChoice(_FakeDelta(content="plan"))]),
            _FakeChunk([_FakeChoice(_FakeDelta(content="</think>"))]),
            _FakeChunk([_FakeChoice(_FakeDelta(content="answer"))]),
            _FakeChunk(
                [_FakeChoice(_FakeDelta())],
                usage=CompletionUsage(
                    prompt_tokens=3, completion_tokens=5, total_tokens=8
                ),
            ),
        ]
        llm = LeanLLM(
            base_url="x", api_key="x", model="m", enable_thinking=True
        )
        llm.client = _FakeClient(chunks)
        out = [c async for c in await llm.astream(messages=[], tools=[])]
        reasoning_text = "".join(
            c.reasoning_delta
            for c in out
            if isinstance(c, ReasoningChunk)
        )
        answer_text = "".join(
            c.delta
            for c in out
            if isinstance(c, TextChunk)
        )
        assert "plan" in reasoning_text
        assert answer_text == "answer"

    asyncio.run(run())


def test_enable_thinking_passed_to_api():
    async def run():
        chunks = [
            _FakeChunk(
                [_FakeChoice(_FakeDelta())],
                usage=CompletionUsage(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2
                ),
            ),
        ]
        llm = LeanLLM(
            base_url="x", api_key="x", model="m", enable_thinking=True
        )
        fake = _FakeClient(chunks)
        llm.client = fake
        _ = [c async for c in await llm.astream(messages=[], tools=[])]
        kwargs = fake.completions.last_kwargs
        assert kwargs is not None
        extra_body = kwargs.get("extra_body")
        assert extra_body is not None
        assert extra_body.get("enable_thinking") is True
        assert (
            extra_body.get("chat_template_kwargs", {}).get("enable_thinking")
            is True
        )

    asyncio.run(run())
