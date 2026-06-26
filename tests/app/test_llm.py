# tests/app/test_llm.py
import sys, os, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.llm import LeanLLM
from common.llm.models import TextChunk, ErrorChunk


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


class _FakeUsage:
    prompt_tokens = 3
    completion_tokens = 5
    total_tokens = 8


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

    async def create(self, **kwargs):
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = type("C", (), {"completions": _FakeCompletions(chunks)})()


def test_astream_emits_text_and_usage():
    async def run():
        chunks = [
            _FakeChunk([_FakeChoice(_FakeDelta(content="Hell"))]),
            _FakeChunk([_FakeChoice(_FakeDelta(content="o"))]),
            _FakeChunk([_FakeChoice(_FakeDelta())], usage=_FakeUsage()),
        ]
        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = _FakeClient(chunks)
        out = [
            c
            async for c in llm.astream(
                messages=[{"role": "user", "content": "hi"}], tools=[]
            )
        ]
        text = "".join(c.delta for c in out)
        assert text == "Hello"
        assert any(c.usage and c.usage.total_tokens == 8 for c in out)

    asyncio.run(run())


def test_astream_coalesces_tool_calls():
    async def run():
        class _TC:
            def __init__(self, index, id=None, name=None, args=None):
                self.index = index
                self.id = id
                self.type = "function"
                self.function = type(
                    "F", (), {"name": name, "arguments": args}
                )()

        chunks = [
            _FakeChunk(
                [
                    _FakeChoice(
                        _FakeDelta(
                            tool_calls=[
                                _TC(0, id="call_1", name="get", args='{"a"')
                            ]
                        )
                    )
                ]
            ),
            _FakeChunk(
                [_FakeChoice(_FakeDelta(tool_calls=[_TC(0, args=":1}")]))]
            ),
            _FakeChunk([_FakeChoice(_FakeDelta())], usage=_FakeUsage()),
        ]
        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = _FakeClient(chunks)
        out = [c async for c in llm.astream(messages=[], tools=[])]
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
        out = [c async for c in llm.astream(messages=[], tools=[])]
        assert len(out) == 1 and isinstance(out[0], ErrorChunk)
        assert out[0].error_type == "llm"

    asyncio.run(run())
