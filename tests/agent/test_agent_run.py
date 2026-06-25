import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
sys.path.insert(0, os.path.dirname(__file__))
from unittest.mock import MagicMock, patch
from llama_index.core.tools import FunctionTool
from common.llm.models import TextChunk, ToolResultChunk, ErrorChunk
from agent.agent import Agent
from agent.tools import ToolBox
from agent.context import AgentContext, RunVars
from agent.message import Message
from fake_llm import FakeLLM, tool_call


def _make_mock_tokenizer():
    """Return a tokenizer mock that counts whitespace-split tokens."""
    tok = MagicMock()
    tok.side_effect = lambda text, **kwargs: {
        "input_ids": text.split() if text else [],
        "offset_mapping": [(i, i + 1) for i in range(len(text.split()) if text else [])],
        "attention_mask": [],
    }
    return tok


def _ctx(tools, turn="hi"):
    return AgentContext(system_prompt="SYS", history=[], current_turn=Message("user", turn),
                        attachments=[], hints=[], tools=tools, run_vars=RunVars(current_datetime="t"))


def _collect(agent, ctx):
    async def run():
        return [c async for c in await agent.run(ctx)]
    return asyncio.run(run())


def _box(fn, name, return_direct=False):
    return ToolBox([FunctionTool.from_defaults(async_fn=fn, name=name, return_direct=return_direct)])


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_plain_text_answer_streams_and_stops(mock_get_tok):
    llm = FakeLLM([[TextChunk(delta="hello "), TextChunk(delta="world")]])
    agent = Agent(llm, max_steps=5)
    async def echo(x: str): return x
    out = _collect(agent, _ctx(_box(echo, "echo")))
    text = "".join(c.delta for c in out if type(c) is TextChunk)
    assert text == "hello world"


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_tool_call_then_final_answer(mock_get_tok):
    llm = FakeLLM([
        [TextChunk(tool_calls=[tool_call(0, "c1", "echo", json.dumps({"x": "hi"}))])],
        [TextChunk(delta="done")],
    ])
    agent = Agent(llm, max_steps=5)
    async def echo(x: str): return f"echoed {x}"
    out = _collect(agent, _ctx(_box(echo, "echo")))
    assert any(isinstance(c, ToolResultChunk) and "echoed hi" in (c.result or "") for c in out)
    assert "".join(c.delta for c in out if type(c) is TextChunk) == "done"


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_return_direct_short_circuits(mock_get_tok):
    llm = FakeLLM([[TextChunk(tool_calls=[tool_call(0, "c1", "faq", "{}")])]])
    agent = Agent(llm, max_steps=5)
    async def faq(): return json.dumps({"result": [{"content": "FAQ answer"}]})
    out = _collect(agent, _ctx(_box(faq, "faq", return_direct=True)))
    assert any("FAQ answer" in c.delta for c in out if type(c) is TextChunk)


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_max_steps_emits_notice(mock_get_tok):
    turns = [[TextChunk(tool_calls=[tool_call(0, f"c{i}", "echo", "{}")])] for i in range(3)]
    llm = FakeLLM(turns)
    agent = Agent(llm, max_steps=2)
    async def echo(): return "x"
    out = _collect(agent, _ctx(_box(echo, "echo")))
    assert any("max" in c.delta.lower() for c in out if type(c) is TextChunk)


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_idle_timeout_yields_error_chunk(mock_get_tok, monkeypatch):
    import agent.agent as agent_mod
    monkeypatch.setattr(agent_mod, "LLM_STREAM_IDLE_TIMEOUT", 0)

    class HangingLLM:
        context_window = 110000
        max_tokens = 8000
        async def astream(self, messages, tools):
            async def gen():
                await asyncio.sleep(3600)  # never yields within timeout
                yield TextChunk(delta="never")
            return gen()

    agent = Agent(HangingLLM(), max_steps=2)
    async def echo(x: str): return x
    out = _collect(agent, _ctx(_box(echo, "echo")))
    assert any(isinstance(c, ErrorChunk) for c in out)
