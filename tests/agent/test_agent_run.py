import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
sys.path.insert(0, os.path.dirname(__file__))
from unittest.mock import MagicMock, patch
from common.llm.models import TextChunk, ErrorChunk
from agent.agent import Agent
from agent.tools import Tool, ToolBox
from agent.context import AgentContext, RunVars
from agent.message import Message
from fake_llm import FakeLLM, tool_call
from agent.core.events import (
    TextDelta, ToolStarted, ToolCompleted, ToolResult, RunCompleted, RunFailed,
)
from openai.types.chat.chat_completion_chunk import CompletionUsage


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
    return ToolBox([Tool(name=name, description=name,
                         parameters={"type": "object", "properties": {}},
                         fn=fn, return_direct=return_direct)])


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_text_only_emits_textdeltas_then_completed(mock_get_tok):
    usage = CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    llm = FakeLLM([[TextChunk(delta="he"), TextChunk(delta="llo"), TextChunk(delta="", usage=usage)]])
    agent = Agent(llm, max_steps=5)
    async def echo(x: str): return x
    out = _collect(agent, _ctx(_box(echo, "echo")))
    assert "".join(e.text for e in out if isinstance(e, TextDelta)) == "hello"
    completed = [e for e in out if isinstance(e, RunCompleted)]
    assert len(completed) == 1 and completed[0].usage.output == 2


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_tool_call_emits_started_completed_result(mock_get_tok):
    llm = FakeLLM([
        [TextChunk(tool_calls=[tool_call(0, "c1", "echo", json.dumps({"x": "hi"}))])],
        [TextChunk(delta="done")],
    ])
    agent = Agent(llm, max_steps=5)
    async def echo(x: str): return f"echoed {x}"
    out = _collect(agent, _ctx(_box(echo, "echo")))
    types = [type(e).__name__ for e in out]
    assert "ToolStarted" in types and "ToolCompleted" in types and "ToolResult" in types
    tr = [e for e in out if isinstance(e, ToolResult)][0]
    assert tr.ok and "echoed hi" in tr.output and tr.call_id == "c1"
    assert "".join(e.text for e in out if isinstance(e, TextDelta)) == "done"


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_idle_timeout_emits_run_failed(mock_get_tok, monkeypatch):
    import agent.agent as agent_mod
    monkeypatch.setattr(agent_mod, "LLM_STREAM_IDLE_TIMEOUT", 0)

    class HangingLLM:
        context_window = 110000
        max_tokens = 8000
        async def astream(self, messages, tools):
            async def gen():
                await asyncio.sleep(3600)
                yield TextChunk(delta="never")
            return gen()

    agent = Agent(HangingLLM(), max_steps=2)
    async def echo(x: str): return x
    out = _collect(agent, _ctx(_box(echo, "echo")))
    failed = [e for e in out if isinstance(e, RunFailed)]
    assert len(failed) == 1 and failed[0].error_type == "llm_stream_timeout"


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_return_direct_emits_textdelta_then_completed(mock_get_tok):
    llm = FakeLLM([[TextChunk(tool_calls=[tool_call(0, "c1", "faq", "{}")])]])
    agent = Agent(llm, max_steps=5)
    async def faq(): return json.dumps({"result": [{"content": "FAQ answer"}]})
    out = _collect(agent, _ctx(_box(faq, "faq", return_direct=True)))
    assert any(isinstance(e, TextDelta) and "FAQ answer" in e.text for e in out)
    assert any(isinstance(e, RunCompleted) for e in out)


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_max_steps_emits_completed_with_incomplete_reason(mock_get_tok):
    turns = [[TextChunk(tool_calls=[tool_call(0, f"c{i}", "echo", "{}")])] for i in range(3)]
    agent = Agent(FakeLLM(turns), max_steps=2)
    async def echo(): return "x"
    out = _collect(agent, _ctx(_box(echo, "echo")))
    completed = [e for e in out if isinstance(e, RunCompleted)]
    assert len(completed) == 1 and completed[0].finish_reason == "max_steps"
