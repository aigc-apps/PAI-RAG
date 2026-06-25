import sys, os, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from llama_index.core.tools import FunctionTool
from agent.tools import ToolBox
from agent.message import ToolCall


def _box(fn, name, return_direct=False):
    tool = FunctionTool.from_defaults(async_fn=fn, name=name, return_direct=return_direct)
    return ToolBox([tool])


def test_dispatch_runs_tool_and_wraps_result():
    async def echo(x: str): return f"got {x}"
    box = _box(echo, "echo")
    tc = ToolCall(id="c1", name="echo", arguments=json.dumps({"x": "hi"}))
    res = asyncio.run(box.dispatch(tc))
    assert res.ok and "got hi" in res.content
    assert res.message.role == "tool" and res.message.tool_call_id == "c1"


def test_dispatch_unknown_tool_is_error_not_crash():
    async def echo(x: str): return x
    box = _box(echo, "echo")
    res = asyncio.run(box.dispatch(ToolCall(id="c2", name="nope", arguments="{}")))
    assert not res.ok and "Unknown tool" in res.message.content


def test_is_return_direct_flag():
    async def faq(q: str): return q
    box = _box(faq, "faq", return_direct=True)
    assert box.is_return_direct("faq") is True
    assert box.is_return_direct("missing") is False


def test_openai_schema_lists_tools():
    async def echo(x: str): return x
    box = _box(echo, "echo")
    schema = box.openai_schema()
    assert schema[0]["function"]["name"] == "echo"
