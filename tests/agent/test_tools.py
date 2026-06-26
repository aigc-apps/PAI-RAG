import sys, os, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.tools import Tool, ToolBox
from agent.message import ToolCall


def _tool(fn, name, return_direct=False):
    return Tool(name=name, description=name, parameters={"type": "object", "properties": {}},
                fn=fn, return_direct=return_direct)


def test_dispatch_runs_tool_and_wraps_result():
    async def echo(x: str): return f"got {x}"
    box = ToolBox([_tool(echo, "echo")])
    res = asyncio.run(box.dispatch(ToolCall(id="c1", name="echo", arguments=json.dumps({"x": "hi"}))))
    assert res.ok and "got hi" in res.content
    assert res.message.role == "tool" and res.message.tool_call_id == "c1"


def test_dispatch_unknown_tool_is_error_not_crash():
    async def echo(x: str): return x
    box = ToolBox([_tool(echo, "echo")])
    res = asyncio.run(box.dispatch(ToolCall(id="c2", name="nope", arguments="{}")))
    assert not res.ok and "Unknown tool" in res.message.content


def test_dispatch_tool_exception_becomes_error_result(monkeypatch):
    import agent.tools.base as base
    # neutralize the 1s retry waits so the test is fast
    monkeypatch.setattr(base._call_with_retry.retry, "wait", __import__("tenacity").wait_none())
    async def boom(): raise RuntimeError("kaboom")
    box = ToolBox([_tool(boom, "boom")])
    res = asyncio.run(box.dispatch(ToolCall(id="c3", name="boom", arguments="{}")))
    assert not res.ok and "kaboom" in res.error


def test_is_return_direct_flag():
    async def faq(q: str): return q
    box = ToolBox([_tool(faq, "faq", return_direct=True)])
    assert box.is_return_direct("faq") is True
    assert box.is_return_direct("missing") is False


def test_openai_schema_lists_tools():
    async def echo(x: str): return x
    box = ToolBox([_tool(echo, "echo")])
    schema = box.openai_schema()
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "echo"
    assert schema[0]["function"]["parameters"]["type"] == "object"
