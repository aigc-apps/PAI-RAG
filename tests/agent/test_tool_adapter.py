import sys, os, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from llama_index.core.tools import FunctionTool
from agent.tools import ToolBox
from agent.tools.adapter import tool_from_function_tool
from agent.message import ToolCall


def test_adapter_preserves_name_schema_and_dispatch():
    async def echo(x: str): return f"echoed {x}"
    ft = FunctionTool.from_defaults(async_fn=echo, name="echo")
    tool = tool_from_function_tool(ft)
    assert tool.name == "echo"
    schema = tool.openai_schema()
    assert schema["function"]["name"] == "echo" and "x" in json.dumps(schema["function"]["parameters"])
    box = ToolBox([tool])
    res = asyncio.run(box.dispatch(ToolCall(id="c1", name="echo", arguments=json.dumps({"x": "hi"}))))
    assert res.ok and "echoed hi" in res.content


def test_adapter_carries_return_direct():
    async def faq(q: str): return q
    ft = FunctionTool.from_defaults(async_fn=faq, name="faq", return_direct=True)
    assert tool_from_function_tool(ft).return_direct is True
