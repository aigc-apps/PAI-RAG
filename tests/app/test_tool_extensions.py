import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.tools.registry import ToolRegistry
from agent.tools.skills import load_skills
from agent.tools.mcp import mcp_tool_to_tool, register_mcp_tools


_SKILL_SRC = '''
from agent.tools.base import Tool

async def _hello(name: str = "world"):
    return f"hello {name}"

def get_tools():
    return [Tool(name="hello", description="greet",
                 parameters={"type": "object", "properties": {"name": {"type": "string"}}},
                 fn=_hello)]
'''

_BROKEN_SRC = "this is not valid python ("


def test_load_skills_registers_tools(tmp_path):
    (tmp_path / "greet.py").write_text(_SKILL_SRC)
    (tmp_path / "broken.py").write_text(_BROKEN_SRC)
    (tmp_path / "_ignored.py").write_text("raise RuntimeError('should not load')")
    reg = ToolRegistry()
    names = load_skills(str(tmp_path), reg)
    assert "hello" in names
    assert reg.get("hello") is not None
    assert asyncio.run(reg.get("hello").fn(name="x")) == "hello x"


def test_load_skills_missing_dir_is_noop():
    reg = ToolRegistry()
    assert load_skills("/no/such/dir", reg) == []


def test_mcp_tool_to_tool_maps_schema_and_calls_client():
    calls = []

    async def call(name, args):
        calls.append((name, args))
        return {"ok": True, "echo": args}

    spec = {"name": "lookup", "description": "look up", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}
    tool = mcp_tool_to_tool(spec, call)
    assert tool.name == "lookup"
    assert tool.parameters["properties"]["q"]["type"] == "string"
    out = asyncio.run(tool.fn(q="hi"))
    assert calls == [("lookup", {"q": "hi"})]
    assert '"echo"' in out and "hi" in out  # dict result stringified as JSON


def test_register_mcp_tools_namespaces_with_prefix():
    async def call(name, args):
        return "ok"

    reg = ToolRegistry()
    specs = [{"name": "a", "description": "", "inputSchema": {"type": "object", "properties": {}}},
             {"name": "b", "description": "", "inputSchema": {"type": "object", "properties": {}}}]
    names = register_mcp_tools(specs, call, reg, prefix="srv.")
    assert names == ["srv.a", "srv.b"]
    assert reg.get("srv.a") is not None
