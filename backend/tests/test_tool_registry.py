import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.tools.registry import ToolRegistry
from agent.tools.base import Tool


def _tool(name):
    async def fn():
        return name
    return Tool(name=name, description=name, parameters={"type": "object", "properties": {}}, fn=fn)


def test_register_get_names():
    r = ToolRegistry()
    r.register(_tool("a"))
    r.register(_tool("b"))
    assert r.get("a").name == "a"
    assert r.get("missing") is None
    assert set(r.names()) == {"a", "b"}


def test_build_toolbox_all_and_subset_and_unknown():
    r = ToolRegistry()
    r.register(_tool("a"))
    r.register(_tool("b"))
    assert {t.name for t in r.build_toolbox().tools} == {"a", "b"}
    assert [t.name for t in r.build_toolbox(["b"]).tools] == ["b"]
    # unknown names are skipped, known ones kept
    assert [t.name for t in r.build_toolbox(["b", "nope"]).tools] == ["b"]


def test_reregister_overwrites():
    r = ToolRegistry()
    r.register(_tool("a"))
    new = _tool("a")
    r.register(new)
    assert r.get("a") is new
    assert r.names().count("a") == 1
