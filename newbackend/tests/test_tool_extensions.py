import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.tools.registry import ToolRegistry
from agent.tools.skills import load_skills
from agent.custom_skills import (
    discover_skill_packages,
    render_skill_instructions,
    resolve_skill_mounts,
    skill_mount_fingerprint,
)
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


def test_discover_skill_packages_and_render_matching_instructions(tmp_path):
    skill_dir = tmp_path / "report"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "id: report\n"
        "name: Report Writer\n"
        "version: 1.2.0\n"
        "description: Write structured reports.\n"
        "triggers:\n"
        "  keywords: [report]\n"
        "permissions:\n"
        "  tools: [knowledge_search]\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("Use concise sections.", encoding="utf-8")

    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    assert [package.capability_id for package in packages] == ["skill.report"]
    rendered = render_skill_instructions(
        packages=packages,
        enabled_ids=["skill.report"],
        query="make a report",
    )
    assert "# Active Skills" in rendered
    assert "Use concise sections." in rendered


def test_resolve_skill_mounts_with_nas_config(tmp_path):
    skill_dir = tmp_path / "report"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "id: report\n"
        "name: Report Writer\n"
        "version: 1.2.0\n",
        encoding="utf-8",
    )
    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    skill_config = type("SkillConfig", (), {
        "mount": {
            "mount_root": "/mnt/skills",
            "nas": {
                "server_addr": "nas-cn-hangzhou.aliyuncs.com:/",
                "remote_path_prefix": "skills",
                "read_only": True,
            },
        }
    })()

    mounts = resolve_skill_mounts(
        packages=packages,
        enabled_ids=["skill.report"],
        skill_config=skill_config,
    )

    assert mounts[0].to_dict()["mount_path"] == "/mnt/skills/report"
    assert mounts[0].nas["remotePath"] == "/skills/report@1.2.0"
    assert mounts[0].nas["mountDir"] == "/mnt/skills/report"
    assert mounts[0].nas["serverAddr"] == "nas-cn-hangzhou.aliyuncs.com:/skills/report@1.2.0"
    assert mounts[0].nas["readOnly"] is True
    assert skill_mount_fingerprint(mounts) != "none"


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
    import asyncio
    calls = []

    async def call(name, args):
        calls.append(name)
        return "ok"

    reg = ToolRegistry()
    specs = [{"name": "a", "description": "", "inputSchema": {"type": "object", "properties": {}}},
             {"name": "b", "description": "", "inputSchema": {"type": "object", "properties": {}}}]
    names = register_mcp_tools(specs, call, reg, prefix="srv.")
    assert names == ["srv.a", "srv.b"]
    assert reg.get("srv.a") is not None
    # invoking the prefixed tool must call the server with the UNPREFIXED remote name
    asyncio.run(reg.get("srv.a").fn())
    asyncio.run(reg.get("srv.b").fn())
    assert calls == ["a", "b"]
