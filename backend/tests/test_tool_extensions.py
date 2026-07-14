import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.tools.registry import ToolRegistry
from agent.tools.skills import load_skills
from agent.custom_skills import (
    discover_skill_packages,
    find_enabled_skill_mount,
    list_skill_files,
    render_skill_catalog,
    render_skill_detail,
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


def test_discover_skill_packages_and_render_detail(tmp_path):
    skill_dir = tmp_path / "report"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "id: report\n"
        "name: Report Writer\n"
        "version: 1.2.0\n"
        "description: Write structured reports.\n"
        "permissions:\n"
        "  tools: [knowledge_search]\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("Use concise sections.", encoding="utf-8")

    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    assert [package.capability_id for package in packages] == ["skill.report"]
    # render_skill_detail is the on-demand Level-2 payload (load_skill returns it);
    # no query gating — the full body is always available once the agent asks.
    rendered = render_skill_detail(packages[0])
    assert "# Skill: Report Writer (`skill.report`)" in rendered
    assert "Version: 1.2.0" in rendered
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
    # Unversioned remote path: must match the dir install_skill writes (root/<mount_id>).
    assert mounts[0].nas["remotePath"] == "/skills/report"
    assert mounts[0].nas["mountDir"] == "/mnt/skills/report"
    assert mounts[0].nas["serverAddr"] == "nas-cn-hangzhou.aliyuncs.com:/skills/report"
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


def test_discover_skill_md_only_package_synthesizes_from_frontmatter(tmp_path):
    """A community Agent Skills package (SKILL.md frontmatter, no skill.yaml)
    must be discovered and rendered without a platform manifest."""
    skill_dir = tmp_path / "skill-creator"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: skill-creator\n"
        "description: Create new skills and improve existing ones.\n"
        "allowed-tools:\n"
        "  - knowledge_search\n"
        "metadata:\n"
        "  version: \"1.4.0\"\n"
        "---\n\n"
        "# Skill Creator\n\nDecide what the skill should do, then draft it.\n",
        encoding="utf-8",
    )

    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    assert [p.capability_id for p in packages] == ["skill.skill-creator"]
    pkg = packages[0]
    assert pkg.name == "skill-creator"
    assert pkg.version == "1.4.0"
    assert pkg.description.startswith("Create new skills")
    assert pkg.permissions == {"tools": ["knowledge_search"]}
    # instructions are the SKILL.md body, frontmatter stripped
    assert "# Skill Creator" in pkg.instructions
    assert "name: skill-creator" not in pkg.instructions

    rendered = render_skill_detail(packages[0])
    assert "# Skill: skill-creator (`skill.skill-creator`)" in rendered
    assert "Decide what the skill should do" in rendered


def test_skill_catalog_lists_enabled_skills_regardless_of_query(tmp_path):
    """A SKILL.md-only skill with no trigger keywords and a cross-language query
    that matches nothing must still appear in the always-injected catalog — this
    is the fix for community skills being invisible to the agent."""
    (tmp_path / "arch").mkdir()
    (tmp_path / "arch" / "SKILL.md").write_text(
        "---\nname: architecture-diagram\n"
        "description: Create architecture diagrams as HTML+SVG files.\n---\n\nDraw boxes.\n",
        encoding="utf-8",
    )
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "skill.yaml").write_text(
        "id: demo\nname: Demo Skill\nversion: 1.0.0\ndescription: A demo.\n",
        encoding="utf-8",
    )
    (tmp_path / "demo" / "SKILL.md").write_text("Echo a marker.", encoding="utf-8")

    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    catalog = render_skill_catalog(
        packages=packages,
        enabled_ids=["skill.architecture-diagram", "skill.demo"],
    )
    # Both skills surface even though the query never runs and arch has no triggers.
    assert "# Available Skills" in catalog
    assert "architecture-diagram" in catalog
    assert "Create architecture diagrams" in catalog
    assert "Demo Skill" in catalog


def test_skill_catalog_only_includes_enabled(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill.\n---\n\nDo alpha.\n", encoding="utf-8"
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "SKILL.md").write_text(
        "---\nname: beta\ndescription: Beta skill.\n---\n\nDo beta.\n", encoding="utf-8"
    )
    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    catalog = render_skill_catalog(packages=packages, enabled_ids=["skill.alpha"])
    assert "alpha" in catalog
    assert "beta" not in catalog


def test_skill_catalog_empty_when_no_enabled():
    assert render_skill_catalog(packages=[], enabled_ids=[]) == ""


def test_catalog_directs_agent_to_load_skill(tmp_path):
    """The always-injected catalog must tell the agent to call load_skill for the
    full instructions — otherwise it would try to act from the summary alone."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill.\n---\n\nDo alpha.\n", encoding="utf-8"
    )
    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    catalog = render_skill_catalog(packages=packages, enabled_ids=["skill.alpha"])
    assert "load_skill" in catalog
    assert "read_skill_resource" in catalog


def test_list_skill_files_lists_bundled_resources_only(tmp_path):
    skill_dir = tmp_path / "arch"
    (skill_dir / "resources").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("body", encoding="utf-8")
    (skill_dir / "skill.yaml").write_text("id: arch\nname: arch\n", encoding="utf-8")
    (skill_dir / "resources" / "template.html").write_text("<html></html>", encoding="utf-8")
    (skill_dir / "reference.md").write_text("ref", encoding="utf-8")
    # Noise that must be skipped.
    (skill_dir / "__pycache__").mkdir()
    (skill_dir / "__pycache__" / "x.pyc").write_text("", encoding="utf-8")

    files = list_skill_files(str(skill_dir))
    assert "resources/template.html" in files
    assert "reference.md" in files
    # Manifest/instruction files and junk dirs are excluded.
    assert "SKILL.md" not in files
    assert "skill.yaml" not in files
    assert not any("__pycache__" in f for f in files)


def test_render_skill_detail_includes_bundled_file_manifest(tmp_path):
    skill_dir = tmp_path / "arch"
    (skill_dir / "resources").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: architecture-diagram\ndescription: Draw diagrams.\n---\n\nStep one.\n",
        encoding="utf-8",
    )
    (skill_dir / "resources" / "template.html").write_text("<html></html>", encoding="utf-8")
    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    detail = render_skill_detail(packages[0])
    assert "Step one." in detail
    assert "## Bundled files" in detail
    assert "resources/template.html" in detail
    assert 'read_skill_resource("skill.architecture-diagram"' in detail
    # Without a mount path, no sandbox path is asserted.
    assert "/mnt/skills" not in detail


def test_render_skill_detail_states_sandbox_mount_path_when_known(tmp_path):
    skill_dir = tmp_path / "arch"
    (skill_dir / "resources").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: architecture-diagram\ndescription: Draw diagrams.\n---\n\nStep one.\n",
        encoding="utf-8",
    )
    (skill_dir / "resources" / "template.html").write_text("<html></html>", encoding="utf-8")
    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    detail = render_skill_detail(packages[0], mount_path="/mnt/skills/architecture-diagram")
    # The agent is told the exact read-only mount dir so it does not guess a path.
    assert "/mnt/skills/architecture-diagram" in detail
    assert "read-only" in detail
    assert "resources/template.html" in detail


def test_find_enabled_skill_mount_matches_with_or_without_prefix():
    mounts = [
        {"id": "skill.arch", "source_path": "/skills/arch"},
        {"id": "skill.writer", "source_path": "/skills/writer"},
    ]
    assert find_enabled_skill_mount(mounts, "arch")["source_path"] == "/skills/arch"
    assert find_enabled_skill_mount(mounts, "skill.arch")["source_path"] == "/skills/arch"
    assert find_enabled_skill_mount(mounts, "skill.nope") is None
    assert find_enabled_skill_mount([], "arch") is None


def test_discover_skill_yaml_overrides_skill_md_when_both_present(tmp_path):
    """When a dir has both, skill.yaml is authoritative (id/name/version come
    from it; SKILL.md is just the instructions body via entry.instructions)."""
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "id: demo\n"
        "name: Demo Skill\n"
        "version: 1.0.0\n"
        "description: Platform override description.\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text(
        "---\nname: ignored-md-name\nversion: 9.9.9\n---\n\nBody text.\n",
        encoding="utf-8",
    )

    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    assert len(packages) == 1
    pkg = packages[0]
    assert pkg.id == "demo"  # skill.yaml id wins
    assert pkg.name == "Demo Skill"
    assert pkg.version == "1.0.0"
    assert pkg.description == "Platform override description."
    assert pkg.instructions.strip() == "Body text."  # SKILL.md body, no frontmatter


def test_discover_skill_md_no_frontmatter_uses_dir_name(tmp_path):
    """A SKILL.md with no frontmatter is still a valid skill; id falls back to
    the directory name."""
    skill_dir = tmp_path / "plain"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Plain Skill\n\nJust markdown.\n", encoding="utf-8")

    packages = discover_skill_packages([{"type": "local", "path": str(tmp_path)}])
    assert [p.capability_id for p in packages] == ["skill.plain"]
    assert packages[0].version == "0.0.0"
    assert "Just markdown." in packages[0].instructions


def test_resolve_skill_mounts_for_skill_md_only_package(tmp_path):
    skill_dir = tmp_path / "writer"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: writer\ndescription: Write articles.\n---\n\nAlways outline first.\n",
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
        enabled_ids=["skill.writer"],
        skill_config=skill_config,
    )
    assert len(mounts) == 1
    assert mounts[0].id == "skill.writer"
    assert mounts[0].mount_path == "/mnt/skills/writer"
    assert mounts[0].nas["mountDir"] == "/mnt/skills/writer"

