# Sandbox-Only Skill Resources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep `SKILL.md` loading host-side while moving all other skill-file discovery, reading, and execution exclusively into the sandbox mount.

**Architecture:** `load_skill` remains the sole host-side progressive-disclosure tool and returns instructions plus the exact read-only mount path. Remove `read_skill_resource` and bundled-file enumeration from every runtime, prompt, frontend, test, and active-document surface.

**Tech Stack:** Python 3.12 backend with pytest; React 19 and TypeScript 5.6 frontend with Vitest.

## Global Constraints

- The backend may read only skill metadata and `SKILL.md` instructions through `load_skill`.
- Bundled resource discovery, reading, and execution happen only through `shell` or `code_interpreter` under `/mnt/skills/<mount-id>`.
- `load_skill` must not enumerate bundled files.
- There is no host-side resource fallback when sandbox access is unavailable.
- Do not change skill installation, authorization, Agent enablement, or NAS mount construction.
- Preserve unknown-tool fallback in the frontend.

---

### Task 1: Skill Catalog and Detail Contract

**Files:**
- Modify: `backend/tests/test_tool_extensions.py`
- Modify: `backend/tests/test_builtin_tools.py`
- Modify: `backend/agent/custom_skills.py`
- Modify: `backend/agent/tools/builtin/load_skill.py`

**Interfaces:**
- Consumes: `SkillPackage`, enabled skill ids, and the optional `mount_path` already present in `ToolScope.skill_mounts`.
- Produces: `render_skill_detail(package, mount_path=None) -> str` containing `SKILL.md` instructions and optional sandbox guidance, with no file manifest.

- [ ] **Step 1: Rewrite the focused tests for the new contract**

In `backend/tests/test_tool_extensions.py`:

- Remove the `list_skill_files` import and `test_list_skill_files_lists_bundled_resources_only`.
- Change the catalog test to assert `load_skill` is present and `read_skill_resource` is absent.
- Replace the two detail tests with:

```python
def test_render_skill_detail_does_not_enumerate_bundled_files(tmp_path):
    skill_dir = tmp_path / "arch"
    (skill_dir / "resources").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: architecture-diagram\ndescription: Draw diagrams.\n---\n\nStep one.\n",
        encoding="utf-8",
    )
    (skill_dir / "resources" / "template.html").write_text(
        "<html></html>", encoding="utf-8"
    )
    package = discover_skill_packages(
        [{"type": "local", "path": str(tmp_path)}]
    )[0]

    detail = render_skill_detail(package)

    assert "Step one." in detail
    assert "resources/template.html" not in detail
    assert "read_skill_resource" not in detail
    assert "/mnt/skills" not in detail


def test_render_skill_detail_directs_resource_access_to_exact_sandbox_mount(
    tmp_path,
):
    skill_dir = tmp_path / "arch"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: architecture-diagram\ndescription: Draw diagrams.\n---\n\nStep one.\n",
        encoding="utf-8",
    )
    package = discover_skill_packages(
        [{"type": "local", "path": str(tmp_path)}]
    )[0]

    detail = render_skill_detail(
        package, mount_path="/mnt/skills/architecture-diagram"
    )

    assert "/mnt/skills/architecture-diagram" in detail
    assert "read-only" in detail
    assert "shell" in detail
    assert "code_interpreter" in detail
    assert "list" in detail
    assert "read_skill_resource" not in detail
```

In `backend/tests/test_builtin_tools.py`, rename `test_load_skill_returns_full_instructions_and_file_manifest` to `test_load_skill_returns_instructions_and_sandbox_mount_guidance` and assert:

```python
assert "# Skill: architecture-diagram" in out
assert "Copy the template at resources/template.html." in out
assert "/mnt/skills/architecture-diagram" in out
assert "shell" in out
assert "resources/template.html\n-" not in out
assert "read_skill_resource" not in out
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/test_tool_extensions.py tests/test_builtin_tools.py -q
```

Expected: detail and catalog assertions FAIL because the current renderer emits a host-side manifest and recommends `read_skill_resource`.

- [ ] **Step 3: Remove host-side file enumeration and update guidance**

In `backend/agent/custom_skills.py`:

- Update `render_skill_catalog` to describe only L1 catalog and L2 `load_skill`, with this resource guidance:

```python
"from the summary alone. Skills may bundle extra files (templates, references, "
"scripts); `load_skill` gives you the exact read-only sandbox directory. Use "
"shell or code_interpreter there to discover, read, or run those files.",
```

- Delete `_SKILL_META_FILES`, `_SKILL_SKIP_DIRS`, and `list_skill_files`.
- Implement the detail suffix without scanning the package directory:

```python
def render_skill_detail(
    package: SkillPackage, mount_path: Optional[str] = None
) -> str:
    """Render host-loaded SKILL.md instructions and optional sandbox access guidance."""
    body = package.instructions.strip() or package.description.strip()
    parts = [
        f"# Skill: {package.name} (`{package.capability_id}`)",
        f"Version: {package.version}",
        "",
        body,
    ]
    if mount_path:
        parts.extend(
            [
                "",
                "## Sandbox files",
                f"This skill's bundled files are mounted read-only at `{mount_path}`. "
                "Use shell or code_interpreter in that exact directory to list, read, "
                "or run them; start by listing the directory instead of guessing paths.",
            ]
        )
    return "\n".join(parts)
```

- Change `find_enabled_skill_mount` documentation to say it authorizes `load_skill` only.

In `backend/agent/tools/builtin/load_skill.py`, update its docstring and description so they promise full `SKILL.md` instructions and the exact sandbox directory, not a bundled-file list.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command from Step 2.

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the rendering contract**

```bash
git add backend/agent/custom_skills.py backend/agent/tools/builtin/load_skill.py backend/tests/test_tool_extensions.py backend/tests/test_builtin_tools.py
git commit -m "refactor(skills): move resource discovery into sandbox"
```

---

### Task 2: Remove the Host-Side Resource Tool

**Files:**
- Delete: `backend/agent/tools/builtin/read_skill_resource.py`
- Modify: `backend/agent/tools/defaults.py`
- Modify: `backend/agent/tools/builtin/shell.py`
- Modify: `backend/app/builder.py`
- Modify: `backend/tests/test_builtin_tools.py`
- Modify: `backend/tests/test_builder_tools.py`

**Interfaces:**
- Consumes: the revised `load_skill` behavior from Task 1.
- Produces: a runtime registry and Agent toolbox containing `load_skill` but never `read_skill_resource`.

- [ ] **Step 1: Write failing absence tests**

Update registry assertions in `backend/tests/test_builtin_tools.py` and `backend/tests/test_builder_tools.py` so a real skill package produces:

```python
assert "load_skill" in reg.names()
assert "read_skill_resource" not in reg.names()
```

In the active-skill include-whitelist test, require only:

```python
assert "load_skill" in names
assert "read_skill_resource" not in names
```

Delete the four direct `make_read_skill_resource_tool` behavior tests and its import from `backend/tests/test_builtin_tools.py`.

Add `read_skill_resource --help` to the tuple of real shell commands in `test_shell_redirects_a_tool_name_typed_as_a_command`:

```python
for cmd in (
    "ls -la",
    "git status",
    "./load_skill",
    "python load_skill.py",
    "read_skill_resource --help",
    "render_notes notes.txt",
    "search_notes ERR_1 notes.txt",
    "list_catalogs",
):
    out = asyncio.run(t.fn(command=cmd))
    assert "one of your own tools" not in out
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/test_builtin_tools.py tests/test_builder_tools.py -q
```

Expected: FAIL because the registry and forced loader list still contain `read_skill_resource` and shell still reserves it.

- [ ] **Step 3: Remove the tool from production surfaces**

Make these minimal changes:

```python
# backend/agent/tools/defaults.py
# Delete the make_read_skill_resource_tool import and registration.
# Rewrite the progressive-disclosure comment to state that load_skill reads SKILL.md;
# sandbox tools access all bundled files.
```

```python
# backend/app/builder.py
_SKILL_LOADER_TOOLS = ("load_skill",)
```

Update builder comments to refer to one loader and sandbox resource access. Remove `read_skill_resource` from `_AGENT_TOOL_NAMES` in `backend/agent/tools/builtin/shell.py`. Delete `backend/agent/tools/builtin/read_skill_resource.py`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command from Step 2.

Expected: all selected tests PASS.

- [ ] **Step 5: Commit runtime tool removal**

```bash
git add backend/agent/tools/defaults.py backend/agent/tools/builtin/shell.py backend/app/builder.py backend/tests/test_builtin_tools.py backend/tests/test_builder_tools.py
git rm backend/agent/tools/builtin/read_skill_resource.py
git commit -m "refactor(skills): remove host-side resource reader"
```

---

### Task 3: Remove Frontend and Active-Documentation References

**Files:**
- Modify: `frontend/src/lib/__tests__/toolCallDisplay.test.ts`
- Modify: `frontend/src/lib/toolCallDisplay.ts`
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/i18n/zh.ts`
- Modify: `docs/design/skill_install_mount_dependencies.md`

**Interfaces:**
- Consumes: the 15-tool backend surface after Task 2.
- Produces: frontend built-in coverage and architecture docs that expose no removed resource tool.

- [ ] **Step 1: Write the failing frontend fallback test**

Remove the existing `read_skill_resource` built-in table row from `toolCallDisplay.test.ts` and add:

```ts
  it("treats the removed skill resource reader as an unknown tool", () => {
    expect(
      getToolCallDisplay(
        "read_skill_resource",
        '{"skill_id":"skill.pdf","path":"references/a.md"}',
      ),
    ).toEqual({});
  });
```

- [ ] **Step 2: Run the frontend test and verify RED**

Run:

```bash
cd frontend && npm test -- src/lib/__tests__/toolCallDisplay.test.ts
```

Expected: FAIL because `read_skill_resource` is still configured as a built-in display policy.

- [ ] **Step 3: Remove frontend configuration and update active docs**

Delete:

```ts
// frontend/src/lib/toolCallDisplay.ts
read_skill_resource: {
  labelKey: "tool.name.readSkillResource",
  summary: (args) => joinValues(stringValue(args.skill_id), stringValue(args.path)),
},
```

Delete `tool.name.readSkillResource` from both i18n dictionaries.

Rewrite the progressive-disclosure section in `docs/design/skill_install_mount_dependencies.md` so:

- L2 returns `SKILL.md` plus the exact mount path, with no manifest;
- L3 is sandbox-only discovery/read/execute through `shell` or `code_interpreter`;
- no-sandbox resource access has no host fallback;
- authorization language applies to `load_skill`, while sandbox mounts remain per-Agent scoped.

- [ ] **Step 4: Run focused frontend test and verify GREEN**

Run the same Vitest command from Step 2.

Expected: all display-policy tests PASS.

- [ ] **Step 5: Commit frontend and documentation cleanup**

```bash
git add frontend/src/lib/toolCallDisplay.ts frontend/src/lib/__tests__/toolCallDisplay.test.ts frontend/src/i18n/en.ts frontend/src/i18n/zh.ts docs/design/skill_install_mount_dependencies.md
git commit -m "docs(skills): document sandbox-only resources"
```

---

### Task 4: Repository-Wide Verification

**Files:**
- Verify all changed files.
- Historical files under `docs/superpowers/` are exempt from the stale-reference scan.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: evidence that the removed tool is absent and all remaining built-ins retain localized frontend display policy.

- [ ] **Step 1: Run backend tests**

```bash
cd backend && .venv/bin/python -m pytest tests/test_tool_extensions.py tests/test_builtin_tools.py tests/test_builder_tools.py -q
```

Expected: all selected backend tests PASS.

- [ ] **Step 2: Run frontend tests and build**

```bash
cd frontend && npm test && npm run build
```

Expected: all frontend tests PASS and production build exits successfully.

- [ ] **Step 3: Scan active code and docs for stale names**

From the repository root:

```bash
if rg -n "read_skill_resource" backend frontend docs/design; then
  echo "stale active reference found" >&2
  exit 1
fi
```

Expected: no matches.

- [ ] **Step 4: Audit backend-to-frontend built-in coverage**

```bash
node <<'JS'
const fs = require('fs');
const path = require('path');
const dir = 'backend/agent/tools/builtin';
const builtin = new Set();
for (const file of fs.readdirSync(dir).filter((name) => name.endsWith('.py'))) {
  const source = fs.readFileSync(path.join(dir, file), 'utf8');
  for (const match of source.matchAll(/name="([a-z_]+)"/g)) builtin.add(match[1]);
}
const policy = fs.readFileSync('frontend/src/lib/toolCallDisplay.ts', 'utf8');
const missing = [...builtin].filter((name) => !new RegExp(`^  ${name}:`, 'm').test(policy)).sort();
if (missing.length) throw new Error(`missing built-in display policies: ${missing.join(', ')}`);
if (builtin.size !== 15) throw new Error(`expected 15 built-ins, found ${builtin.size}`);
console.log('covered 15 built-in tools');
JS
```

Expected: `covered 15 built-in tools`.

- [ ] **Step 5: Verify repository state and push**

```bash
git diff --check
git status --short
git push origin personal/yfei/agent-core
```

Expected: no diff errors, clean worktree before push, and successful remote update.
