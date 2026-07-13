# Knowledge Tool Bundle and Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose knowledge as one complete Agent tool bundle named `knowledge_search`, `knowledge_read`, `knowledge_find`, and `knowledge_list`, with direct removal of all legacy runtime names.

**Architecture:** A small backend bundle module owns canonical names and legacy configuration normalization. Registry construction, Agent tool selection, prompts, and frontend toggles consume that contract so a partial knowledge enablement cannot occur. Existing stored profiles are normalized on load/save; runtime aliases are deliberately not registered.

**Tech Stack:** Python 3.12, FastAPI/Pydantic, pytest, React 19, TypeScript, Vitest.

## Global Constraints

- The only runtime names are `knowledge_search`, `knowledge_read`, `knowledge_find`, and `knowledge_list`.
- Remove `view_file`, `grep_file`, and `list_knowledge_bases` directly; do not register compatibility aliases.
- Enabling knowledge exposes the entire canonical bundle; disabling or excluding knowledge hides the entire bundle.
- Preserve existing retrieval, reranking, permission, scoring, and empty-result behavior.
- Internal `document_id` and `chunk_id` values remain tool inputs and must not appear in final user answers.
- Do not commit the unrelated generated file `frontend/next-env.d.ts`.

---

### Task 1: Canonical Bundle Contract and Stored-Config Normalization

**Files:**
- Create: `backend/agent/tools/knowledge_bundle.py`
- Modify: `backend/app/agent_config.py`
- Modify: `backend/app/builder.py`
- Test: `backend/tests/test_routes_config.py`
- Test: `backend/tests/test_builder_tools.py`

**Interfaces:**
- Produces: `KNOWLEDGE_TOOL_NAMES: tuple[str, ...]`, `LEGACY_KNOWLEDGE_TOOL_MAP: dict[str, str]`, and `normalize_knowledge_tool_lists(include: list[str], exclude: list[str]) -> tuple[list[str], list[str]]`.
- Consumes: `AgentToolsConfig.include` and `AgentToolsConfig.exclude` from persisted Agent profiles.

- [ ] **Step 1: Write failing normalization tests**

Add focused tests proving that a legacy or partial include becomes the complete canonical bundle, any canonical/legacy exclusion disables the entire bundle, and repeated normalization is idempotent:

```python
def test_merge_normalizes_legacy_knowledge_tools_to_complete_bundle():
    doc = _merge_default(
        {
            "agents": [
                {
                    "id": "main",
                    "tools": {
                        "include": ["current_datetime", "view_file"],
                        "exclude": [],
                    },
                }
            ],
        }
    )
    tools = doc.agents[0].tools
    assert set(KNOWLEDGE_TOOL_NAMES) <= set(tools.include)
    assert "view_file" not in tools.include


def test_knowledge_exclusion_disables_entire_bundle_idempotently():
    include, exclude = normalize_knowledge_tool_lists(
        ["knowledge_search", "shell"], ["grep_file"]
    )
    assert include == ["shell"]
    assert set(KNOWLEDGE_TOOL_NAMES) <= set(exclude)
    assert normalize_knowledge_tool_lists(include, exclude) == (
        include,
        exclude,
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/test_routes_config.py tests/test_builder_tools.py -q
```

Expected: FAIL because the canonical bundle module and normalization behavior do not exist.

- [ ] **Step 3: Implement the canonical bundle helper**

Create `backend/agent/tools/knowledge_bundle.py` with deterministic order and de-duplication:

```python
KNOWLEDGE_TOOL_NAMES = (
    "knowledge_search",
    "knowledge_read",
    "knowledge_find",
    "knowledge_list",
)

LEGACY_KNOWLEDGE_TOOL_MAP = {
    "view_file": "knowledge_read",
    "grep_file": "knowledge_find",
    "list_knowledge_bases": "knowledge_list",
}


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def normalize_knowledge_tool_lists(
    include: list[str], exclude: list[str]
) -> tuple[list[str], list[str]]:
    knowledge_names = set(KNOWLEDGE_TOOL_NAMES) | set(
        LEGACY_KNOWLEDGE_TOOL_MAP
    )
    mapped_include = [
        LEGACY_KNOWLEDGE_TOOL_MAP.get(name, name) for name in include
    ]
    mapped_exclude = [
        LEGACY_KNOWLEDGE_TOOL_MAP.get(name, name) for name in exclude
    ]
    disabled = any(name in knowledge_names for name in exclude)
    nonknowledge_include = [
        name for name in mapped_include if name not in KNOWLEDGE_TOOL_NAMES
    ]
    nonknowledge_exclude = [
        name for name in mapped_exclude if name not in KNOWLEDGE_TOOL_NAMES
    ]
    if disabled:
        return _unique(nonknowledge_include), _unique(
            [*nonknowledge_exclude, *KNOWLEDGE_TOOL_NAMES]
        )
    enabled = any(name in knowledge_names for name in include)
    if enabled:
        return _unique(
            [*nonknowledge_include, *KNOWLEDGE_TOOL_NAMES]
        ), _unique(nonknowledge_exclude)
    return _unique(nonknowledge_include), _unique(nonknowledge_exclude)
```

- [ ] **Step 4: Normalize defaults, loaded documents, and persisted documents**

Update the default `main` Agent include list to contain all four canonical names. Add one helper in `backend/app/agent_config.py` that normalizes every Agent tools dictionary, call it after raw/default merging before `AgentConfigDocument(**merged)`, and apply the same helper to the dictionary returned by `authored_config_dict()` so SQL/YAML saves persist the canonical form.

- [ ] **Step 5: Make backend tool selection enforce bundle semantics**

In `_select_tool_names()` normalize the profile include/exclude lists before filtering registry names. This is defense in depth for runtime documents assembled outside the normal SQL/YAML loaders:

```python
include, exclude_list = normalize_knowledge_tool_lists(include, list(exclude))
exclude = set(exclude_list)
```

Add a builder test whose profile includes only `knowledge_search` but whose registry exposes all four canonical tools; assert the effective toolbox contains all four.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
cd backend
uv run pytest tests/test_routes_config.py tests/test_builder_tools.py -q
```

Expected: PASS.

Commit:

```bash
git add backend/agent/tools/knowledge_bundle.py backend/app/agent_config.py backend/app/builder.py backend/tests/test_routes_config.py backend/tests/test_builder_tools.py
git commit -m "feat(knowledge): enforce complete agent tool bundle"
```

### Task 2: Rename Runtime Tools and Remove Legacy Names

**Files:**
- Create: `backend/agent/tools/builtin/knowledge_read.py`
- Create: `backend/agent/tools/builtin/knowledge_find.py`
- Create: `backend/agent/tools/builtin/knowledge_list.py`
- Delete: `backend/agent/tools/builtin/view_file.py`
- Delete: `backend/agent/tools/builtin/grep_file.py`
- Delete: `backend/agent/tools/builtin/list_kbs.py`
- Modify: `backend/agent/tools/defaults.py`
- Modify: `backend/agent/tools/builtin/knowledge.py`
- Modify: `backend/agent/tools/builtin/shell.py`
- Modify: `backend/app/agent_config.py`
- Test: `backend/tests/test_knowledge_view_grep_tools.py` (rename to `backend/tests/test_knowledge_inspection_tools.py`)
- Test: `backend/tests/test_knowledge_tool.py`
- Test: `backend/tests/test_routes_config.py`

**Interfaces:**
- Consumes: canonical names and normalization from Task 1.
- Produces: `make_knowledge_read_tool()`, `make_knowledge_find_tool()`, and `make_knowledge_list_tool()` returning `Tool` objects with canonical `.name` values.

- [ ] **Step 1: Rename the inspection tests and make them fail on canonical names**

Rename the test module and imports, then change assertions to require:

```python
read_tool = make_knowledge_read_tool(service)
find_tool = make_knowledge_find_tool(service)
list_tool = make_knowledge_list_tool(service)
assert read_tool.name == "knowledge_read"
assert find_tool.name == "knowledge_find"
assert list_tool.name == "knowledge_list"
```

Update friendly-error assertions so returned messages use the canonical tool name. Add a registry assertion that none of the legacy names is registered.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/test_knowledge_inspection_tools.py tests/test_knowledge_tool.py tests/test_routes_config.py -q
```

Expected: FAIL because canonical modules/factories do not exist and registry names are still legacy.

- [ ] **Step 3: Create canonical tool modules and delete legacy modules**

Move the existing implementations without changing their service calls, argument schemas, pagination, or permission handling. Apply this exact symbol/name mapping:

| Legacy module/factory/tool name | Canonical module/factory/tool name |
| --- | --- |
| `builtin/view_file.py` / `make_view_file_tool` / `view_file` | `builtin/knowledge_read.py` / `make_knowledge_read_tool` / `knowledge_read` |
| `builtin/grep_file.py` / `make_grep_file_tool` / `grep_file` | `builtin/knowledge_find.py` / `make_knowledge_find_tool` / `knowledge_find` |
| `builtin/list_kbs.py` / `make_list_kbs_tool` / `list_knowledge_bases` | `builtin/knowledge_list.py` / `make_knowledge_list_tool` / `knowledge_list` |

Every user-facing error, pagination hint, tool description, and cross-tool recommendation must use `knowledge_read`, `knowledge_find`, or `knowledge_list`. Delete the old modules rather than leaving forwarding imports.

- [ ] **Step 4: Register the complete bundle behind the knowledge capability**

Update `build_default_registry()` to import the canonical factories, register all four tools when a live KnowledgeService exists and knowledge is not explicitly disabled, and register none when the `knowledge` capability is disabled. Preserve registration when `agent_config is None` for isolated tests and hosts without control-plane configuration.

- [ ] **Step 5: Update all backend mappings and collision guards**

Replace executable references in:

- `backend/agent/tools/builtin/knowledge.py` result instructions;
- `backend/agent/tools/builtin/shell.py` direct-tool collision guard;
- `backend/app/agent_config.py` skill permission-to-capability mapping;
- route/config tests and registry tests.

The search result footer must say:

```text
To inspect a hit in context, use knowledge_read(chunk_id=…, mode="locate").
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
cd backend
uv run pytest tests/test_knowledge_inspection_tools.py tests/test_knowledge_tool.py tests/test_routes_config.py tests/test_builder_tools.py -q
```

Expected: PASS.

Commit the new files, deletions, and tests:

```bash
git add backend/agent/tools backend/app/agent_config.py backend/tests/test_knowledge_inspection_tools.py backend/tests/test_knowledge_tool.py backend/tests/test_routes_config.py backend/tests/test_builder_tools.py
git commit -m "refactor(knowledge): namespace inspection tools"
```

### Task 3: Update Prompting, Subagents, and Run Diagnostics

**Files:**
- Modify: `backend/agent/soul.py`
- Modify: `backend/app/subagent.py`
- Modify: `backend/app/builder.py`
- Test: `backend/tests/test_soul.py`
- Test: `backend/tests/test_builder.py`
- Test: `backend/tests/test_subagent.py`

**Interfaces:**
- Consumes: canonical bundle constants from Task 1 and registered tools from Task 2.
- Produces: system prompts and subagent profiles containing only canonical names; debug diagnostics containing Agent/tool/scope metadata but no document content.

- [ ] **Step 1: Write failing prompt and metadata-log tests**

Require the stable prompt with the complete bundle to contain `knowledge_read`, `knowledge_find`, and `knowledge_list`, and to contain none of the three legacy names. Capture Loguru output from context construction and assert it records `agent_id`, canonical effective knowledge tools, configured KB IDs, and rerank enabled state without query or passage content.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/test_soul.py tests/test_builder.py tests/test_subagent.py -q
```

Expected: FAIL on legacy prompt names and missing context diagnostics.

- [ ] **Step 3: Update stable and auxiliary knowledge guidance**

Replace the auxiliary flow with canonical wording:

```text
Use knowledge_list only when discovery or narrowing is useful. When a retrieved
passage is incomplete, ambiguous, or lacks context, call knowledge_read with the
internal document_id or chunk_id. Use knowledge_find for exact identifiers,
error codes, API names, or literal phrases.
```

Keep `knowledge_search` as the capability anchor and inject auxiliary guidance only when canonical inspection tools are present.

- [ ] **Step 4: Update subagent knowledge profile and guidance**

Change the built-in explore worker's allowed-tool list to all four canonical names and update its instructions. Do not retain legacy names as accepted tools.

- [ ] **Step 5: Add safe run-construction diagnostics**

After effective `tool_names` and Agent knowledge configuration are resolved, log a structured summary:

```python
logger.debug(
    "agent knowledge tools resolved: agent_id={} tools={} kb_ids={} rerank_enabled={}",
    _agent_id(agent_config, agent_profile),
    [name for name in tool_names if name in KNOWLEDGE_TOOL_NAMES],
    list(agent_kbs or []),
    bool(agent_rerank and agent_rerank.enabled),
)
```

Do not log user queries, snippets, document IDs, or document text in this diagnostic.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
cd backend
uv run pytest tests/test_soul.py tests/test_builder.py tests/test_subagent.py -q
```

Expected: PASS.

Commit:

```bash
git add backend/agent/soul.py backend/app/subagent.py backend/app/builder.py backend/tests/test_soul.py backend/tests/test_builder.py backend/tests/test_subagent.py
git commit -m "fix(knowledge): guide canonical inspection flow"
```

### Task 4: Make the Frontend Knowledge Toggle Manage the Complete Bundle

**Files:**
- Modify: `frontend/src/components/SettingsView.tsx`
- Modify: `frontend/src/components/__tests__/SettingsView.test.tsx`
- Modify: `frontend/src/api/agentConfig.ts` only if a shared exported constant is needed for tests; otherwise leave unchanged.

**Interfaces:**
- Consumes: canonical tool names defined by the backend contract.
- Produces: `TOOL_BUNDLES.knowledge_search` containing all four canonical names and UI enabled-state logic that recognizes any canonical member.

- [ ] **Step 1: Write failing toggle tests**

Add assertions that enabling the knowledge card writes exactly the complete bundle while preserving unrelated tools, and disabling it removes all canonical and legacy knowledge names:

```typescript
expect(agent.tools.include).toEqual(expect.arrayContaining([
  "knowledge_search",
  "knowledge_read",
  "knowledge_find",
  "knowledge_list",
]));
expect(agent.tools.include).not.toEqual(expect.arrayContaining([
  "view_file",
  "grep_file",
  "list_knowledge_bases",
]));
```

- [ ] **Step 2: Run the frontend test and verify RED**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/SettingsView.test.tsx
```

Expected: FAIL because the knowledge bundle currently contains only `knowledge_search`.

- [ ] **Step 3: Implement complete frontend bundle behavior**

Set:

```typescript
const KNOWLEDGE_TOOL_BUNDLE = [
  "knowledge_search",
  "knowledge_read",
  "knowledge_find",
  "knowledge_list",
];

const TOOL_BUNDLES: Record<string, string[]> = {
  knowledge_search: KNOWLEDGE_TOOL_BUNDLE,
  // existing bundles unchanged
};
```

Include legacy names only in removal/detection aliases so saving a migrated Agent strips them; never write them back into include/exclude.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/SettingsView.test.tsx
npm run build
```

Expected: tests and build PASS.

Commit:

```bash
git add frontend/src/components/SettingsView.tsx frontend/src/components/__tests__/SettingsView.test.tsx
git commit -m "fix(settings): toggle complete knowledge tool bundle"
```

### Task 5: Repository-Wide Verification and Stale-Name Gate

**Files:**
- Modify only files discovered by the stale-name scan when they contain executable/runtime references.
- Test: all backend and frontend suites.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified canonical-only runtime surface.

- [ ] **Step 1: Scan for stale executable names**

Run:

```bash
rg -n "view_file|grep_file|list_knowledge_bases" backend frontend/src \
  --glob '!**/docs/**' --glob '!**/*.md'
```

Expected: no executable references. Historical specs/plans may still contain old names.

- [ ] **Step 2: Run backend full verification**

Run:

```bash
cd backend
uv run pytest -q
```

Expected: all tests PASS; existing warnings may remain.

- [ ] **Step 3: Run frontend full verification**

Run:

```bash
cd frontend
npm test -- --run
npm run build
```

Expected: all tests and production build PASS; the existing large-chunk warning is non-blocking.

- [ ] **Step 4: Run changed-file quality gates**

Run Ruff and Mypy through the repository hooks on the changed Python files, then verify whitespace:

```bash
pre-commit run ruff --files \
  backend/agent/tools/knowledge_bundle.py \
  backend/agent/tools/builtin/knowledge.py \
  backend/agent/tools/builtin/knowledge_read.py \
  backend/agent/tools/builtin/knowledge_find.py \
  backend/agent/tools/builtin/knowledge_list.py \
  backend/agent/tools/builtin/shell.py \
  backend/agent/tools/defaults.py \
  backend/agent/soul.py backend/app/agent_config.py backend/app/builder.py \
  backend/app/subagent.py backend/tests/test_builder.py \
  backend/tests/test_builder_tools.py backend/tests/test_knowledge_inspection_tools.py \
  backend/tests/test_knowledge_tool.py backend/tests/test_routes_config.py \
  backend/tests/test_soul.py backend/tests/test_subagent.py
pre-commit run mypy --files \
  backend/agent/tools/knowledge_bundle.py \
  backend/agent/tools/builtin/knowledge.py \
  backend/agent/tools/builtin/knowledge_read.py \
  backend/agent/tools/builtin/knowledge_find.py \
  backend/agent/tools/builtin/knowledge_list.py \
  backend/agent/tools/defaults.py backend/agent/soul.py \
  backend/app/agent_config.py backend/app/builder.py backend/app/subagent.py
git diff --check
```

Expected: all checks PASS.

- [ ] **Step 5: Review final config behavior and working-tree boundaries**

Verify the persisted-config tests demonstrate migration and idempotency, `git diff --cached` is empty, and `frontend/next-env.d.ts` remains untracked and uncommitted.

- [ ] **Step 6: Confirm delivery state**

Run `git status --short` and `git log --oneline -5`. Expected: the implementation files are committed in Tasks 1–4, no staged changes remain, and only the pre-existing untracked `frontend/next-env.d.ts` is outside the commits.
