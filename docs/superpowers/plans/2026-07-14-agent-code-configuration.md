# Per-Agent Code Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move code-repository prompt control from sandbox-provider settings into a default-off nested Agent configuration with an `AGENT_CODE_PATH` override and `/opt/code` fallback.

**Architecture:** `AgentProfile.code` is the only source of code-layer enablement and manifest text. The builder maps it into the existing prompt renderer for main Agents and subagents; the sandbox provider remains responsible only for execution and mounts. The UI edits the nested object through the existing SQL-backed whole-document configuration API.

**Tech Stack:** Python 3.12, Pydantic, FastAPI, SQLAlchemy, pytest, React, TypeScript, Vitest.

## Global Constraints

- `code.enabled` defaults to `false` and is enabled manually per Agent.
- The schema is `code: { enabled: boolean, manifest: string }`.
- Do not retain `code_manifest` or provider-level `code_layer_enabled` compatibility.
- `AGENT_CODE_PATH` overrides the repository root; `/opt/code` is the default.
- Inject code guidance only when enabled and `shell` or `code_interpreter` is effective.
- Do not add an Alembic migration; SQL already stores the full Agent document.

---

### Task 1: Nested Agent code model and persistence

**Files:**
- Modify: `backend/app/agent_config.py`
- Modify: `backend/tests/test_routes_config.py`
- Modify: `backend/tests/test_agent_config_store_sql.py`
- Modify: `frontend/src/api/agentConfig.ts`

**Interfaces:**
- Produces: `AgentCodeConfig(enabled: bool = False, manifest: str = "")`.
- Produces: TypeScript `AgentCodeConfig` and `AgentProfile.code`.
- Removes: `AgentProfile.code_manifest`.

- [ ] **Step 1: Write failing model and SQL persistence tests**

```python
agent = AgentProfile(id="main", name="Main")
assert agent.code.enabled is False
assert agent.code.manifest == ""

doc.agents[0].code.enabled = True
doc.agents[0].code.manifest = "- repo-a"
await store.save(doc, expected_revision=stored.revision)
reloaded = await store.load()
assert reloaded.doc.agents[0].code.model_dump() == {
    "enabled": True,
    "manifest": "- repo-a",
}
```

- [ ] **Step 2: Run tests and verify RED**

```bash
backend/.venv/bin/python -m pytest -q \
  backend/tests/test_routes_config.py::test_agent_code_config_survives_save_load \
  backend/tests/test_agent_config_store_sql.py::test_sql_agent_code_config_roundtrip
```

Expected: failures because `AgentProfile.code` does not exist.

- [ ] **Step 3: Implement the nested models**

```python
class AgentCodeConfig(BaseModel):
    enabled: bool = False
    manifest: str = ""


class AgentProfile(BaseModel):
    code: AgentCodeConfig = Field(default_factory=AgentCodeConfig)
```

Remove `code_manifest` and the provider default setting. Mirror the type in
TypeScript and initialize new Agents with `code: { enabled: false, manifest: "" }`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run Step 2. Expected: both tests pass.

### Task 2: Agent-owned runtime prompt injection

**Files:**
- Modify: `backend/app/builder.py`
- Modify: `backend/agent/soul.py`
- Modify: `backend/agent/tools/builtin/shell.py`
- Modify: `backend/tests/test_builder.py`
- Modify: `backend/tests/test_soul.py`
- Modify: `backend/tests/test_builtin_tools.py`

**Interfaces:**
- Consumes: `AgentProfile.code.enabled` and `AgentProfile.code.manifest`.
- Produces: guidance using `CODE_PATH="${AGENT_CODE_PATH:-/opt/code}"`.
- Removes: reads of `registry.sandbox_provider.code_layer_enabled`.

- [ ] **Step 1: Write failing prompt-isolation tests**

Cover two Agents sharing one registry:

```python
disabled = AgentProfile(id="disabled", name="Disabled")
enabled = AgentProfile(
    id="enabled",
    name="Enabled",
    code={"enabled": True, "manifest": "- repo-a"},
)
assert "/opt/code" not in disabled_context.system_prompt
assert "${AGENT_CODE_PATH:-/opt/code}" in enabled_context.system_prompt
assert "- repo-a" in enabled_context.system_prompt
```

Also assert an enabled Agent without sandbox tools gets no code guidance and the
shared shell description never mentions `/opt/code`.

- [ ] **Step 2: Run tests and verify RED**

```bash
backend/.venv/bin/python -m pytest -q \
  backend/tests/test_builder.py backend/tests/test_soul.py \
  backend/tests/test_builtin_tools.py
```

Expected: Agent-owned enablement and fallback-path assertions fail.

- [ ] **Step 3: Implement builder and prompt behavior**

```python
code_config = getattr(agent_profile, "code", None)
code_enabled = bool(getattr(code_config, "enabled", False))
code_manifest = getattr(code_config, "manifest", "") or ""
```

Pass these values to main and subagent prompt rendering. In code guidance, tell the
model to assign `CODE_PATH="${AGENT_CODE_PATH:-/opt/code}"` and quote
`"$CODE_PATH"`. Remove the `/opt/code` suffix from the shared shell description.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run Step 2. Expected: all selected tests pass.

### Task 3: Remove provider ownership and environment injection

**Files:**
- Modify: `backend/agent/tools/sandbox_providers.py`
- Modify: `backend/app/agent_config.py`
- Modify: `backend/tests/test_builder_aliyun_injection.py`
- Modify locally: `backend/data/config.yaml` (ignored runtime configuration)

**Interfaces:**
- Produces: `_build_env_contract(...)` without `AGENT_CODE_PATH`.
- Removes: `AgentRunRestSandboxProvider.code_layer_enabled`.

- [ ] **Step 1: Change environment tests to the desired behavior**

```python
envs = _build_env_contract(_fake_provider(), scope, "s:u1")
assert "AGENT_CODE_PATH" not in envs
```

Delete fake-provider code flags and assert the provider default document has no
`code_layer_enabled` key.

- [ ] **Step 2: Run test and verify RED**

```bash
backend/.venv/bin/python -m pytest -q backend/tests/test_builder_aliyun_injection.py
```

Expected: old provider-owned expectations fail.

- [ ] **Step 3: Remove provider setting and injection**

Delete `self.code_layer_enabled`, the provider default key, and:

```python
if provider.code_layer_enabled:
    envs["AGENT_CODE_PATH"] = "/opt/code"
```

Remove `code_layer_enabled: true` from ignored `backend/data/config.yaml`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run Step 2. Expected: all tests pass.

### Task 4: Agent-gated manifest generation API

**Files:**
- Modify: `backend/app/routes/agents.py`
- Modify: `backend/tests/test_agents.py`

**Interfaces:**
- Consumes: requested `AgentProfile.code.enabled`.
- Produces: 404 unknown Agent, 409 disabled code, 503 missing runtime, success text.

- [ ] **Step 1: Write failing endpoint tests**

```python
assert (
    client.post("/v1/agents/missing/code-manifest/generate").status_code == 404
)
assert (
    client.post("/v1/agents/disabled/code-manifest/generate").status_code
    == 409
)
assert (
    client.post("/v1/agents/enabled/code-manifest/generate").status_code == 503
)
```

Capture the generation request and assert it contains
`${AGENT_CODE_PATH:-/opt/code}`.

- [ ] **Step 2: Run tests and verify RED**

```bash
backend/.venv/bin/python -m pytest -q backend/tests/test_agents.py
```

Expected: route still checks the provider flag or returns statuses in the old order.

- [ ] **Step 3: Implement Agent resolution and gating**

```python
profile = next((agent for agent in agents if agent.id == agent_id), None)
if profile is None:
    return _err(404, f"unknown agent: {agent_id}")
if not profile.code.enabled:
    return _err(409, "code access is not enabled for this agent")
if getattr(getattr(state, "registry", None), "sandbox_provider", None) is None:
    return _err(503, "sandbox provider is unavailable")
```

Update the instruction to resolve `CODE_PATH` from the environment with the default.

- [ ] **Step 4: Run endpoint tests and verify GREEN**

Run Step 2. Expected: all tests pass.

### Task 5: Agent-level frontend switch and nested manifest

**Files:**
- Modify: `frontend/src/components/SettingsView.tsx`
- Modify: `frontend/src/components/__tests__/SettingsView.test.tsx`
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/i18n/zh.ts`

**Interfaces:**
- Consumes/produces: `AgentProfile.code.enabled` and `.manifest`.
- Keeps: `generateCodeManifest(agent.id)`.

- [ ] **Step 1: Write failing UI tests**

Assert new Agents default off, a switch named “代码仓库访问” persists enablement,
the manifest editor is conditional, and generation saves:

```ts
expect(saved.agents[0].code.enabled).toBe(true);
expect(saved.agents[0].code.manifest).toBe("- repo-a — the API server");
```

Also assert the switch does not mutate `agent.tools`.

- [ ] **Step 2: Run test and verify RED**

```bash
npm test --prefix frontend -- --run src/components/__tests__/SettingsView.test.tsx
```

Expected: nested fields and the manual switch are absent.

- [ ] **Step 3: Implement switch and nested updates**

```ts
applyAgentPatch(doc, agent.id, {
  code: { ...agent.code, enabled: !agent.code.enabled },
})
```

Show `CodeManifestSection` only when `agent.code.enabled`; read and write
`agent.code.manifest` without changing the tool list.

- [ ] **Step 4: Run UI test and verify GREEN**

Run Step 2. Expected: all `SettingsView` tests pass.

### Task 6: Cleanup and full validation

**Files:**
- Modify as needed: `backend/agent/souls/pai-rec.soul.md`
- Modify as needed: `backend/app/subagent.py`
- Modify: comments/docs containing obsolete runtime semantics.

**Interfaces:**
- Removes: executable `code_layer_enabled` and `code_manifest` references.
- Keeps: `/opt/code` only as the documented default root.

- [ ] **Step 1: Scan stale references**

```bash
rg -n "code_layer_enabled|code_manifest" backend frontend \
  --glob '!frontend/node_modules/**' --glob '!*.pyc'
```

Expected: no executable legacy references remain.

- [ ] **Step 2: Run full verification**

```bash
/opt/homebrew/bin/ruff check backend
backend/.venv/bin/python -m pytest -q backend/tests
npm test --prefix frontend
npm run build --prefix frontend
git diff --check
```

Expected: lint and tests pass, production build succeeds, and only pre-existing
warning classes remain.

- [ ] **Step 3: Review final diff**

Confirm ignored `backend/data/config.yaml` has no `code_layer_enabled`, the user-owned
`frontend/next-env.d.ts` remains untouched, and unrelated changes are preserved.
