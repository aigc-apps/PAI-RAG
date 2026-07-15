# Default OpenAI-Compatible Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the explicit default OpenAI-compatible provider prefer `OPENAI_BASE_URL` and `OPENAI_API_KEY`, with secure UI-editable manual fallbacks.

**Architecture:** Keep authored provider values separate from runtime-resolved values. `ProviderConfig` owns environment-first resolution, configuration routes mask and preserve manual secrets, and the Models settings UI renders the environment-managed provider through a dedicated branch while leaving custom providers unchanged.

**Tech Stack:** Python 3, Pydantic, FastAPI, pytest, React, TypeScript, Vitest, Testing Library, i18n dictionaries.

## Global Constraints

- Only a provider explicitly marked `type: openai_compatible` and `use_default_env: true` reads `OPENAI_BASE_URL` and `OPENAI_API_KEY`.
- Environment values take precedence independently; blank or whitespace-only values count as absent.
- Actual environment values never enter persisted configuration or API responses.
- Manual API keys are masked as `********` and preserved when the mask is submitted unchanged.
- Existing custom-provider `base_url`, `api_key_env`, and `api_key` behavior remains compatible.
- All new user-facing strings are available in Chinese and English.
- Do not introduce a secret manager or claim at-rest encryption.

---

### Task 1: Environment-first provider resolution

**Files:**
- Modify: `backend/app/providers.py`
- Modify: `backend/app/agent_config.py`
- Test: `backend/tests/test_providers.py`
- Create: `backend/tests/test_agent_config_models.py`

**Interfaces:**
- Produces: `ProviderConfig.type: Literal["openai_compatible"]`, `ProviderConfig.use_default_env: bool`, `ProviderConfig.resolve_base_url() -> str`, and environment-first `ProviderConfig.resolve_key() -> str`.
- Produces: `ModelConfig.use_default_env: bool`, `ModelConfig.resolve_base_url() -> str`, and matching environment-first `ModelConfig.resolve_key() -> str` for all client builders.
- Consumes: `OPENAI_BASE_URL` and `OPENAI_API_KEY` from `os.environ` only at runtime.

- [ ] **Step 1: Add failing provider-resolution tests**

Add focused cases equivalent to:

```python
def test_default_provider_prefers_openai_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", " https://env.example/v1 ")
    monkeypatch.setenv("OPENAI_API_KEY", " env-key ")
    provider = ProviderConfig(
        name="openai",
        type="openai_compatible",
        use_default_env=True,
        base_url="https://manual.example/v1",
        api_key="manual-key",
        models=[ModelSpec(id="chat")],
    )
    config = ModelConfig.from_provider(provider, provider.models[0])
    assert config.resolve_base_url() == "https://env.example/v1"
    assert config.resolve_key() == "env-key"


def test_default_provider_falls_back_independently(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    provider = ProviderConfig(
        name="openai",
        type="openai_compatible",
        use_default_env=True,
        base_url="https://manual.example/v1",
        api_key="manual-key",
        models=[ModelSpec(id="chat")],
    )
    config = ModelConfig.from_provider(provider, provider.models[0])
    assert config.resolve_base_url() == "https://manual.example/v1"
    assert config.resolve_key() == "env-key"
```

Also cover whitespace-only environment values, custom providers ignoring the two variables, missing URL, and missing key.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `cd backend && pytest -q tests/test_providers.py -k 'default_provider or openai_environment'`

Expected: failures because provider type, default-env marker, and base-URL resolver do not exist.

- [ ] **Step 3: Implement the authored and resolved provider fields**

Implement the schema and resolver in `backend/app/providers.py` with this contract:

```python
ProviderType = Literal["openai_compatible"]


def _non_blank_env(name: str) -> str:
    return os.environ.get(name, "").strip()


class ProviderConfig(BaseModel):
    name: str
    type: ProviderType = "openai_compatible"
    use_default_env: bool = False
    base_url: str = ""
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    models: List[ModelSpec]

    def resolve_base_url(self) -> str:
        if self.use_default_env:
            return _non_blank_env("OPENAI_BASE_URL") or self.base_url.strip()
        return self.base_url.strip()

    def resolve_key(self) -> str:
        if self.use_default_env:
            return (
                _non_blank_env("OPENAI_API_KEY")
                or (self.api_key or "").strip()
            )
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""
```

Copy `type` and `use_default_env` into `ModelConfig`. Preserve DashScope-native per-model URL precedence; otherwise resolve the provider URL at client-use time. Raise the two actionable default-provider messages before constructing a client when URL or key is absent. Retain keyless custom-provider behavior when no key source is configured.

- [ ] **Step 4: Update the built-in default document and compatibility merge**

Change `DEFAULT_DOCUMENT.models.providers[0]` in `backend/app/agent_config.py` to:

```python
{
    "name": "openai",
    "type": "openai_compatible",
    "use_default_env": True,
    "base_url": "",
    "api_key": "",
    "models": [...],
}
```

During `_merge_default`, upgrade only the provider occupying the built-in
default-provider slot when it is the merged shipped provider; do not rewrite
arbitrary custom providers named `openai`. Preserve authored connection fields
and models.

- [ ] **Step 5: Run provider and agent-config tests**

Run: `cd backend && pytest -q tests/test_providers.py tests/test_agent_config_models.py`

Expected: all tests pass.

- [ ] **Step 6: Commit the provider resolver**

```bash
git add backend/app/providers.py backend/app/agent_config.py backend/tests/test_providers.py backend/tests/test_agent_config_models.py
git commit -m "feat(models): resolve default provider from environment"
```

### Task 2: Secret masking, preservation, and runtime status

**Files:**
- Modify: `backend/app/agent_config.py`
- Modify: `backend/app/routes/config.py`
- Test: `backend/tests/test_routes_config.py`

**Interfaces:**
- Consumes: model providers in `AgentConfigDocument.models["providers"]`.
- Produces: non-authored booleans `env_base_url_configured`, `env_api_key_configured`, `manual_base_url_configured`, and `manual_api_key_configured` on runtime API provider objects.
- Produces: `_preserve_masked_model_secrets(doc, current) -> None` behavior inside the configuration update path.

- [ ] **Step 1: Add failing route tests for masking and preservation**

Add tests that assert:

```python
assert response_provider["api_key"] == "********"
assert "actual-env-url" not in response.text
assert "actual-env-key" not in response.text
assert response_provider["env_base_url_configured"] is True
assert response_provider["env_api_key_configured"] is True
```

Then submit the masked document through `PUT /v1/config` and assert the authored
provider still contains the original manual key. Cover replacing the mask with a
new key and independent configured booleans.

- [ ] **Step 2: Run route tests and verify failure**

Run: `cd backend && pytest -q tests/test_routes_config.py -k 'model_provider and (mask or environment or secret)'`

Expected: failures because model-catalog secrets are not masked, preserved, or decorated.

- [ ] **Step 3: Extend authored/runtime filtering**

In `authored_config_dict`, remove the four runtime-only status flags from every
`models.providers` item. In `mask_secrets`, deep-copy and mask a non-empty model
provider `api_key`. In `apply_runtime_status`, decorate only the explicitly
environment-managed provider using environment-presence booleans and authored
fallback-presence booleans; never attach resolved values.

- [ ] **Step 4: Preserve masked model secrets on update**

Extend `_preserve_masked_secrets` in `backend/app/routes/config.py` so providers
are matched by `name` inside `doc.models["providers"]`. When incoming
`api_key == "********"`, restore the current authored key or remove the field if
none exists. Run preservation before `ModelCatalog(**doc.models)` validation.

- [ ] **Step 5: Run config-route and provider tests**

Run: `cd backend && pytest -q tests/test_routes_config.py tests/test_providers.py`

Expected: all tests pass and no environment values appear in serialized output.

- [ ] **Step 6: Commit API secret handling**

```bash
git add backend/app/agent_config.py backend/app/routes/config.py backend/tests/test_routes_config.py
git commit -m "feat(config): protect default provider fallbacks"
```

### Task 3: Environment-managed default Provider UI

**Files:**
- Modify: `frontend/src/api/agentConfig.ts`
- Modify: `frontend/src/components/ConnectionsPanel.tsx`
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/i18n/zh.ts`
- Test: `frontend/src/components/__tests__/ConnectionsPanel.test.tsx`

**Interfaces:**
- Consumes: the provider fields and runtime booleans produced by Task 2.
- Produces: manual `base_url` and `api_key` fallback edits for the environment-managed provider.
- Preserves: the current custom-provider dialog using `base_url` and `api_key_env`.

- [ ] **Step 1: Add failing component tests**

Create a fixture with:

```ts
{
  name: "openai",
  type: "openai_compatible",
  use_default_env: true,
  base_url: "",
  api_key: "",
  env_base_url_configured: true,
  env_api_key_configured: false,
  manual_base_url_configured: false,
  manual_api_key_configured: false,
  models: [{ id: "gpt-4o-mini" }],
}
```

Assert the row shows the localized environment-default badge and source states;
the edit dialog shows fixed `OPENAI_BASE_URL` and `OPENAI_API_KEY` labels, a
manual URL fallback, and a password API-key fallback. Enter fallbacks, save, and
assert `api_key_env` is not authored. Add a separate assertion that a custom
provider still displays the API-key environment-name input.

- [ ] **Step 2: Run the component tests and verify failure**

Run: `cd frontend && npm test -- --run src/components/__tests__/ConnectionsPanel.test.tsx`

Expected: failures because environment-managed fields and copy are absent.

- [ ] **Step 3: Extend frontend provider types**

Add optional fields to `ModelProviderDoc`:

```ts
type?: "openai_compatible";
use_default_env?: boolean;
api_key?: string;
env_base_url_configured?: boolean;
env_api_key_configured?: boolean;
manual_base_url_configured?: boolean;
manual_api_key_configured?: boolean;
```

- [ ] **Step 4: Implement the dedicated default-provider edit branch**

Track `apiKey` state. For `p.use_default_env === true`, show fixed environment
source statuses and manual fallback fields, with the key input using
`type="password"` and a configured placeholder. Submit `type`,
`use_default_env`, trimmed `base_url`, and a newly typed `api_key`; preserve the
masked value when it was loaded and left unchanged. For custom providers,
retain the current base-URL and `api_key_env` controls without default-env
fallback behavior.

In the provider table, add compact localized type/source indicators without
showing environment values or secret values.

- [ ] **Step 5: Add Chinese and English copy**

Add matching keys for: OpenAI-compatible, environment default, configured/not
configured, environment source, manual fallback, fallback URL, fallback API
key, precedence explanation, and accessible labels. Update section help text so
it no longer incorrectly claims that model providers can only store environment
variable names.

- [ ] **Step 6: Run frontend tests and build**

Run: `cd frontend && npm test -- --run src/components/__tests__/ConnectionsPanel.test.tsx`

Expected: component suite passes.

Run: `cd frontend && npm run build`

Expected: TypeScript and Vite build succeed.

- [ ] **Step 7: Commit the frontend experience**

```bash
git add frontend/src/api/agentConfig.ts frontend/src/components/ConnectionsPanel.tsx frontend/src/i18n/en.ts frontend/src/i18n/zh.ts frontend/src/components/__tests__/ConnectionsPanel.test.tsx
git commit -m "feat(ui): configure default model provider fallbacks"
```

### Task 4: Full regression verification and delivery

**Files:**
- Modify only files required to fix regressions found by the commands below.

**Interfaces:**
- Consumes: the completed backend and frontend behavior from Tasks 1–3.
- Produces: a clean, tested branch ready to push.

- [ ] **Step 1: Run backend focused regression suites**

Run: `cd backend && pytest -q tests/test_providers.py tests/test_routes_config.py tests/test_routes_models.py tests/test_lean_main_boot.py`

Expected: all tests pass.

- [ ] **Step 2: Run the full frontend unit suite**

Run: `cd frontend && npm test -- --run`

Expected: all tests pass.

- [ ] **Step 3: Run the frontend production build**

Run: `cd frontend && npm run build`

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 4: Check formatting and repository state**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intentional implementation files are modified.

- [ ] **Step 5: Commit any regression-only corrections**

```bash
git add <only-the-intentional-regression-fix-files>
git commit -m "test: cover environment-backed default provider"
```

Skip this commit when no additional corrections are needed.

- [ ] **Step 6: Push the current branch**

Run: `git push origin personal/yfei/agent-core`

Expected: the remote branch advances to the final local commit.
