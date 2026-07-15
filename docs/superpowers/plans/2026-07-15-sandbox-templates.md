# Sandbox Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the singleton sandbox template with a named `templates` map that Agents bind by key, and split the one Dockerfile into a shared base plus `pairec` and `turbox` layers.

**Architecture:** `sandbox.default.settings.templates` maps a key to `{name, code_writable, env_refs}`. `AgentProfile.sandbox.template` picks one. The provider stays a singleton and resolves the key at sandbox-create time from `ToolScope.metadata["sandbox_template"]` — the same path `default_kb_ids` already uses. Images split into `sandbox/base` (mount contract + toolchain) and two thin per-template Dockerfiles.

**Tech Stack:** Python 3.11 / pydantic / pytest / loguru (backend), React + TypeScript + vitest (frontend), Docker BuildKit.

**Spec:** `docs/superpowers/specs/2026-07-15-sandbox-templates-design.md`

## Global Constraints

- `template_name` is **removed outright**. No back-compat shim, no normalization fallback. Every gate that read it now tests `templates` non-empty.
- An unknown template key **raises**, listing valid keys. Never fall back to a default.
- Only the template **key** enters `ToolScope.metadata`. Resolving `env_refs` reads `os.environ` and stays inside the provider — secrets never cross the builder or the scope.
- `code_writable` lives in the `templates` map, never on `AgentProfile`.
- `agent-sandbox-bootstrap` is unchanged; `CONTRACT_VERSION` stays `"1"`.
- Run backend tests with `cd backend && python -m pytest`, frontend with `cd frontend && npx vitest run`.

---

### Task 1: Config model — `templates` map and `AgentProfile.sandbox`

**Files:**
- Modify: `backend/app/agent_config.py:172-196` (add `AgentSandboxConfig`, wire into `AgentProfile`)
- Modify: `backend/app/agent_config.py:286-300` (`DEFAULT_DOCUMENT` sandbox provider settings)
- Modify: `backend/app/agent_config.py:693-725` (`apply_runtime_status` grading)
- Test: `backend/tests/test_lean_main_boot.py:38`
- Test: `backend/tests/test_agent_config_sandbox_templates.py` (create)

**Interfaces:**
- Produces: `AgentSandboxConfig(template: str = "")`; `AgentProfile.sandbox: AgentSandboxConfig`. Provider settings key `templates: Dict[str, Dict[str, Any]]` and `default_template: str`. Task 2 reads both from `settings`; Task 4 reads `templates` via `agent_config.providers`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_agent_config_sandbox_templates.py`:

```python
from app.agent_config import (
    DEFAULT_DOCUMENT,
    AgentConfigDocument,
    AgentProfile,
    apply_runtime_status,
)


def test_agent_profile_defaults_to_empty_sandbox_template():
    profile = AgentProfile(id="main", name="Main")
    assert profile.sandbox.template == ""


def test_agent_profile_accepts_sandbox_template():
    profile = AgentProfile(**{
        "id": "turbox-helper",
        "name": "Turbo-X",
        "sandbox": {"template": "turbox"},
    })
    assert profile.sandbox.template == "turbox"


def test_default_document_ships_no_template_name():
    provider = next(p for p in DEFAULT_DOCUMENT.providers if p.id == "sandbox.default")
    assert "template_name" not in provider.settings
    assert provider.settings["templates"] == {}
    assert provider.settings["default_template"] == ""


def _doc_with(settings: dict) -> AgentConfigDocument:
    return AgentConfigDocument(**{
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun REST",
            "settings": {"provider": "agentrun_rest", **settings},
            "used_by": ["sandbox"],
        }],
        "capabilities": [{
            "id": "sandbox",
            "kind": "core_tool",
            "name": "Sandbox",
            "enabled": True,
            "permission": "auto",
            "provider_refs": ["sandbox.default"],
        }],
    })


def test_runtime_status_healthy_with_templates():
    doc = _doc_with({
        "templates": {"pairec": {"name": "sandbox-code-feiyue"}},
        "api_key": "secret",
        "account_id": "acct-1",
    })
    apply_runtime_status(doc)
    provider = next(p for p in doc.providers if p.id == "sandbox.default")
    assert provider.status == "healthy"


def test_runtime_status_missing_config_with_empty_templates():
    doc = _doc_with({"templates": {}, "api_key": "secret", "account_id": "acct-1"})
    apply_runtime_status(doc)
    provider = next(p for p in doc.providers if p.id == "sandbox.default")
    assert provider.status == "missing_config"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_agent_config_sandbox_templates.py -v`
Expected: FAIL — `AgentProfile` has no attribute `sandbox`; `templates` not in default settings.

- [ ] **Step 3: Add `AgentSandboxConfig` and wire it into `AgentProfile`**

In `backend/app/agent_config.py`, directly after `AgentCodeConfig` (ends line 176):

```python
class AgentSandboxConfig(BaseModel):
    """Which sandbox template this agent runs on. ``template`` is a key into
    ``sandbox.default.settings.templates``; blank uses that provider's
    ``default_template``. An unknown key is a hard error at sandbox-create time
    (see ``sandbox_providers._resolve_template``) rather than a silent fallback —
    running an agent on the wrong image gives it a ``/opt/code`` that contradicts
    its own code manifest, which is far harder to diagnose than a failed create."""

    template: str = ""
```

In `AgentProfile` (line 193, beside `code`):

```python
    # Which sandbox image this agent gets. Coupled to ``code`` above: the manifest
    # describes the repositories the bound template's image actually ships.
    sandbox: AgentSandboxConfig = Field(default_factory=AgentSandboxConfig)
```

- [ ] **Step 4: Swap `template_name` for `templates` in `DEFAULT_DOCUMENT`**

In `backend/app/agent_config.py:294`, replace the `"template_name": "",` line with:

```python
                "templates": {},
                "default_template": "",
```

- [ ] **Step 5: Regrade `apply_runtime_status` on `templates`**

In `backend/app/agent_config.py`, replace line 706 (`sandbox_settings.get("template_name")`) with:

```python
                bool(sandbox_settings.get("templates"))
```

and line 712 (`bool(sandbox_settings.get("template_name"))`) with:

```python
                bool(sandbox_settings.get("templates"))
```

Update the operator-facing message at line 723-725 to name the new key:

```python
        sandbox_provider.error = None if sandbox_provider.status != "missing_config" else (
            "Configure sandbox provider, templates, and credentials when required"
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_agent_config_sandbox_templates.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Fix the lean-boot assertion**

`test_lean_main_boot.py:38` asserts the `DEFAULT_DOCUMENT` value (`_new_database_config_seed` ignores the YAML entirely and returns a deep copy of the default). Replace line 38:

```python
    assert sandbox.settings["templates"] == {}
```

- [ ] **Step 8: Run the lean-boot test**

Run: `cd backend && python -m pytest tests/test_lean_main_boot.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add backend/app/agent_config.py backend/tests/test_agent_config_sandbox_templates.py backend/tests/test_lean_main_boot.py
git commit -m "feat(config): sandbox templates map and per-agent template binding"
```

---

### Task 2: Provider — resolve the template at create time

**Files:**
- Modify: `backend/agent/tools/scope.py:51` (add `scope_sandbox_template`)
- Modify: `backend/agent/tools/sandbox_providers.py:42-54` (`__init__`), `:428-455` (REST create), `:701-712` (SDK create), `:1155-1175` (`make_sandbox_provider`)
- Test: `backend/tests/test_sandbox_templates.py` (create)
- Test: `backend/tests/test_builtin_tools.py` (12 existing sites)

**Interfaces:**
- Consumes: settings keys `templates`, `default_template` from Task 1.
- Produces: `scope_sandbox_template() -> str`; `ScopedSandboxProvider.templates: Dict[str, Dict[str, Any]]`, `.default_template: str`, `._resolve_template() -> Tuple[str, Dict[str, Any]]`; `_template_env_refs(tpl) -> Dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_sandbox_templates.py`:

```python
import pytest

from agent.tools.sandbox_providers import (
    AgentRunRestSandboxProvider,
    _template_env_refs,
    make_sandbox_provider,
)
from agent.tools.scope import ToolScope, reset_current_tool_scope, set_current_tool_scope
from app.agent_config import AgentConfigDocument


def _provider() -> AgentRunRestSandboxProvider:
    return AgentRunRestSandboxProvider({
        "api_key": "secret",
        "account_id": "acct-1",
        "region": "cn-hangzhou",
        "templates": {
            "pairec": {"name": "sandbox-code-feiyue"},
            "turbox": {
                "name": "sandbox-turbox-feiyue",
                "code_writable": True,
                "env_refs": {"GITLAB_TOKEN": "GITLAB_TOKEN"},
            },
        },
        "default_template": "pairec",
    })


def _with_scope(template: str):
    return set_current_tool_scope(
        ToolScope(agent_id="a1", metadata={"sandbox_template": template})
    )


def test_resolve_uses_scope_template():
    token = _with_scope("turbox")
    try:
        key, tpl = _provider()._resolve_template()
    finally:
        reset_current_tool_scope(token)
    assert key == "turbox"
    assert tpl["name"] == "sandbox-turbox-feiyue"


def test_resolve_falls_back_to_default_when_scope_is_blank():
    token = _with_scope("")
    try:
        key, tpl = _provider()._resolve_template()
    finally:
        reset_current_tool_scope(token)
    assert key == "pairec"
    assert tpl["name"] == "sandbox-code-feiyue"


def test_resolve_unknown_key_raises_and_lists_valid_keys():
    token = _with_scope("nope")
    try:
        with pytest.raises(RuntimeError) as exc:
            _provider()._resolve_template()
    finally:
        reset_current_tool_scope(token)
    message = str(exc.value)
    assert "'nope'" in message
    assert "pairec" in message and "turbox" in message


def test_env_refs_resolve_from_environment(monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-xyz")
    tpl = _provider().templates["turbox"]
    assert _template_env_refs(tpl) == {"GITLAB_TOKEN": "glpat-xyz"}


def test_pairec_template_carries_no_env_refs(monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-xyz")
    tpl = _provider().templates["pairec"]
    assert _template_env_refs(tpl) == {}


def test_unset_env_ref_is_skipped_with_a_warning(monkeypatch):
    """Unset source var: skipped, never raises. The provider logs through loguru,
    which does not propagate to pytest's caplog, so capture the sink directly
    rather than asserting on caplog.records."""
    from loguru import logger

    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), format="{message}", level="WARNING")
    try:
        assert _template_env_refs(_provider().templates["turbox"]) == {}
    finally:
        logger.remove(sink_id)
    assert any("GITLAB_TOKEN" in m for m in messages)


def _doc_with(settings: dict) -> AgentConfigDocument:
    return AgentConfigDocument(**{
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun REST",
            "settings": {"provider": "agentrun_rest", **settings},
            "used_by": ["sandbox"],
        }],
        "capabilities": [{
            "id": "sandbox",
            "kind": "core_tool",
            "name": "Sandbox",
            "enabled": True,
            "permission": "auto",
            "provider_refs": ["sandbox.default"],
        }],
    })


def test_make_sandbox_provider_returns_none_without_templates():
    doc = _doc_with({"templates": {}, "api_key": "secret", "account_id": "acct-1"})
    assert make_sandbox_provider(doc) is None


def test_make_sandbox_provider_builds_with_templates():
    doc = _doc_with({
        "templates": {"pairec": {"name": "sandbox-code-feiyue"}},
        "api_key": "secret",
        "account_id": "acct-1",
    })
    assert make_sandbox_provider(doc) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_sandbox_templates.py -v`
Expected: FAIL — `ImportError: cannot import name '_template_env_refs'`

- [ ] **Step 3: Add `scope_sandbox_template`**

In `backend/agent/tools/scope.py`, after `scope_default_kb_ids` (ends line 51):

```python
def scope_sandbox_template() -> str:
    """The active agent's sandbox template key, threaded through
    ``metadata['sandbox_template']`` by the builder. The sandbox provider is a
    singleton built from global config, so this is how a per-agent template
    reaches it. Only the key travels — the provider resolves it (and any
    ``env_refs`` secrets) against its own settings."""
    return str(get_current_tool_scope().metadata.get("sandbox_template") or "")
```

- [ ] **Step 4: Store the map at construction**

In `backend/agent/tools/sandbox_providers.py`, add the import at line 17:

```python
from agent.tools.scope import get_current_tool_scope, scope_sandbox_template
```

Replace line 44 (`self.template_name = str(settings.get("template_name") or "")`) with:

```python
        self.templates = dict(settings.get("templates") or {})
        self.default_template = str(settings.get("default_template") or "")
```

- [ ] **Step 5: Add `_resolve_template` and `_template_env_refs`**

Add to `ScopedSandboxProvider` (after `__init__`):

```python
    def _resolve_template(self) -> Tuple[str, Dict[str, Any]]:
        """Map the active agent's template key to its settings block. Resolution
        happens here, per create, rather than in ``__init__``: the provider is a
        singleton shared by every agent, while the template is per-agent."""
        key = scope_sandbox_template() or self.default_template
        tpl = self.templates.get(key)
        if tpl is None:
            raise RuntimeError(
                f"sandbox template {key!r} is not defined; "
                f"valid templates: {sorted(self.templates)}"
            )
        return key, tpl
```

Add at module level, beside the other `_`-prefixed helpers:

```python
def _template_env_refs(tpl: Dict[str, Any]) -> Dict[str, str]:
    """Resolve a template's ``env_refs`` ({VAR: SOURCE_ENV_NAME}) against the
    service's own environment. Kept in the provider so secrets never cross the
    builder or the tool scope — only the template key travels.

    An unset source variable is skipped with a warning rather than failing the
    create: env delivery is already best-effort (see _bootstrap_env_async), so a
    token-less sandbox is reachable regardless, and it still serves every use
    that isn't `git fetch`. The warning is what turns the eventual 401 into a
    lookup instead of a mystery."""
    resolved: Dict[str, str] = {}
    for var, env_name in (tpl.get("env_refs") or {}).items():
        value = os.environ.get(str(env_name))
        if not value:
            logger.warning(
                "sandbox template env_ref {}={} is unset; sandbox will start without it",
                var, env_name,
            )
            continue
        resolved[str(var)] = value
    return resolved
```

Ensure `Tuple` is in the `typing` import at the top of the file.

- [ ] **Step 6: Resolve in the REST create path**

In `_create_sandbox_async` (line ~428), replace the `scope`/`env_contract`/`payload` block:

```python
        scope = get_current_tool_scope()
        key, tpl = self._resolve_template()
        env_contract = _build_env_contract(self, scope, scope_key) or {}
        # env_refs ride the same ~/.bash_env path as the AGENT_* contract
        # (_bootstrap_env_async) — no separate delivery mechanism.
        env_contract.update(_template_env_refs(tpl))
        payload = _compact_dict({
            "templateName": tpl["name"],
            "templateType": self.template_type or None,
            "sandboxId": _scoped_sandbox_id(self.settings, scope_key),
            "nasConfig": _build_nas_config(self, scope, scope_key),
            # NOTE: AgentRun's CreateSandbox input has no `envs` field — this is
            # ignored by the platform and kept only for forward-compat. The env
            # contract is delivered by _bootstrap_env_async below (writes
            # ~/.bash_env + ~/.aliyun/config.json inside the started sandbox).
            "envs": env_contract,
        })
```

Update the create-failure message (line ~450) to carry the key, so an operator can see which agent bound wrong:

```python
            raise RuntimeError(
                f"sandbox create failed (template={key!r} -> {tpl['name']!r}, "
                f"account_id={self.parent_id!r}): {exc}. "
                f"Verify the template exists in AgentRun under that account (and "
                f"region), and that settings.templates[{key!r}].name matches it."
            ) from exc
```

- [ ] **Step 7: Resolve in the SDK create path**

In `AgentRunSdkSandboxProvider._create_sandbox` (line ~701), replace the `if not self.template_name:` guard and the `template_name=` argument:

```python
        key, tpl = self._resolve_template()
```

then pass `template_name=tpl["name"],` to `Sandbox.create`. Delete the old
`raise RuntimeError("sandbox provider requires settings.template_name")` — an
unresolvable template now raises from `_resolve_template` with a better message.

- [ ] **Step 8: Regate `make_sandbox_provider`**

At line ~1161 replace `settings.get("template_name")` with `settings.get("templates")`, and at line ~1168 replace `if not settings.get("template_name"):` with:

```python
        if not settings.get("templates"):
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_sandbox_templates.py -v`
Expected: PASS (8 tests)

- [ ] **Step 10: Migrate the existing provider tests**

In `backend/tests/test_builtin_tools.py`, replace every occurrence of:

```python
                "template_name": "code-template",
```

with:

```python
                "templates": {"default": {"name": "code-template"}},
                "default_template": "default",
```

Sites (12): lines 346, 376, 398, 414, 473, 604, 750, 811, 877, 938, 993, 1017. Watch the indentation — line 398 and 473 sit in a flatter dict literal than the rest.

- [ ] **Step 11: Run the full backend suite**

Run: `cd backend && python -m pytest tests/test_builtin_tools.py tests/test_sandbox_templates.py -v`
Expected: PASS, no `template_name` references remain:

Run: `rg -n "template_name" backend/` → Expected: no matches.

- [ ] **Step 12: Commit**

```bash
git add backend/agent/tools/scope.py backend/agent/tools/sandbox_providers.py backend/tests/test_sandbox_templates.py backend/tests/test_builtin_tools.py
git commit -m "feat(sandbox): resolve template per agent at create time"
```

---

### Task 3: Prompt — branch the code-layer block on `code_writable`

**Files:**
- Modify: `backend/agent/soul.py:197-217` (`_code_layer_block`), `:220-240` (`_render_capability_guidance`), `:243-262` (`render_stable_system_prompt`), `:288-302` (`render_subagent_system_prompt`)
- Test: `backend/tests/test_soul_code_writable.py` (create)

**Interfaces:**
- Produces: `code_writable: bool = False` keyword on `_code_layer_block`, `_render_capability_guidance`, `render_stable_system_prompt`, `render_subagent_system_prompt`. Task 4 passes it from the resolved template.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_soul_code_writable.py`:

```python
from agent.soul import render_stable_system_prompt, render_subagent_system_prompt

MANIFEST = "- image-metadata: PAI image metadata\n- pai-wiki: internal docs"
TOOLS = ["shell", "code_interpreter"]


def _prompt(writable: bool, render=render_stable_system_prompt) -> str:
    return render(
        "You are a helper.", tool_names=TOOLS,
        code_enabled=True, code_manifest=MANIFEST, code_writable=writable,
    )


def test_read_only_prompt_forbids_modification():
    prompt = _prompt(False)
    assert "do not try to modify it" in prompt
    assert "read-only code layer" in prompt


def test_writable_prompt_does_not_forbid_modification():
    prompt = _prompt(True)
    assert "do not try to modify it" not in prompt
    assert "read-only reference material" not in prompt


def test_writable_prompt_describes_git_working_copies():
    prompt = _prompt(True)
    assert "git working copies" in prompt
    assert "check out" in prompt
    assert "discarded" in prompt  # ephemerality must be stated


def test_writable_flag_reaches_subagent_prompt():
    prompt = _prompt(True, render=render_subagent_system_prompt)
    assert "do not try to modify it" not in prompt
    assert "git working copies" in prompt


def test_manifest_still_leads_the_block():
    assert MANIFEST in _prompt(True)
    assert MANIFEST in _prompt(False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_soul_code_writable.py -v`
Expected: FAIL — `render_stable_system_prompt() got an unexpected keyword argument 'code_writable'`

- [ ] **Step 3: Branch `_code_layer_block`**

Replace `backend/agent/soul.py:197-217` entirely:

```python
def _code_layer_block(code_manifest: str, code_writable: bool = False) -> str:
    """Code repository guidance. With a manifest (the per-agent, admin-curated
    list of what each repo is), lead with it so the model knows the repos up
    front; without one, fall back to discover-by-`ls`.

    ``code_writable`` comes from the bound template's settings, not the agent —
    it describes what the *image* did (turbox chowns /opt/code to the runtime
    uid; pairec leaves it root-owned). Getting this wrong contradicts the
    manifest inside one prompt, which reads to the model as an instruction not
    to touch repositories its manifest says it may check out."""
    manifest = (code_manifest or "").strip()
    if not manifest:
        return _CODE_LAYER_GUIDANCE
    intro = (
        "The code layer at `/opt/code` holds the source repositories behind this "
        "system as git working copies, one per subdirectory. The available "
        "repositories:"
        if code_writable else
        "A read-only code layer at `/opt/code` holds the source repositories "
        "behind this system, one per subdirectory. The available repositories:"
    )
    outro = (
        "You may check out other branches in these repositories (every branch "
        "that existed when the image was built is already fetched; `git fetch "
        "--deepen` or `--unshallow` for older history). Any change you make is "
        "local to this sandbox and is discarded when it ends — nothing you do "
        "here reaches the upstream repository, and a checkout does not persist "
        "to your next session."
        if code_writable else
        "It is read-only reference material — do not try to modify it — and it "
        "is a fallback for source-level questions, not a replacement for "
        "knowledge_search on document questions."
    )
    return (
        intro + "\n\n" + manifest + "\n\n"
        "When knowledge_search / the knowledge base does not answer a question "
        "that is really about how this system's code behaves, fall back to the "
        "code: open the relevant repository under `/opt/code` and explore it with "
        "shell / code_interpreter (ripgrep or grep to find symbols, cat to read "
        "files); run `ls /opt/code` for anything the list above does not cover. "
        + outro
    )
```

- [ ] **Step 4: Thread the flag through the three callers**

`_render_capability_guidance` (line ~220) — add the parameter and pass it:

```python
def _render_capability_guidance(
    *,
    tool_names: List[str],
    aliyun_pai_enabled: bool,
    code_enabled: bool,
    code_manifest: str,
    code_writable: bool = False,
) -> List[str]:
```

and at its `_code_layer_block` call:

```python
            blocks.append(_code_layer_block(code_manifest, code_writable))
```

`render_stable_system_prompt` (line ~243) — add `code_writable: bool = False,` to the signature and pass `code_writable=code_writable,` into `_render_capability_guidance`.

`render_subagent_system_prompt` (line ~288) — add `code_writable: bool = False,` to the signature and pass `code_writable=code_writable,` into its `render_stable_system_prompt` call.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_soul_code_writable.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Run the existing soul/prompt tests for regressions**

Run: `cd backend && python -m pytest tests/ -k "soul or prompt or instructions" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/agent/soul.py backend/tests/test_soul_code_writable.py
git commit -m "feat(soul): branch code-layer guidance on template writability"
```

---

### Task 4: Builder — thread the template key and writability

**Files:**
- Modify: `backend/app/builder.py:165-176` (top-level render), `:243-250` (metadata), `:310-318` (subagent render), `:339-350` (subagent metadata)
- Test: `backend/tests/test_builder_sandbox_template.py` (create)

**Interfaces:**
- Consumes: `AgentProfile.sandbox.template` (Task 1); `code_writable` kwarg (Task 3); `metadata["sandbox_template"]` read by `scope_sandbox_template` (Task 2).
- Produces: `_sandbox_template_settings(agent_config) -> Dict[str, Any]`, `_resolve_agent_template(agent_config, profile) -> Tuple[str, bool]`, `_apply_sandbox_template(metadata: Dict[str, Any], sandbox_template: str) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_builder_sandbox_template.py`:

```python
from app.agent_config import AgentConfigDocument
from app.builder import _resolve_agent_template


def _doc() -> AgentConfigDocument:
    return AgentConfigDocument(**{
        "default_agent": "main",
        "agents": [
            {"id": "main", "name": "Main"},
            {"id": "turbox-helper", "name": "Turbo-X", "sandbox": {"template": "turbox"}},
        ],
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun REST",
            "settings": {
                "provider": "agentrun_rest",
                "templates": {
                    "pairec": {"name": "sandbox-code-feiyue"},
                    "turbox": {"name": "sandbox-turbox-feiyue", "code_writable": True},
                },
                "default_template": "pairec",
            },
        }],
    })


def test_blank_agent_template_resolves_to_default_and_read_only():
    doc = _doc()
    profile = next(a for a in doc.agents if a.id == "main")
    key, writable = _resolve_agent_template(doc, profile)
    assert key == ""          # metadata stays blank; the provider applies its default
    assert writable is False  # ...but the prompt must match the default template


def test_bound_agent_template_resolves_to_writable():
    doc = _doc()
    profile = next(a for a in doc.agents if a.id == "turbox-helper")
    key, writable = _resolve_agent_template(doc, profile)
    assert key == "turbox"
    assert writable is True


def test_unknown_template_is_not_writable():
    doc = _doc()
    profile = next(a for a in doc.agents if a.id == "turbox-helper")
    profile.sandbox.template = "nope"
    key, writable = _resolve_agent_template(doc, profile)
    assert key == "nope"      # passed through; the provider raises on create
    assert writable is False


def test_missing_provider_resolves_to_read_only():
    doc = AgentConfigDocument(**{"agents": [{"id": "main", "name": "Main"}]})
    profile = doc.agents[0]
    assert _resolve_agent_template(doc, profile) == ("", False)
```

And the set-or-pop tests, which must call the production helper — reimplementing
the if/else in the test body would pass whether or not `builder.py` pops, and the
missing pop is exactly the bug these guard:

```python
from app.builder import _apply_sandbox_template


def test_apply_sets_the_childs_own_template_over_the_parents():
    metadata = {"sandbox_template": "pairec", "subagent_depth": 0}
    _apply_sandbox_template(metadata, "turbox")
    assert metadata["sandbox_template"] == "turbox"


def test_apply_clears_an_inherited_template_when_the_child_has_none():
    """A subagent's metadata starts as a copy of the parent's scope. Without the
    pop, an unbound child silently runs on the parent's image."""
    metadata = {"sandbox_template": "pairec", "subagent_depth": 0}
    _apply_sandbox_template(metadata, "")
    assert "sandbox_template" not in metadata
    assert metadata["subagent_depth"] == 0  # unrelated keys survive


def test_apply_is_a_noop_when_there_is_nothing_to_inherit_or_set():
    metadata = {"subagent_depth": 0}
    _apply_sandbox_template(metadata, "")
    assert metadata == {"subagent_depth": 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_builder_sandbox_template.py -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_agent_template'`

- [ ] **Step 3: Add the resolution helpers**

In `backend/app/builder.py`, beside `_capability_enabled` (line ~371). This mirrors `aliyun_sts.provider_settings`:

```python
def _sandbox_template_settings(agent_config) -> Dict[str, Any]:
    """The ``sandbox.default`` provider's settings block, or ``{}`` (duck-typed
    so callers don't depend on the pydantic model)."""
    for p in (getattr(agent_config, "providers", []) or []):
        if getattr(p, "id", "") == "sandbox.default":
            return dict(getattr(p, "settings", {}) or {})
    return {}


def _resolve_agent_template(agent_config, agent_profile) -> Tuple[str, bool]:
    """Return (template key for the tool scope, code_writable for the prompt).

    The key is passed through verbatim — including an unknown one, which the
    provider rejects at create time with a message listing valid keys. Only the
    key travels into the scope; ``env_refs`` are resolved inside the provider so
    secrets never cross this layer.

    ``code_writable`` is looked up eagerly because the prompt is rendered now,
    before any sandbox exists. A blank key still resolves against the provider's
    ``default_template`` so the prompt matches the image the agent will actually
    get, and anything unresolvable falls back to read-only — the conservative
    wording is wrong-but-harmless, whereas a spurious "you may check out
    branches" against a root-owned /opt/code is not.
    """
    settings = _sandbox_template_settings(agent_config)
    key = str(getattr(getattr(agent_profile, "sandbox", None), "template", "") or "")
    templates = settings.get("templates") or {}
    lookup = key or str(settings.get("default_template") or "")
    tpl = templates.get(lookup) or {}
    return key, bool(tpl.get("code_writable"))


def _apply_sandbox_template(metadata: Dict[str, Any], sandbox_template: str) -> None:
    """Set-or-pop the scope's template key. Never inherit: a subagent's metadata
    starts as a copy of the parent's scope, so a child with no binding of its own
    must clear the parent's key and fall back to the provider's default rather
    than silently running on the parent's image."""
    if sandbox_template:
        metadata["sandbox_template"] = sandbox_template
    else:
        metadata.pop("sandbox_template", None)
```

Ensure `Tuple`, `Dict`, and `Any` are imported from `typing` at the top of `builder.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_builder_sandbox_template.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Wire the top-level context**

In `build_context`, replace lines 166-172:

```python
    code_config = getattr(agent_profile, "code", None)
    code_enabled = bool(getattr(code_config, "enabled", False))
    code_manifest = getattr(code_config, "manifest", "") or ""
    sandbox_template, code_writable = _resolve_agent_template(agent_config, agent_profile)
    # The agent's ``instructions`` markdown IS the persona (base system prompt);
    # blank falls back to the built-in DEFAULT_INSTRUCTIONS.
    instructions_md = (getattr(agent_profile, "instructions", "") or "").strip() or DEFAULT_INSTRUCTIONS
    system_prompt = render_stable_system_prompt(
        instructions_md, tool_names=tool_names, project_context=project_context,
        aliyun_pai_enabled=_aliyun_pai_enabled(),
        code_enabled=code_enabled, code_manifest=code_manifest,
        code_writable=code_writable,
    )
```

Then, in the metadata block beside `metadata["default_kb_ids"]` (line ~247), add:

```python
    # Per-agent sandbox template. Only the key — the provider resolves it (and any
    # env_refs secrets) against its own settings at create time.
    _apply_sandbox_template(metadata, sandbox_template)
```

- [ ] **Step 6: Wire the subagent context**

In `build_subagent_context`, replace lines 310-317:

```python
    code_config = getattr(profile, "code", None)
    sandbox_template, code_writable = _resolve_agent_template(agent_config, profile)
    instructions_md = (getattr(profile, "instructions", "") or "").strip() or DEFAULT_INSTRUCTIONS
    system_prompt = render_subagent_system_prompt(
        instructions_md, tool_names=effective_names, project_context=project_context,
        aliyun_pai_enabled=_aliyun_pai_enabled(),
        code_enabled=bool(getattr(code_config, "enabled", False)),
        code_manifest=getattr(code_config, "manifest", "") or "",
        code_writable=code_writable,
    )
```

Then in the inherited-metadata block (line ~341, right after `metadata["subagent_depth"] = depth`), call the same helper. **This must be explicit**: the block starts from `dict(parent_scope.metadata or {})`, so without the pop a turbo-x subagent under a pai-rec parent silently inherits and runs on the pai-rec image.

```python
    _apply_sandbox_template(metadata, sandbox_template)
```

- [ ] **Step 7: Run the full backend suite**

Run: `cd backend && python -m pytest`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/app/builder.py backend/tests/test_builder_sandbox_template.py
git commit -m "feat(builder): thread sandbox template key and writability per agent"
```

---

### Task 5: Images — split base / pairec / turbox

**Files:**
- Create: `sandbox/base/Dockerfile`, `sandbox/pairec/Dockerfile`, `sandbox/turbox/Dockerfile`, `sandbox/build.sh`
- Move: `sandbox/agent-sandbox-bootstrap` → `sandbox/base/agent-sandbox-bootstrap` (contents unchanged)
- Delete: `sandbox/Dockerfile`

**Interfaces:**
- Consumes: nothing from earlier tasks (images are independent of the Python change).
- Produces: local image tags `sandbox-base:latest`, `sandbox-pairec:latest`, `sandbox-turbox:latest`.

- [ ] **Step 1: Move the bootstrap unchanged**

```bash
mkdir -p sandbox/base sandbox/pairec sandbox/turbox
git mv sandbox/agent-sandbox-bootstrap sandbox/base/agent-sandbox-bootstrap
```

Do **not** edit it. It validates only `/mnt/system|skills|user`; `/opt/code` is outside its checks, so the mount contract is unaffected and `CONTRACT_VERSION` stays `"1"`.

- [ ] **Step 2: Write `sandbox/base/Dockerfile`**

Take `sandbox/Dockerfile` as the source, drop the `PAIREC_CODE_ARCHIVE_URL` layer and the `aliyun plugin install` layer, and add the toolchain + credential helper:

```dockerfile
# syntax=docker/dockerfile:1
# =============================================================================
# PAI-Loop AgentRun sandbox — shared base layer.
#
# Owns the runtime contract (docs/design/skill_install_mount_dependencies.md):
# the three NAS mount points, the AGENT_* env vars, the startup validation, and
# the toolchain every template shares. Per-template images (sandbox/pairec,
# sandbox/turbox) FROM this and add only their code layer and aliyun plugins.
#
#   docker build -t sandbox-base:latest \
#     --build-arg BASE_IMAGE=<official-code-interpreter-image-ref> sandbox/base/
#
# Not pushed to a registry — sandbox/build.sh builds it locally first. A clean
# CI machine must run both steps or `FROM sandbox-base:latest` will not resolve.
# =============================================================================

ARG BASE_IMAGE=registry.aliyuncs.com/agentrun/sandbox-code-interpreter:latest
FROM ${BASE_IMAGE}

ARG ALIYUN_CLI_URL=https://easyrec.oss-cn-beijing.aliyuncs.com/aliyun-cli-spec/aliyun-cli-linux-latest-amd64.tgz
ARG YQ_VERSION=v4.44.3

# The platform runs the sandbox as uid/gid 1000 (matches nasConfig.userId /
# groupId). Content for /mnt/system and /mnt/skills arrives via read-only NAS
# mounts at sandbox start — they stay empty dirs in the image.
USER root
RUN mkdir -p /mnt/system /mnt/skills /mnt/user /opt/code /home/user \
 && chown -R 1000:1000 /mnt/user /home/user

# Toolchain shared by every template: ripgrep/jq for the browse+grep workload the
# agent is prompted to run against /opt/code, git for template code layers that
# are working copies rather than snapshots.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ripgrep jq git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && rg --version && jq --version && git --version

# yq is not in Debian main; pinned binary from the upstream release.
RUN curl -fsSL "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64" \
      -o /usr/local/bin/yq \
 && chmod 0755 /usr/local/bin/yq && yq --version

# Aliyun CLI binary only — each template installs its own plugins as uid 1000.
RUN tmpdir="$(mktemp -d)" \
    && curl -fsSL "${ALIYUN_CLI_URL}" -o /tmp/aliyun-cli.tgz \
    && tar -xzf /tmp/aliyun-cli.tgz -C "${tmpdir}" \
    && install -m 0755 "${tmpdir}/aliyun" /usr/local/bin/aliyun \
    && rm -rf "${tmpdir}" /tmp/aliyun-cli.tgz \
    && aliyun version

# Secret-free credential helper. `credential.helper = env` makes git exec
# git-credential-env from PATH; the token itself arrives at runtime in
# $GITLAB_TOKEN via the agent service's env contract (~/.bash_env), so nothing
# secret is baked into any layer. Templates that need it bind it to a host.
RUN printf '%s\n' \
      '#!/bin/sh' \
      '# git credential helper: reads the token from the environment.' \
      '[ "$1" = "get" ] || exit 0' \
      'echo "username=oauth2"' \
      'echo "password=${GITLAB_TOKEN}"' \
      > /usr/local/bin/git-credential-env \
 && chmod 0755 /usr/local/bin/git-credential-env

# Startup validation: fails fast if a contract mount is missing/unreadable.
COPY agent-sandbox-bootstrap /usr/local/bin/agent-sandbox-bootstrap
RUN chmod +x /usr/local/bin/agent-sandbox-bootstrap

# Runtime env contract. HOME is explicit because a numeric USER does not always
# populate it, and the aliyun plugins each template installs must land in $HOME.
ENV AGENT_SYSTEM_PATH=/mnt/system \
    AGENT_SKILL_PATH=/mnt/skills \
    AGENT_USER_PATH=/mnt/user \
    AGENT_CODE_PATH=/opt/code \
    AGENT_ENV_PATH=/mnt/system/skill-envs/current \
    PATH=/mnt/system/skill-envs/current/bin:/usr/local/bin:/mnt/system/bin:${PATH} \
    VIRTUAL_ENV=/mnt/system/skill-envs/current \
    HOME=/home/user

# NOTE: /mnt/skills is intentionally NOT added to PYTHONPATH globally —
# cross-skill module name collisions. Skill scripts are invoked by absolute path.

ENV SANDBOX_BASE_ENTRYPOINT=/usr/local/bin/entrypoint.sh \
    SANDBOX_BASE_CMD="process-compose up --tui=false --no-server"

USER 1000

# FC uses the image ENTRYPOINT as the start command and CMD as its args. Setting
# ENTRYPOINT clears the CMD inherited from the base image, so the base startup
# chain is restored explicitly; keep CMD's first arg as "process-compose" so
# entrypoint.sh takes its config-injection branch.
ENTRYPOINT ["agent-sandbox-bootstrap"]
CMD ["/usr/local/bin/entrypoint.sh", "process-compose", "up", "--tui=false", "--no-server"]
```

- [ ] **Step 3: Write `sandbox/pairec/Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1
# =============================================================================
# pai-rec sandbox template. Adds a release-pinned source snapshot at /opt/code
# and the PAI recommendation plugins to the shared base.
#
#   sandbox/build.sh pairec
#
# Register in AgentRun as a code-interpreter template with network mode PUBLIC,
# under the same account_id + region the agent service uses, and point
# settings.templates.pairec.name at it.
# =============================================================================

FROM sandbox-base:latest

# Release-pinned snapshot of the source repositories (one per subdirectory) the
# agent greps and reads when the knowledge base can't answer. Baked rather than
# NAS-mounted so grep/read hit local disk instead of NFS. Bump the URL to ship
# newer source; pass an empty value to build without the layer.
ARG CODE_ARCHIVE_URL=https://pai-rag.oss-cn-hangzhou.aliyuncs.com/production_artifacts/pairec_code/pairec_code_20260623.tar.gz

# Root-owned + world-readable: this template's /opt/code is a read-only snapshot
# (templates.pairec has no code_writable), so the runtime user only ever reads it.
# --strip-components=1 drops the archive's single top-level wrapper dir so repos
# land at /opt/code/<repo> rather than /opt/code/<wrapper>/<repo>.
USER root
RUN if [ -n "${CODE_ARCHIVE_URL}" ]; then \
      curl -fsSL "${CODE_ARCHIVE_URL}" -o /tmp/code.tar.gz \
      && tar -xzf /tmp/code.tar.gz --strip-components=1 -C /opt/code \
      && rm -f /tmp/code.tar.gz \
      && chmod -R a+rX /opt/code \
      && echo "baked code layer:" && ls -1 /opt/code ; \
    fi

# Installed AS uid 1000 so they land in the agent's $HOME (~/.aliyun) and are
# available to skill scripts at run time. Requires network access during build.
USER 1000
RUN aliyun plugin install --names \
       aliyun-cli-eas \
       aliyun-cli-pairecservice \
       aliyun-cli-pai-dsw \
       aliyun-cli-paifeaturestore \
 && aliyun plugin list
```

- [ ] **Step 4: Write `sandbox/turbox/Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1
# =============================================================================
# turbo-x sandbox template. Clones the intranet gitlab repositories into
# /opt/code as live git working copies the agent can check out branches in.
#
#   sandbox/build.sh turbox      # needs GITLAB_TOKEN and intranet access
#
# Register in AgentRun as a code-interpreter template with network mode VPC
# (vswitch + security group reaching gitlab.alibaba-inc.com), under the same
# account_id + region as pai-rec, and point settings.templates.turbox.name at it.
# settings.templates.turbox must set code_writable: true — the prompt's wording
# is driven by it, and a mismatch tells the model not to touch repositories its
# own manifest says it may check out.
# =============================================================================

FROM sandbox-base:latest

ARG TURBOX_REPOS="\
  http://gitlab.alibaba-inc.com/PAI/image-metadata.git@master \
  http://gitlab.alibaba-inc.com/pai-ee/pai-wiki.git@master"

# --depth 1 --no-single-branch fetches every branch tip at one commit each, so the
# agent can check out any branch that existed at build time offline, and
# `git fetch --deepen`/`--unshallow` for history (the VPC template reaches gitlab).
#
# The token arrives as a BuildKit secret and is consumed via `store --file=`, so
# the remote URL stays clean. Do NOT embed it in the URL
# (http://user:token@gitlab...): .git is deliberately RETAINED here, so the token
# would persist in /opt/code/<repo>/.git/config and ship with the image.
#
# chown to 1000 is what makes checkout possible, and is why this template sets
# code_writable. Writes land in the container's per-instance layer and are
# discarded with the sandbox — the image and upstream are never touched.
USER root
RUN --mount=type=secret,id=gitcred,target=/root/.git-credentials \
    set -eu; \
    for spec in ${TURBOX_REPOS}; do \
      url="${spec%@*}"; ref="${spec##*@}"; name="$(basename "$url" .git)"; \
      git -c credential.helper='store --file=/root/.git-credentials' \
          clone --depth 1 --no-single-branch -b "$ref" "$url" "/opt/code/$name"; \
    done \
 && git config --system credential."http://gitlab.alibaba-inc.com".helper env \
 && chown -R 1000:1000 /opt/code \
 && echo "cloned code layer:" && ls -1 /opt/code

USER 1000
RUN aliyun plugin install --names aliyun-cli-pai-dsw && aliyun plugin list
```

- [ ] **Step 5: Write `sandbox/build.sh`**

```bash
#!/usr/bin/env bash
# Build a sandbox template image. Always builds the shared base first — the
# per-template Dockerfiles do `FROM sandbox-base:latest`, which does not resolve
# on a machine that has never built it.
#
#   sandbox/build.sh pairec
#   sandbox/build.sh turbox            # needs GITLAB_TOKEN + intranet access
#
# BASE_IMAGE  the official AgentRun code-interpreter image ref (console:
#             code-interpreter template -> image address).
# GITLAB_TOKEN  read token for gitlab.alibaba-inc.com; turbox only. Passed as a
#             BuildKit secret, never as a build-arg (build-args land in
#             `docker history`).
set -euo pipefail

TEMPLATE="${1:-}"
BASE_IMAGE="${BASE_IMAGE:-registry.aliyuncs.com/agentrun/sandbox-code-interpreter:latest}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$TEMPLATE" in
  pairec|turbox) ;;
  *) echo "usage: $0 {pairec|turbox}" >&2; exit 2 ;;
esac

echo "==> building sandbox-base:latest (BASE_IMAGE=${BASE_IMAGE})"
DOCKER_BUILDKIT=1 docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t sandbox-base:latest \
  "${here}/base"

echo "==> building sandbox-${TEMPLATE}:latest"
if [ "$TEMPLATE" = "turbox" ]; then
  : "${GITLAB_TOKEN:?turbox needs GITLAB_TOKEN (read token for gitlab.alibaba-inc.com)}"
  cred="$(mktemp)"
  trap 'rm -f "$cred"' EXIT
  printf 'http://oauth2:%s@gitlab.alibaba-inc.com\n' "$GITLAB_TOKEN" > "$cred"
  DOCKER_BUILDKIT=1 docker build \
    --secret "id=gitcred,src=${cred}" \
    -t "sandbox-${TEMPLATE}:latest" \
    "${here}/${TEMPLATE}"
else
  DOCKER_BUILDKIT=1 docker build \
    -t "sandbox-${TEMPLATE}:latest" \
    "${here}/${TEMPLATE}"
fi

echo "==> built sandbox-${TEMPLATE}:latest"
```

```bash
chmod +x sandbox/build.sh
```

- [ ] **Step 6: Delete the old Dockerfile**

```bash
git rm sandbox/Dockerfile
```

- [ ] **Step 7: Verify the pairec image builds and behaves**

Run (substitute the real base image ref):

```bash
BASE_IMAGE=<official-code-interpreter-image-ref> sandbox/build.sh pairec
docker run --rm --entrypoint sh sandbox-pairec:latest -c \
  'rg --version && jq --version && yq --version && git --version && aliyun version && ls -1 /opt/code && stat -c "%u %n" /opt/code'
```

Expected: all five tools print versions; `/opt/code` lists repo subdirectories; `stat` shows owner `0` (root-owned read-only snapshot).

- [ ] **Step 8: Verify the turbox image builds and behaves**

Requires intranet access and a read token:

```bash
GITLAB_TOKEN=<read-token> BASE_IMAGE=<official-code-interpreter-image-ref> sandbox/build.sh turbox
docker run --rm --entrypoint sh sandbox-turbox:latest -c \
  'stat -c "%u %n" /opt/code && ls -1 /opt/code && git -C /opt/code/pai-wiki branch -r'
```

Expected: `/opt/code` owner is `1000`; both `image-metadata` and `pai-wiki` present; `branch -r` lists every remote branch **without network** (proving `--no-single-branch` worked).

- [ ] **Step 9: Verify no token reached any layer**

The `:?` guard is load-bearing: with `GITLAB_TOKEN` unset, `grep "$GITLAB_TOKEN"` searches for the empty string, matches every line, and reports a false "token leaked".

```bash
: "${GITLAB_TOKEN:?set it to the same token you built with, or this check is meaningless}"
docker history --no-trunc sandbox-turbox:latest | grep -i -c "$GITLAB_TOKEN" || echo "OK: not in history"
docker save sandbox-turbox:latest | tar -xO | strings | grep -c "$GITLAB_TOKEN" || echo "OK: not in layers"
docker run --rm --entrypoint sh sandbox-turbox:latest -c 'cat /opt/code/pai-wiki/.git/config'
```

Expected: both greps print `0` then `OK:` (grep -c prints the count and exits 1 on no match, which is what fires the `||`), and `.git/config`'s remote URL is a bare `http://gitlab.alibaba-inc.com/...` with no credentials.

- [ ] **Step 10: Commit**

```bash
git add sandbox/
git commit -m "feat(sandbox): split image into shared base plus pairec and turbox templates"
```

---

### Task 6: Frontend — templates editor and per-agent picker

**Files:**
- Modify: `frontend/src/components/SettingsView.tsx:2414-2520` (provider template field), `:1075-1100` (agent code section)
- Modify: `frontend/src/i18n/en.ts`, `frontend/src/i18n/zh.ts`
- Test: `frontend/src/components/__tests__/SettingsView.test.tsx:68, 185, 222, 230`

**Interfaces:**
- Consumes: settings shape `templates: Record<string, {name, code_writable?, env_refs?}>`, `default_template: string` (Task 1); `agent.sandbox.template` (Task 1).

- [ ] **Step 1: Read the current sandbox settings section**

Run: `sed -n '2405,2525p' frontend/src/components/SettingsView.tsx`

Note how `templateName` state is initialized, validated (`!templateName.trim()` gates the save), and submitted, and how the sibling inputs are laid out. Match that structure.

- [ ] **Step 2: Update the existing tests to the new shape**

In `frontend/src/components/__tests__/SettingsView.test.tsx`, line 68 replace `template_name: "",` with:

```ts
        templates: {},
        default_template: "",
```

At lines 185 and 230 replace the assertion:

```ts
    expect(provider?.settings.templates).toEqual({ default: { name: "code-template" } });
```

At line 222 update the comment to read `// required trio (templates + api_key + account_id) without an endpoint.`

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/__tests__/SettingsView.test.tsx`
Expected: FAIL — the form still writes `template_name`.

- [ ] **Step 4: Replace the single input with a templates editor**

In `SettingsView.tsx`, replace the `templateName` state (line 2414) with:

```tsx
  const [templates, setTemplates] = useState<Record<string, SandboxTemplate>>(
    () => (settings.templates as Record<string, SandboxTemplate>) ?? {},
  );
  const [defaultTemplate, setDefaultTemplate] = useState(String(settings.default_template ?? ""));
```

Add the type near the component:

```tsx
type SandboxTemplate = {
  name: string;
  code_writable?: boolean;
  env_refs?: Record<string, string>;
};
```

Replace the validation gate (line 2425, `!templateName.trim() ||`) with:

```tsx
      Object.keys(templates).length === 0 ||
      Object.values(templates).some((t) => !t.name.trim()) ||
```

Replace the submitted field (line 2447, `template_name: templateName,`) with:

```tsx
                templates,
                default_template: defaultTemplate,
```

Replace the single text input (line ~2514) with a row-per-template editor. The handlers below are complete; take the class names and the label/help wrapper from the sibling inputs you read in Step 1 rather than introducing new styling.

```tsx
const setTemplate = (key: string, patch: Partial<SandboxTemplate>) =>
  setTemplates((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }));

const renameTemplate = (from: string, to: string) =>
  setTemplates((prev) => {
    if (!to.trim() || (to !== from && to in prev)) return prev;
    const next: Record<string, SandboxTemplate> = {};
    // rebuild in order so the row does not jump while typing
    for (const [k, v] of Object.entries(prev)) next[k === from ? to : k] = v;
    return next;
  });

const removeTemplate = (key: string) =>
  setTemplates((prev) => {
    const { [key]: _drop, ...rest } = prev;
    return rest;
  });

const addTemplate = () =>
  setTemplates((prev) => ({ ...prev, [`template-${Object.keys(prev).length + 1}`]: { name: "" } }));

// env_refs <-> "VAR=SOURCE_ENV_VAR, VAR2=OTHER" round-trip
const envRefsToText = (refs?: Record<string, string>) =>
  Object.entries(refs ?? {}).map(([v, e]) => `${v}=${e}`).join(", ");

const envRefsFromText = (text: string): Record<string, string> =>
  Object.fromEntries(
    text.split(",").map((p) => p.trim()).filter(Boolean)
      .map((p) => p.split("=", 2)).filter(([v, e]) => v?.trim() && e?.trim())
      .map(([v, e]) => [v.trim(), e.trim()]),
  );
```

```tsx
{Object.entries(templates).map(([key, tpl]) => (
  <div key={key}>
    <input value={key} onChange={(e) => renameTemplate(key, e.target.value)} />
    <input
      value={tpl.name}
      placeholder={t("settings.sandbox.templateName")}
      onChange={(e) => setTemplate(key, { name: e.target.value })}
    />
    <label>
      <input
        type="checkbox"
        checked={!!tpl.code_writable}
        onChange={(e) => setTemplate(key, { code_writable: e.target.checked })}
      />
      {t("settings.sandbox.templateWritable")}
    </label>
    <input
      value={envRefsToText(tpl.env_refs)}
      placeholder={t("settings.sandbox.templateEnvRefs")}
      onChange={(e) => setTemplate(key, { env_refs: envRefsFromText(e.target.value) })}
    />
    <button type="button" onClick={() => removeTemplate(key)}>×</button>
  </div>
))}
<button type="button" onClick={addTemplate}>+</button>

<label>{t("settings.sandbox.defaultTemplate")}</label>
<select value={defaultTemplate} onChange={(e) => setDefaultTemplate(e.target.value)}>
  <option value="">—</option>
  {Object.keys(templates).map((key) => (
    <option key={key} value={key}>{key}</option>
  ))}
</select>
```

`code_writable`'s help text describes what the **image** did (turbox chowns `/opt/code` to uid 1000) — the copy must not read as a per-agent permission, or an admin will tick it on a pai-rec template and hand that agent a prompt promising checkouts against a root-owned directory.

- [ ] **Step 5: Add the agent template picker**

In the agent section (line ~1083, beside the `agent.code.enabled` checkbox), add a select bound to `agent.sandbox.template`, with options from the provider's `templates` keys plus a blank "use default" option:

```tsx
<select
  value={agent.sandbox?.template ?? ""}
  onChange={(e) => onChange({
    ...agent,
    sandbox: { ...agent.sandbox, template: e.target.value },
  })}
>
  <option value="">{t("settings.agent.sandboxTemplateDefault")}</option>
  {Object.keys(sandboxTemplates).map((key) => (
    <option key={key} value={key}>{key}</option>
  ))}
</select>
```

Place it beside the code config — the two are coupled: the manifest describes the repositories the bound template's image actually ships.

- [ ] **Step 6: Add the i18n keys**

In `frontend/src/i18n/en.ts`:

```ts
  "settings.sandbox.templates": "Sandbox templates",
  "settings.sandbox.templatesHelp": "Each key names an AgentRun template. Agents bind one by key.",
  "settings.sandbox.templateName": "Template name in AgentRun",
  "settings.sandbox.templateWritable": "Code layer is writable",
  "settings.sandbox.templateWritableHelp": "Set this when the image's /opt/code is owned by the runtime user (git working copies the agent can check out). Must match the image.",
  "settings.sandbox.templateEnvRefs": "Env refs (VAR=SOURCE_ENV_VAR)",
  "settings.sandbox.defaultTemplate": "Default template",
  "settings.agent.sandboxTemplate": "Sandbox template",
  "settings.agent.sandboxTemplateDefault": "Use default",
```

In `frontend/src/i18n/zh.ts`:

```ts
  "settings.sandbox.templates": "沙箱模板",
  "settings.sandbox.templatesHelp": "每个 key 对应一个 AgentRun 模板，Agent 按 key 绑定。",
  "settings.sandbox.templateName": "AgentRun 中的模板名",
  "settings.sandbox.templateWritable": "代码层可写",
  "settings.sandbox.templateWritableHelp": "当镜像的 /opt/code 归运行时用户所有（agent 可 checkout 的 git 工作区）时勾选。必须与镜像一致。",
  "settings.sandbox.templateEnvRefs": "环境变量透传（VAR=来源环境变量名）",
  "settings.sandbox.defaultTemplate": "默认模板",
  "settings.agent.sandboxTemplate": "沙箱模板",
  "settings.agent.sandboxTemplateDefault": "使用默认",
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/__tests__/SettingsView.test.tsx`
Expected: PASS

- [ ] **Step 8: Typecheck and run the full frontend suite**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add frontend/src
git commit -m "feat(ui): edit sandbox templates and bind one per agent"
```

---

### Task 7: Docs

**Files:**
- Rewrite: `sandbox/README.md`
- Modify: `docs/design/skill_install_mount_dependencies.md:69-71, 182-225`
- Modify: `backend/README.md:36`

- [ ] **Step 1: Rewrite `sandbox/README.md`**

Cover: the base + two templates layout; `build.sh <template>` (base first, and why a clean CI machine needs both steps); `BASE_IMAGE` and `GITLAB_TOKEN`; registering **two** templates in AgentRun (pai-rec public, turbo-x VPC, same account + region); wiring `templates` / `default_template` / `code_writable` / `env_refs` in `data/config.yaml`; and `POST /v1/config/reload-env` after `.env` changes.

Reword the `/opt/*` vs `/mnt/*` section — the rule stands, its justification changes. It currently says "`/opt/*` = baked into the image (**read-only** code layer)". Read-only no longer holds for turbox, but the convention was always about **backing store**:

```markdown
- **`/opt/*` = image layer (local disk).** Currently just `/opt/code` (the code
  layer). Baked rather than mounted because local disk beats NFS for the
  browse/grep workload; content is fixed at image-build time. Whether it is
  writable is per-template: pai-rec bakes a root-owned snapshot, turbox clones
  git working copies owned by the runtime user so the agent can check out
  branches. Writes land in the container's per-instance layer and are discarded
  with the sandbox.
- **`/mnt/*` = NAS mounts, attached per-sandbox at create time** via `nasConfig`.
  These are the three contract paths: `/mnt/system`, `/mnt/skills` (read-only),
  and `/mnt/user` (read-write, per-user). Content is dynamic and lives on the
  NAS, not in the image; `agent-sandbox-bootstrap` validates all three exist on
  start (contract v1).

Do **not** move a NAS mount (e.g. skills) under `/opt`, or a baked layer under
`/mnt` — that mixes backing stores within one prefix and breaks the mount
contract the bootstrap enforces.
```

Keep the contract-version section as-is: the bootstrap and `CONTRACT_VERSION` are unchanged by this work.

- [ ] **Step 2: Update the mount-dependencies design doc**

In `docs/design/skill_install_mount_dependencies.md`, replace the `code_layer_enabled` config sample (lines 69-71) and the `/opt/code` prose (lines 182-225) to describe: per-agent `sandbox.template` binding, `templates[key].code_writable` driving prompt wording, and `/opt/code` being read-only for pai-rec / writable working copies for turbox. Keep `AGENT_CODE_PATH=/opt/code` as the documented default root.

- [ ] **Step 3: Update the backend README**

`backend/README.md:36` currently reads "`sandbox.default.settings` supports AgentRun `template_name`, user/tenant/conversation …". Replace `template_name` with `templates` / `default_template` and note that Agents bind a key via `agents[].sandbox.template`.

- [ ] **Step 4: Verify no stale references remain**

Run: `rg -n "template_name|code_layer_enabled|PAIREC_CODE_ARCHIVE_URL" . --glob '!docs/superpowers/**' --glob '!*.lock'`
Expected: no matches. (Historical plans and specs under `docs/superpowers/` legitimately keep the old names.)

- [ ] **Step 5: Commit**

```bash
git add sandbox/README.md docs/design/skill_install_mount_dependencies.md backend/README.md
git commit -m "docs: sandbox templates, per-agent binding, and the /opt backing-store rule"
```

---

## Final verification

- [ ] `cd backend && python -m pytest` — PASS
- [ ] `cd frontend && npx tsc --noEmit && npx vitest run` — PASS
- [ ] `rg -n "template_name" . --glob '!docs/superpowers/**'` — no matches
- [ ] `sandbox/build.sh pairec` and `sandbox/build.sh turbox` both build; the turbox checks in Task 5 Steps 8-9 pass (`/opt/code` owned by 1000, branches list offline, no token in any layer)
