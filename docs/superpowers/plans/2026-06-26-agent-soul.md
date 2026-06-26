# Agent SOUL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the lean agent a configurable identity and a professional, layered system prompt via a `Soul` model and a pure prompt-composition function, wired into `build_context`.

**Architecture:** A new `agent/soul.py` defines `Soul` (persona layers) + `DEFAULT_SOUL` + the pure `render_system_prompt(soul, *, tool_names, extra="")`. `build_context` composes the effective soul (default ← request `soul` override ← `instructions`→`extra_instructions`) and renders it into `AgentContext.system_prompt`. `AppState` carries the default soul; `ResponsesRequest` carries a per-request `soul` override. Tools are not wired yet (this plan renders with `tool_names=[]`); that is Plan B.

**Tech Stack:** Python 3.11, pydantic v2, FastAPI, pytest. Tests run from `backend/`: `cd backend && python -m pytest ../tests/app -q`. Async tests use the repo pattern (`def test_*` calling `asyncio.run(run())`).

**Reference spec:** `docs/superpowers/specs/2026-06-26-agent-soul-and-tools-design.md`. Branch: `personal/yfei/agent-core`. The tools/registry/extensibility plan (`2026-06-26-agent-tools.md`) is SEPARATE and depends on this one.

## Global Constraints

- **Import-lean:** `agent/soul.py` imports only stdlib + pydantic. The lean service boot/import-isolation tests stay green.
- **Pure renderer:** `render_system_prompt` does no I/O and reads no clock — the time header stays a per-turn concern of `build_messages`. This keeps it string-testable.
- **Route-compatible signature:** `build_context` gains a keyword-only `soul: Soul = DEFAULT_SOUL` with a default, so adding `registry` in Plan B is additive and the route keeps working between plans.
- **`instructions` semantics change** (deliberate): `request.instructions` becomes `extra_instructions` appended to the composed prompt, NOT the whole system prompt. The one existing builder test asserting equality is updated to assert containment.
- Run the full app suite at the end of every task: `cd backend && python -m pytest ../tests/app -q`.

---

## Key existing contracts (verified, do not re-derive)

- **`agent/context.py`** — `@dataclass AgentContext(system_prompt: str, history, current_turn, attachments, hints, tools, run_vars)`; `RunVars(current_datetime=<factory>)`.
- **`agent/agent.py`** — `build_messages` does `Message("system", ctx.system_prompt)` then the current turn with a `[System Time: …]` prefix. (No change in this plan.)
- **`app/builder.py`** — current `build_context(request, store) -> Tuple[AgentContext, Optional[str]]`: sets `system_prompt = request.instructions or DEFAULT_SYSTEM_PROMPT`, `tools=ToolBox([])`, mints/resolves `conversation_id`. Helpers `items_to_messages`, `_input_to_turn`, `_item_text` (keep). `DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."` (removed by this plan).
- **`app/schemas.py`** — `ResponsesRequest(BaseModel, extra="ignore")` with `model, input, instructions, previous_response_id, conversation, user_id, store, stream, metadata, tools`.
- **`app/deps.py`** — `@dataclass AppState(store, llm, default_model, context_window=110000, max_output_tokens=8000)` + `make_agent()` + `get_state(request)`.
- **`app/routes/responses.py`** — calls `ctx, conversation_id = await build_context(request, state.store)`.
- **`app/config.py`** — `Settings(BaseSettings)` with `openai_base_url, openai_api_key, default_model, db_url, store_backend`; `get_settings()`.
- **`app/lean_main.py`** — builds `AppState(store=, llm=, default_model=)` in lifespan.
- **Existing builder tests** (`tests/app/test_builder.py`): `test_build_context_from_string_input` asserts `ctx.system_prompt == "be terse"` for `instructions="be terse"` — UPDATED here. Other builder tests don't assert `system_prompt`.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/agent/soul.py` (new) | `Soul` model, `DEFAULT_SOUL`, `Soul.merge`, pure `render_system_prompt` |
| `backend/app/schemas.py` (modify) | `ResponsesRequest.soul: Optional[dict]` |
| `backend/app/builder.py` (modify) | compose effective soul → `system_prompt`; `soul` kw param |
| `backend/app/deps.py` (modify) | `AppState.soul: Soul` |
| `backend/app/config.py` (modify) | `agent_name`, `agent_role` settings (optional default-soul overrides) |
| `backend/app/routes/responses.py` (modify) | pass `soul=state.soul` to `build_context` |
| `backend/app/lean_main.py` (modify) | build `AppState.soul` from settings |
| `tests/app/test_soul.py` (new) | `Soul`/`merge`/`render_system_prompt` |
| `tests/app/test_builder.py` (modify) | soul composition assertions |

---

## Task 1: `Soul` model + `render_system_prompt`

The persona model + the pure composition function — the testable core of this plan.

**Files:**
- Create: `backend/agent/soul.py`
- Test: `tests/app/test_soul.py`

**Interfaces:**
- Produces:
  - `class Soul(BaseModel)` with fields `name, role, identity, personality: list[str], principles: list[str], expertise: list[str], style, constraints: list[str], extra_instructions: str, tools_enabled: Optional[list[str]]` and defaults forming a professional general assistant.
  - `DEFAULT_SOUL: Soul`.
  - `Soul.merge(self, override: dict) -> Soul` — returns a copy with known, non-None override fields replaced.
  - `render_system_prompt(soul: Soul, *, tool_names: list[str], extra: str = "") -> str` — pure; sections Identity / Personality / Operating principles / Tools / Safety / Additional instructions.

- [ ] **Step 1: Write the failing test** — `tests/app/test_soul.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.soul import Soul, DEFAULT_SOUL, render_system_prompt


def test_default_soul_has_identity_and_name():
    assert DEFAULT_SOUL.name
    assert DEFAULT_SOUL.role
    assert DEFAULT_SOUL.identity
    assert DEFAULT_SOUL.personality and DEFAULT_SOUL.principles
    assert DEFAULT_SOUL.tools_enabled is None


def test_merge_replaces_known_fields_and_ignores_none_and_unknown():
    merged = DEFAULT_SOUL.merge({"name": "Helper", "role": None, "bogus": "x"})
    assert merged.name == "Helper"
    assert merged.role == DEFAULT_SOUL.role  # None override ignored
    assert not hasattr(merged, "bogus")
    assert DEFAULT_SOUL.name != "Helper"  # original unchanged


def test_render_includes_identity_personality_principles_safety():
    out = render_system_prompt(DEFAULT_SOUL, tool_names=[])
    assert DEFAULT_SOUL.name in out
    assert DEFAULT_SOUL.role in out
    assert "# Identity" in out
    assert "# Personality" in out
    assert "# Operating principles" in out
    assert "# Safety" in out
    # the first principle text shows up
    assert DEFAULT_SOUL.principles[0] in out


def test_render_lists_tools_when_present_and_says_none_when_empty():
    none_out = render_system_prompt(DEFAULT_SOUL, tool_names=[])
    assert "no tools" in none_out.lower()
    tools_out = render_system_prompt(DEFAULT_SOUL, tool_names=["web_search", "web_fetch"])
    assert "web_search" in tools_out and "web_fetch" in tools_out
    assert "# Tools" in tools_out


def test_render_includes_extra_instructions_only_when_present():
    soul = DEFAULT_SOUL.merge({"extra_instructions": "Always answer in French."})
    out = render_system_prompt(soul, tool_names=[])
    assert "Always answer in French." in out
    assert "# Additional instructions" in out
    blank = render_system_prompt(DEFAULT_SOUL, tool_names=[])
    assert "# Additional instructions" not in blank


def test_render_includes_expertise_when_set():
    soul = DEFAULT_SOUL.merge({"expertise": ["tax law", "accounting"]})
    out = render_system_prompt(soul, tool_names=[])
    assert "tax law" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_soul.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.soul'`.

- [ ] **Step 3: Implement `backend/agent/soul.py`**

```python
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Soul(BaseModel):
    """The configurable persona of an agent: who it is and how it behaves.

    Kept separate from the stable "engine" prompt (tool protocol + safety),
    which `render_system_prompt` adds. Every field is data, so a custom agent is
    a Soul override — no code change.
    """

    name: str = "Aria"
    role: str = "a general-purpose AI assistant"
    identity: str = (
        "You are a capable, trustworthy assistant that helps people think, find "
        "information, and get work done. You do the work rather than describe it, "
        "and you tell the user plainly what you did, what you found, and what is "
        "still uncertain."
    )
    personality: List[str] = [
        "Warm but concise — you respect the user's time.",
        "Curious and precise — you verify rather than guess.",
        "Calm under ambiguity — you state your assumptions and proceed.",
    ]
    principles: List[str] = [
        "Act on what you can determine; ask only when you are genuinely blocked.",
        "Ground factual claims in evidence; when you are unsure, say so plainly.",
        "Prefer the simplest answer that fully addresses the request.",
        "Surface key tradeoffs and give a recommendation, not an exhaustive menu.",
        "Report outcomes faithfully, including failures, gaps, and assumptions.",
    ]
    expertise: List[str] = []
    style: str = (
        "Write in clear, well-structured Markdown. Lead with the answer, then "
        "support it. Use lists and code blocks where they aid scanning. Avoid "
        "filler, hedging, and unnecessary preamble."
    )
    constraints: List[str] = [
        "Decline requests to cause harm or break the law.",
        "Never fabricate facts, sources, quotes, or tool output.",
        "Respect privacy; do not invent personal data.",
    ]
    extra_instructions: str = ""
    tools_enabled: Optional[List[str]] = None  # None = all registered tools

    def merge(self, override: dict) -> "Soul":
        """Return a copy with known, non-None override fields replaced.
        Lists are replaced wholesale (not concatenated)."""
        valid = {
            k: v
            for k, v in (override or {}).items()
            if k in type(self).model_fields and v is not None
        }
        return self.model_copy(update=valid)


DEFAULT_SOUL = Soul()


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {it}" for it in items)


# The stable "engine" layer: capabilities/tool protocol + safety. Persona-agnostic.
_TOOL_PROTOCOL = (
    "When a tool would materially help, call it with well-formed arguments. "
    "Never invent tool output or claim you used a tool you did not. Ground "
    "factual and time-sensitive answers in tool results. Take one logical "
    "action at a time, and stop once the request is satisfied."
)


def render_system_prompt(
    soul: Soul, *, tool_names: List[str], extra: str = ""
) -> str:
    """Compose the persona (soul) + the engine layer into a system prompt.
    Pure: no I/O, no clock (the time header is added per-turn elsewhere)."""
    parts: List[str] = []

    identity = f"# Identity\nYou are {soul.name}, {soul.role}.\n\n{soul.identity}"
    if soul.expertise:
        identity += "\n\nYour areas of expertise: " + ", ".join(soul.expertise) + "."
    parts.append(identity)

    personality = "# Personality\n" + _bullets(soul.personality)
    if soul.style:
        personality += "\n\n" + soul.style
    parts.append(personality)

    parts.append("# Operating principles\n" + _bullets(soul.principles))

    tools_section = "# Tools\n" + _TOOL_PROTOCOL
    if tool_names:
        tools_section += "\n\nTools available this session: " + ", ".join(tool_names) + "."
    else:
        tools_section += "\n\nYou have no tools enabled in this session; answer from your own knowledge."
    parts.append(tools_section)

    if soul.constraints:
        parts.append("# Safety\n" + _bullets(soul.constraints))

    tail = "\n\n".join(p for p in (soul.extra_instructions.strip(), extra.strip()) if p)
    if tail:
        parts.append("# Additional instructions\n" + tail)

    return "\n\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_soul.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/agent/soul.py tests/app/test_soul.py
git commit -m "feat(agent): Soul persona model + pure render_system_prompt"
```

---

## Task 2: Compose the soul in `build_context` + wiring

Use the soul to build `system_prompt`, accept a per-request override, and thread the default soul through `AppState`/route/settings.

**Files:**
- Modify: `backend/app/schemas.py`, `backend/app/builder.py`, `backend/app/deps.py`, `backend/app/config.py`, `backend/app/routes/responses.py`, `backend/app/lean_main.py`
- Test: `tests/app/test_builder.py` (update + add)

**Interfaces:**
- Consumes: `Soul`, `DEFAULT_SOUL`, `render_system_prompt` (Task 1).
- Produces:
  - `ResponsesRequest.soul: Optional[Dict[str, Any]] = None`.
  - `build_context(request, store, *, soul: Soul = DEFAULT_SOUL) -> Tuple[AgentContext, Optional[str]]` — composes the effective soul and renders `system_prompt` (with `tool_names=[]` for now).
  - `AppState.soul: Soul` (default `DEFAULT_SOUL`).
  - `Settings.agent_name`, `Settings.agent_role`.

- [ ] **Step 1: Write the failing test** — update `tests/app/test_builder.py`

Replace the body of `test_build_context_from_string_input` so it no longer asserts equality with the instruction:

```python
def test_build_context_from_string_input():
    async def run():
        st = InMemoryStore()
        req = ResponsesRequest(
            model="m", input="hi there", instructions="be terse"
        )
        ctx, conv_id = await build_context(req, st)
        assert ctx.current_turn.role == "user"
        assert ctx.current_turn.content == "hi there"
        # instructions now compose into the system prompt (not replace it)
        assert "be terse" in ctx.system_prompt
        assert "# Additional instructions" in ctx.system_prompt
        assert "# Identity" in ctx.system_prompt  # the soul is rendered
        assert ctx.history == []
        assert conv_id is not None

    asyncio.run(run())
```

Append new tests:

```python
def test_build_context_renders_default_soul_into_system_prompt():
    async def run():
        from agent.soul import DEFAULT_SOUL
        st = InMemoryStore()
        req = ResponsesRequest(model="m", input="hi")
        ctx, _ = await build_context(req, st)
        assert DEFAULT_SOUL.name in ctx.system_prompt
        assert "# Operating principles" in ctx.system_prompt
        # no tools wired in this plan
        assert "no tools" in ctx.system_prompt.lower()

    asyncio.run(run())


def test_build_context_applies_request_soul_override():
    async def run():
        st = InMemoryStore()
        req = ResponsesRequest(
            model="m", input="hi", soul={"name": "Lex", "role": "a legal analyst"}
        )
        ctx, _ = await build_context(req, st)
        assert "You are Lex, a legal analyst." in ctx.system_prompt

    asyncio.run(run())


def test_build_context_accepts_explicit_soul_argument():
    async def run():
        from agent.soul import Soul
        st = InMemoryStore()
        req = ResponsesRequest(model="m", input="hi")
        ctx, _ = await build_context(req, st, soul=Soul(name="Custom", role="a tutor"))
        assert "You are Custom, a tutor." in ctx.system_prompt

    asyncio.run(run())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_builder.py -q`
Expected: FAIL — `ResponsesRequest` has no `soul` / `build_context` has no `soul` kwarg / `system_prompt` lacks `# Identity`.

- [ ] **Step 3: Add `soul` to `ResponsesRequest`** in `backend/app/schemas.py`

Add after `tools`:

```python
    soul: Optional[Dict[str, Any]] = None
```

(`Dict`/`Any`/`Optional` are already imported.)

- [ ] **Step 4: Compose the soul in `backend/app/builder.py`**

Replace the imports + `DEFAULT_SYSTEM_PROMPT` + the `build_context` body:

Replace:
```python
from app.store.base import Item, new_conversation_id

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
```
with:
```python
from app.store.base import Item, new_conversation_id
from agent.soul import Soul, DEFAULT_SOUL, render_system_prompt
```

Change the `build_context` signature + the `AgentContext` assembly:

```python
async def build_context(
    request: ResponsesRequest, store, *, soul: Soul = DEFAULT_SOUL
) -> Tuple[AgentContext, Optional[str]]:
    """Resolve prior history via the store and assemble the AgentContext.
    Composes the effective soul (default <- request.soul <- instructions) into a
    layered system prompt. Tools are wired in a later plan (tool_names=[] here).
    Raises ValueError on previous_response_id/conversation conflict (-> HTTP 400)."""
    history_items: List[Item] = []
    conversation_id = request.conversation
    if request.previous_response_id or request.conversation:
        history_items = await store.resolve_history(
            previous_response_id=request.previous_response_id,
            conversation=request.conversation,
        )
        if request.previous_response_id:
            resp = await store.get_response(request.previous_response_id)
            if resp is not None and resp.conversation_id:
                conversation_id = resp.conversation_id

    if conversation_id is None:
        conversation_id = new_conversation_id()

    override = dict(request.soul or {})
    if request.instructions:
        override["extra_instructions"] = request.instructions
    effective_soul = soul.merge(override)
    tool_names: List[str] = []  # populated from the tool registry in a later plan
    system_prompt = render_system_prompt(effective_soul, tool_names=tool_names)

    ctx = AgentContext(
        system_prompt=system_prompt,
        history=items_to_messages(history_items),
        current_turn=_input_to_turn(request.input),
        attachments=[],
        hints=[],
        tools=ToolBox([]),
        run_vars=RunVars(),
    )
    return ctx, conversation_id
```

- [ ] **Step 5: Add `AppState.soul`** in `backend/app/deps.py`

Add imports:

```python
from dataclasses import dataclass, field
from agent.soul import Soul, DEFAULT_SOUL
```

Add the field (after `max_output_tokens`):

```python
    soul: Soul = field(default_factory=lambda: DEFAULT_SOUL)
```

- [ ] **Step 6: Add settings** in `backend/app/config.py`

Add to `Settings` (after `store_backend`):

```python
    agent_name: str = "Aria"
    agent_role: str = "a general-purpose AI assistant"
```

- [ ] **Step 7: Pass the soul through the route** in `backend/app/routes/responses.py`

Change the `build_context` call in `create_response`:

```python
        ctx, conversation_id = await build_context(request, state.store, soul=state.soul)
```

- [ ] **Step 8: Build the default soul from settings** in `backend/app/lean_main.py`

Add the import:

```python
from agent.soul import Soul
```

In `lifespan`, build the soul and pass it to `AppState`:

```python
    soul = Soul(name=settings.agent_name, role=settings.agent_role)
    app.state.app_state = AppState(
        store=store, llm=_build_llm(settings), default_model=settings.default_model,
        soul=soul,
    )
```

- [ ] **Step 9: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_builder.py ../tests/app/test_soul.py -q`
Expected: PASS.

- [ ] **Step 10: Run the full app suite + boot/isolation gates**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (existing route tests still pass — the agent still gets a valid system prompt; `test_routes_responses.py` echo tests assert `startswith("echo:")`, unaffected).

- [ ] **Step 11: Commit**

```bash
git add backend/app/schemas.py backend/app/builder.py backend/app/deps.py backend/app/config.py backend/app/routes/responses.py backend/app/lean_main.py tests/app/test_builder.py
git commit -m "feat(app): compose configurable Soul into the system prompt (build_context + wiring)"
```

---

## Self-Review (completed against the spec)

- **`Soul` model with persona layers + `DEFAULT_SOUL` + `merge`** → Task 1.
- **Pure `render_system_prompt` (Identity / Personality / Operating principles / Tools / Safety / Additional instructions); lists tools or says none; engine tool-protocol + safety layers** → Task 1.
- **`build_context` composes default ← `request.soul` ← `instructions`→`extra_instructions`; renders `system_prompt`; `tool_names=[]` (tools are Plan B)** → Task 2.
- **`ResponsesRequest.soul`, `AppState.soul`, settings `agent_name`/`agent_role`, route + lean_main wiring** → Task 2.
- **`instructions` semantics change + the one existing builder test updated** → Task 2 Step 1.
- **Import-lean + boot/isolation green; route-compatible signature (keyword `soul` default) so Plan B's `registry` add is additive** → Global Constraints + Task 2.

No placeholders; signatures (`Soul.merge(override)`, `render_system_prompt(soul, *, tool_names, extra="")`, `build_context(request, store, *, soul=DEFAULT_SOUL)`) are consistent across tasks and match Plan B's expectations (Plan B passes real `tool_names` and adds a `registry` kwarg).
