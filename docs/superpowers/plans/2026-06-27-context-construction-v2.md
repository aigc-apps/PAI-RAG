# Context Construction v2 Implementation Plan (rolling summary + cacheable layered prompt)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (③) Split the prompt into a stable, cacheable system layer + a volatile context block (user memory + rolling summary + per-request instructions) and add a static project-context layer; (①) maintain a persisted, rolling per-conversation summary that bridges old turns instead of losing them. (② relevance retrieval is out of scope — user memory stays injected within its cap.)

**Architecture:** `render_stable_system_prompt` (cacheable) + `render_context_block` (volatile) replace the single `render_system_prompt`; `AgentContext.context_block` carries the volatile layer and `build_messages` emits `[stable] + history + [volatile] + user` (cacheable prefix). `Conversation` gains `summary`/`summarized_seq`; an async, gated, guarded post-turn `maybe_summarize_conversation` folds old items into the rolling summary; `build_context` serves the summary + only items after `summarized_seq`.

**Tech Stack:** Python 3.11, pydantic/SQLModel, FastAPI + TestClient, pytest. Backend tests from `backend/`: `cd backend && python -m pytest ../tests/app -q`.

**Reference spec:** `docs/superpowers/specs/2026-06-27-context-construction-v2-design.md`. Branch: `personal/yfei/agent-core`.

## Global Constraints

- **Import-lean; both stores in lockstep; gated + async memory/summary never break a turn.** Existing backend tests (155) stay green (with the prompt-split test updates in Task 1).
- **Cacheable ordering:** stable system first, history next (append-only), volatile context as a trailing `system` message right before the user turn. Do not put volatile content in the front system message.
- **Rolling summary is primary compaction; `fit_to_budget` (L1–L4) stays the in-call fallback (unchanged).**
- Run the full app suite at the end of each task.

---

## Key existing contracts (verified)

- **`agent/soul.py`** — `render_system_prompt(soul, *, tool_names, memories=None, extra="")` with `_bullets`, `_TOOL_PROTOCOL`, sections Identity/Personality/Operating principles/Memory/Tools/Safety/Additional instructions. `Soul` has `extra_instructions`.
- **`agent/context.py`** — `@dataclass AgentContext(system_prompt, history, current_turn, attachments, hints, tools, run_vars)`.
- **`agent/agent.py`** — `build_messages(ctx)` returns `[Message("system", ctx.system_prompt)] + ctx.history + [render_current_turn(...)]`. `run` loops `messages = self.budget.fit(messages)` per step.
- **`app/builder.py`** — `build_context(request, store, *, soul=DEFAULT_SOUL, registry=None)`: `effective_soul = soul.merge({**request.soul, "extra_instructions": request.instructions})`; computes `tool_names`/`toolbox`; `memories = [m.text for m in await store.list_memories(uid, limit=MEMORY_INJECT_LIMIT)] if uid else []`; `system_prompt = render_system_prompt(effective_soul, tool_names=tool_names, memories=memories)`. `MEMORY_INJECT_LIMIT=30`.
- **`app/store/base.py`** — `Conversation(id, user_id, title, last_response_id, created_at, updated_at)`; `Item(...seq...)`; store conversation methods (`get_conversation`, `resolve_history`, `get_conversation_items`, etc.).
- **`app/models.py`** — `Conversation` table; `_now()`.
- **`app/store/memory.py` / `sql.py`** — `InMemoryStore` `_convs`; `SqlStore` `_to_conv`, `ConvRow`.
- **`app/routes/responses.py`** — `_persist`, `_schedule_memory_update` (async post-turn, gated by `state.memory_enabled`); called after `_persist` in sync/stream/background paths.
- **`app/memory.py`** — `MemoryExtractor`, `make_complete(llm)` (single-shot over `astream`). Reuse `make_complete` for the summarizer.
- **`app/config.py`** — `Settings`; `app/deps.py` `AppState(..., memory_enabled, memory_model)`; `app/lean_main.py` builds AppState.
- **Affected existing tests:** `tests/app/test_soul.py` (render_system_prompt), `tests/app/test_builder.py` (system_prompt assertions), `tests/app/test_memory_inject.py` (memory in system_prompt). These are UPDATED in Task 1.

---

## Task 1: Split the prompt — stable system + volatile context block (+ project context)

**Files:** Modify `agent/soul.py`, `agent/context.py`, `agent/agent.py`, `app/builder.py`, `app/config.py`. Update tests `tests/app/test_soul.py`, `tests/app/test_builder.py`, `tests/app/test_memory_inject.py`.

**Interfaces:**
- Produces: `render_stable_system_prompt(soul, *, tool_names, project_context="") -> str` (Identity/Personality/Operating principles/Project/Tools/Safety; NO memory/instructions/summary); `render_context_block(*, memories=None, summary="", instructions="") -> str` (Memory/Conversation summary/Additional instructions; "" when empty); `AgentContext.context_block: str = ""`; `build_messages` emits `[stable] + history + [volatile?] + user`; `Settings.project_context: str = ""`.

- [ ] **Step 1: Write/ъ update the failing tests**

Replace the render tests in `tests/app/test_soul.py` (keep the `Soul`/`merge` tests) with:

```python
def test_stable_prompt_has_persona_tools_safety_not_memory_or_instructions():
    out = render_stable_system_prompt(DEFAULT_SOUL, tool_names=["web_fetch"])
    assert DEFAULT_SOUL.name in out and "# Identity" in out
    assert "# Personality" in out and "# Operating principles" in out
    assert "# Safety" in out and "web_fetch" in out and "# Tools" in out
    assert "# Memory" not in out and "# Additional instructions" not in out


def test_stable_prompt_lists_no_tools_when_empty_and_project_when_set():
    assert "no tools" in render_stable_system_prompt(DEFAULT_SOUL, tool_names=[]).lower()
    out = render_stable_system_prompt(DEFAULT_SOUL, tool_names=[], project_context="Repo: PAI-RAG")
    assert "# Project context" in out and "Repo: PAI-RAG" in out
    assert "# Project context" not in render_stable_system_prompt(DEFAULT_SOUL, tool_names=[])


def test_context_block_renders_memory_summary_instructions_and_empty():
    assert render_context_block() == ""
    out = render_context_block(memories=["likes tea"], summary="talked about X",
                               instructions="be terse")
    assert "# Memory" in out and "likes tea" in out
    assert "# Conversation summary" in out and "talked about X" in out
    assert "# Additional instructions" in out and "be terse" in out
```

Update `tests/app/test_soul.py` imports: `from agent.soul import Soul, DEFAULT_SOUL, render_stable_system_prompt, render_context_block`.

In `tests/app/test_builder.py`, replace `test_build_context_from_string_input` body's prompt assertions:

```python
        # instructions now live in the volatile context block, not the stable system prompt
        assert "be terse" in ctx.context_block
        assert "# Additional instructions" in ctx.context_block
        assert "be terse" not in ctx.system_prompt
        assert "# Identity" in ctx.system_prompt
```

In `tests/app/test_memory_inject.py`, change the two tests to assert on `render_context_block` / `ctx.context_block`:

```python
def test_context_block_includes_memory_when_present():
    out = render_context_block(memories=["likes tea", "in NYC"])
    assert "# Memory" in out and "likes tea" in out and "in NYC" in out
    assert render_context_block(memories=[]) == ""


def test_build_context_injects_user_memories_into_context_block():
    async def run():
        st = InMemoryStore()
        await st.add_memory(MemoryItem(user_id="u1", text="prefers Python"))
        ctx, _ = await build_context(ResponsesRequest(model="m", input="hi", user="u1"), st)
        assert "prefers Python" in ctx.context_block
        ctx2, _ = await build_context(ResponsesRequest(model="m", input="hi"), st)
        assert "# Memory" not in ctx2.context_block
    asyncio.run(run())
```

Update that file's imports: `from agent.soul import render_context_block`.

- [ ] **Step 2: Run to verify failure** — `cd backend && python -m pytest ../tests/app/test_soul.py ../tests/app/test_builder.py ../tests/app/test_memory_inject.py -q` → FAIL (functions/fields missing).

- [ ] **Step 3: Add the two render functions** in `agent/soul.py`

Add (you may keep `render_system_prompt` or remove it; this plan REMOVES it and replaces its only caller in Task 1 Step 6). Add:

```python
def render_stable_system_prompt(
    soul: Soul, *, tool_names: List[str], project_context: str = ""
) -> str:
    """Stable, cacheable layer: persona + project + tool protocol + safety.
    Excludes volatile content (memory, per-request instructions, conversation summary)."""
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
    if project_context.strip():
        parts.append("# Project context\n" + project_context.strip())
    tools_section = "# Tools\n" + _TOOL_PROTOCOL
    if tool_names:
        tools_section += "\n\nTools available this session: " + ", ".join(tool_names) + "."
    else:
        tools_section += "\n\nYou have no tools enabled in this session; answer from your own knowledge."
    parts.append(tools_section)
    if soul.constraints:
        parts.append("# Safety\n" + _bullets(soul.constraints))
    return "\n\n".join(parts)


def render_context_block(
    *, memories: Optional[List[str]] = None, summary: str = "", instructions: str = ""
) -> str:
    """Volatile per-turn context: user memory + rolling conversation summary +
    per-request instructions. Returns '' when all are empty."""
    parts: List[str] = []
    if memories:
        parts.append(
            "# Memory\nWhat you remember about this user (use it naturally; "
            "do not recite it verbatim):\n" + _bullets(memories)
        )
    if summary.strip():
        parts.append(
            "# Conversation summary\nSummary of earlier turns in this conversation:\n"
            + summary.strip()
        )
    if instructions.strip():
        parts.append("# Additional instructions\n" + instructions.strip())
    return "\n\n".join(parts)
```

Remove `render_system_prompt` (and its now-removed tests are replaced in Step 1).

- [ ] **Step 4: Add `context_block`** in `agent/context.py`

```python
    context_block: str = ""
```
(after `system_prompt`; it has a default so existing constructions are unaffected.)

- [ ] **Step 5: Emit the volatile message** in `agent/agent.py` `build_messages`

```python
    @staticmethod
    def build_messages(ctx: AgentContext) -> List[Message]:
        msgs: List[Message] = [Message("system", ctx.system_prompt)]
        msgs += ctx.history
        block = getattr(ctx, "context_block", "")
        if block:
            msgs.append(Message("system", block))
        msgs.append(render_current_turn(ctx.current_turn, ctx.attachments, ctx.hints, ctx.run_vars))
        logger.info("[agent] model input: %d msgs", len(msgs))
        return msgs
```

- [ ] **Step 6: Wire `build_context`** in `app/builder.py`

Change the import:
```python
from agent.soul import Soul, DEFAULT_SOUL, render_stable_system_prompt, render_context_block
```
Replace the soul/prompt section (the `override`/`effective_soul`/`system_prompt` part) with:
```python
    override = dict(request.soul or {})
    effective_soul = soul.merge(override)
    # tools (unchanged) ...
    tool_names = [t.name for t in toolbox.tools]

    project_context = getattr(request, "_project_context", "") or ""  # default; AppState wires the real one in Task — see note
    system_prompt = render_stable_system_prompt(
        effective_soul, tool_names=tool_names, project_context=project_context
    )

    memories: List[str] = []
    uid = request.resolved_user_id
    if uid:
        memories = [m.text for m in await store.list_memories(uid, limit=MEMORY_INJECT_LIMIT)]
    instructions = "\n\n".join(s for s in [
        (effective_soul.extra_instructions or "").strip(), (request.instructions or "").strip()
    ] if s)
    context_block = render_context_block(memories=memories, instructions=instructions)
```
and pass `context_block=context_block` into the `AgentContext(...)` constructor.

NOTE: keep `project_context` simple for this task — pass `""` (the project-context source is wired via a `project_context` keyword in this same task; see Step 6b). Do NOT keep the old `override["extra_instructions"] = request.instructions` line (instructions now flow via the volatile block).

- [ ] **Step 6b: Thread `project_context` cleanly**

Give `build_context` a keyword param instead of the `getattr` placeholder:
```python
async def build_context(request, store, *, soul=DEFAULT_SOUL, registry=None,
                        project_context: str = "") -> Tuple[AgentContext, Optional[str]]:
```
and use that `project_context` in `render_stable_system_prompt`. Add `Settings.project_context: str = ""` in `app/config.py`. (The route/AppState wiring to pass `state`'s project context is done in Task 5 alongside the other wiring; for now the default `""` keeps everything green.)

- [ ] **Step 7: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_soul.py ../tests/app/test_builder.py ../tests/app/test_memory_inject.py -q` → pass.

- [ ] **Step 8: Full suite** — `cd backend && python -m pytest ../tests/app -q` → all pass (echo/route tests assert on output, unaffected; the agent now sends an extra trailing system message only when context_block is non-empty).

- [ ] **Step 9: Commit**

```bash
git add backend/agent/soul.py backend/agent/context.py backend/agent/agent.py backend/app/builder.py backend/app/config.py tests/app/test_soul.py tests/app/test_builder.py tests/app/test_memory_inject.py
git commit -m "feat(agent): split prompt into stable (cacheable) system + volatile context block (+ project context)"
```

---

## Task 2: Conversation summary columns + store method

**Files:** Modify `app/store/base.py`, `app/models.py`, `app/store/memory.py`, `app/store/sql.py`. Test: `tests/app/test_summary_store.py`.

**Interfaces:**
- Produces: `Conversation.summary: Optional[str] = None`, `Conversation.summarized_seq: int = -1`; `update_conversation_summary(conversation_id, summary, summarized_seq) -> None` (both stores); `get_conversation` returns the new fields; `ConversationSummaryRow` columns.

- [ ] **Step 1: Write the failing test** — `tests/app/test_summary_store.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import pytest
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.db import make_engine, create_all


async def _mem():
    return InMemoryStore()


async def _sql():
    e = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(e)
    return SqlStore(e)


STORES = [_mem, _sql]


@pytest.mark.parametrize("make_store", STORES)
def test_summary_defaults_and_update(make_store):
    async def run():
        st = await make_store()
        await st.ensure_conversation("c1", user_id="u1", title="t")
        conv = await st.get_conversation("c1")
        assert conv.summary is None and conv.summarized_seq == -1
        await st.update_conversation_summary("c1", "they discussed hiking", 7)
        conv = await st.get_conversation("c1")
        assert conv.summary == "they discussed hiking" and conv.summarized_seq == 7
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — FAIL (`summary`/`update_conversation_summary` missing).

- [ ] **Step 3: Add fields + Protocol** in `app/store/base.py`

Add to the `Conversation` dataclass (after `last_response_id`):
```python
    summary: Optional[str] = None
    summarized_seq: int = -1
```
Add to `ResponseStore`:
```python
    async def update_conversation_summary(self, conversation_id: str, summary: str,
                                          summarized_seq: int) -> None: ...
```

- [ ] **Step 4: Add columns** in `app/models.py` `Conversation`

```python
    summary: Optional[str] = Field(default=None, sa_column=Column("summary", Text))
    summarized_seq: int = Field(default=-1)
```
(Add `from sqlalchemy import Text` if not present.)

- [ ] **Step 5: InMemoryStore** (`app/store/memory.py`)

```python
    async def update_conversation_summary(self, conversation_id, summary, summarized_seq) -> None:
        conv = self._convs.get(conversation_id)
        if conv is not None:
            conv.summary = summary
            conv.summarized_seq = summarized_seq
```

- [ ] **Step 6: SqlStore** (`app/store/sql.py`)

In `_to_conv`, map the new fields:
```python
    return Conversation(id=row.id, user_id=row.user_id, title=row.title,
                        last_response_id=row.last_response_id,
                        created_at=row.created_at, updated_at=row.updated_at,
                        summary=row.summary, summarized_seq=row.summarized_seq)
```
Add the method:
```python
    async def update_conversation_summary(self, conversation_id, summary, summarized_seq) -> None:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ConvRow, conversation_id)
            if row is not None:
                row.summary = summary
                row.summarized_seq = summarized_seq
                s.add(row)
                await s.commit()
```

- [ ] **Step 7: Run to verify pass + full suite** — `cd backend && python -m pytest ../tests/app/test_summary_store.py -q` → 2 pass; full suite green.

- [ ] **Step 8: Commit**

```bash
git add backend/app/store/base.py backend/app/models.py backend/app/store/memory.py backend/app/store/sql.py tests/app/test_summary_store.py
git commit -m "feat(store): Conversation.summary/summarized_seq + update_conversation_summary (both stores)"
```

---

## Task 3: `ConversationSummarizer` + `maybe_summarize_conversation` (pure)

**Files:** Create `app/summarizer.py`. Test: `tests/app/test_summarizer.py`.

**Interfaces:**
- Consumes: `Item` + store (`get_conversation`, `get_conversation_items`, `update_conversation_summary`); a `complete` fn.
- Produces:
  - `ConversationSummarizer(complete)` `async def summarize(self, prior_summary, items) -> str` — folds prior summary + the given items into a new concise summary; returns `prior_summary` on any failure (never raises).
  - `async def maybe_summarize_conversation(store, conversation_id, complete, keep_recent=20, batch=20) -> bool` — if `#items with seq > summarized_seq > keep_recent + batch`, fold items in `(summarized_seq, fold_to_seq]` (all but the last `keep_recent`) and persist `(summary, fold_to_seq)`. Returns True if it summarized. Fully guarded.

- [ ] **Step 1: Write the failing test** — `tests/app/test_summarizer.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.store.memory import InMemoryStore
from app.store.base import Item
from app.summarizer import ConversationSummarizer, maybe_summarize_conversation


def _complete_returning(payload):
    async def complete(prompt):
        return payload
    return complete


def _complete_raising():
    async def complete(prompt):
        raise RuntimeError("llm down")
    return complete


def test_summarize_returns_text_and_prior_on_failure():
    s = ConversationSummarizer(_complete_returning("NEW SUMMARY"))
    out = asyncio.run(s.summarize("old", [Item(type="message", role="user", content={"text": "hi"})]))
    assert out == "NEW SUMMARY"
    s2 = ConversationSummarizer(_complete_raising())
    assert asyncio.run(s2.summarize("PRIOR", [])) == "PRIOR"


def test_maybe_summarize_below_threshold_noop():
    async def run():
        st = InMemoryStore()
        await st.ensure_conversation("c1", user_id="u1", title="t")
        await st.append_items("c1", [Item(type="message", role="user", content={"text": f"m{i}"})
                                     for i in range(5)])
        did = await maybe_summarize_conversation(st, "c1", _complete_returning("S"),
                                                 keep_recent=20, batch=20)
        assert did is False
        assert (await st.get_conversation("c1")).summary is None
    asyncio.run(run())


def test_maybe_summarize_folds_overflow_and_keeps_recent():
    async def run():
        st = InMemoryStore()
        await st.ensure_conversation("c1", user_id="u1", title="t")
        await st.append_items("c1", [Item(type="message", role="user", content={"text": f"m{i}"})
                                     for i in range(10)])  # seq 0..9
        did = await maybe_summarize_conversation(st, "c1", _complete_returning("SUM"),
                                                 keep_recent=3, batch=3)  # 10 > 3+3
        assert did is True
        conv = await st.get_conversation("c1")
        assert conv.summary == "SUM"
        # folded all but the last keep_recent(3): summarized_seq == 6 (items 0..6 folded, 7,8,9 kept)
        assert conv.summarized_seq == 6
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: app.summarizer`.

- [ ] **Step 3: Implement `app/summarizer.py`**

```python
from __future__ import annotations
from typing import Awaitable, Callable, List
from loguru import logger
from app.store.base import Item

CompleteFn = Callable[[str], Awaitable[str]]

_SUMMARY_INSTRUCTIONS = """You maintain a running summary of a conversation so older
turns can be dropped from context without losing important information.
Given the prior summary and the next batch of older messages, produce an UPDATED,
concise summary (a few short paragraphs max) that preserves durable facts, decisions,
open questions, and user intent. Do not include pleasantries or verbatim transcripts.
Output ONLY the updated summary text."""


def _item_line(it: Item) -> str:
    c = it.content or {}
    if it.type == "message":
        return f"{it.role or 'user'}: {c.get('text', '')}"
    if it.type == "function_call":
        return f"assistant called {c.get('name', '')}({c.get('arguments', '')})"
    if it.type == "function_call_output":
        return f"tool result: {c.get('output', '')}"
    return ""


class ConversationSummarizer:
    def __init__(self, complete: CompleteFn):
        self._complete = complete

    async def summarize(self, prior_summary: str, items: List[Item]) -> str:
        lines = "\n".join(line for line in (_item_line(it) for it in items) if line) or "(none)"
        prompt = (
            _SUMMARY_INSTRUCTIONS
            + "\n\nPrior summary:\n" + (prior_summary or "(none)")
            + "\n\nNew messages to fold in:\n" + lines
            + "\n\nUpdated summary:"
        )
        try:
            out = await self._complete(prompt)
        except Exception:
            logger.exception("conversation summarize failed")
            return prior_summary
        out = (out or "").strip()
        return out or prior_summary


async def maybe_summarize_conversation(
    store, conversation_id: str, complete: CompleteFn,
    keep_recent: int = 20, batch: int = 20,
) -> bool:
    """Fold old unsummarized items into the rolling summary when they overflow the
    kept-recent window. Returns True if a summary was written. Fully guarded."""
    try:
        conv = await store.get_conversation(conversation_id)
        if conv is None:
            return False
        items = await store.get_conversation_items(conversation_id)
        unsummarized = [it for it in items if it.seq > conv.summarized_seq]
        if len(unsummarized) <= keep_recent + batch:
            return False
        to_fold = unsummarized[:-keep_recent]  # all but the most recent keep_recent
        if not to_fold:
            return False
        fold_to_seq = to_fold[-1].seq
        new_summary = await ConversationSummarizer(complete).summarize(
            conv.summary or "", to_fold)
        await store.update_conversation_summary(conversation_id, new_summary, fold_to_seq)
        return True
    except Exception:
        logger.exception("maybe_summarize_conversation failed")
        return False
```

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_summarizer.py -q` → 4 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/summarizer.py tests/app/test_summarizer.py
git commit -m "feat(app): rolling ConversationSummarizer + maybe_summarize_conversation (pure, guarded)"
```

---

## Task 4: Serve the summary in `build_context`

**Files:** Modify `app/builder.py`. Test: `tests/app/test_summary_inject.py`.

**Interfaces:**
- Consumes: `get_conversation` (summary/summarized_seq), `render_context_block(summary=)`.
- Produces: when the conversation has a summary, `build_context` injects it into the context block and serves only history items with `seq > summarized_seq`.

- [ ] **Step 1: Write the failing test** — `tests/app/test_summary_inject.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.builder import build_context
from app.schemas import ResponsesRequest
from app.store.memory import InMemoryStore
from app.store.base import Item, StoredResponse


def test_build_context_injects_summary_and_trims_old_history():
    async def run():
        st = InMemoryStore()
        conv = await st.create_conversation()
        await st.append_items(conv.id, [
            Item(type="message", role="user", content={"text": "old q"}, response_id="r0"),
            Item(type="message", role="assistant", content={"text": "old a"}, response_id="r0"),
            Item(type="message", role="user", content={"text": "recent q"}, response_id="r1"),
            Item(type="message", role="assistant", content={"text": "recent a"}, response_id="r1"),
        ])  # seq 0..3
        await st.save_response(StoredResponse(id="r1", conversation_id=conv.id, model="m", status="completed"))
        await st.update_conversation_summary(conv.id, "earlier: discussed old q", 1)  # fold seq 0,1
        req = ResponsesRequest(model="m", input="next", conversation=conv.id)
        ctx, _ = await build_context(req, st)
        # summary present in the volatile block
        assert "earlier: discussed old q" in ctx.context_block
        assert "# Conversation summary" in ctx.context_block
        # only items with seq>1 remain in history (old q/old a dropped)
        hist_text = " ".join(m.content for m in ctx.history if isinstance(m.content, str))
        assert "recent q" in hist_text and "old q" not in hist_text
    asyncio.run(run())


def test_no_summary_serves_full_history():
    async def run():
        st = InMemoryStore()
        conv = await st.create_conversation()
        await st.append_items(conv.id, [
            Item(type="message", role="user", content={"text": "q1"}, response_id="r0"),
        ])
        await st.save_response(StoredResponse(id="r0", conversation_id=conv.id, model="m", status="completed"))
        ctx, _ = await build_context(ResponsesRequest(model="m", input="x", conversation=conv.id), st)
        assert "# Conversation summary" not in ctx.context_block
        assert any("q1" in (m.content or "") for m in ctx.history if isinstance(m.content, str))
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — FAIL (summary not injected; old history not trimmed).

- [ ] **Step 3: Inject in `build_context`** (`app/builder.py`)

After resolving `history_items` and `conversation_id` (and before building messages), load the conversation summary and trim:
```python
    summary = ""
    if conversation_id:
        conv = await store.get_conversation(conversation_id)
        if conv is not None and conv.summary:
            summary = conv.summary
            history_items = [it for it in history_items if it.seq > conv.summarized_seq]
```
Pass `summary=summary` into `render_context_block(...)`:
```python
    context_block = render_context_block(memories=memories, summary=summary, instructions=instructions)
```

NOTE: `history_items` is the list returned by `resolve_history`; `Item.seq` is available. Place this block after `history_items` is assigned and `conversation_id` resolved, before `items_to_messages(history_items)` is called in the `AgentContext(...)` construction. (If `items_to_messages` is called inline in the constructor, compute `history = items_to_messages(history_items)` into a local first, after trimming.)

- [ ] **Step 4: Run to verify pass + full suite** — `cd backend && python -m pytest ../tests/app/test_summary_inject.py -q` → 2 pass; full suite green.

- [ ] **Step 5: Commit**

```bash
git add backend/app/builder.py tests/app/test_summary_inject.py
git commit -m "feat(app): serve rolling summary + trim summarized history in build_context"
```

---

## Task 5: Async post-turn summarization + project-context wiring

**Files:** Modify `app/routes/responses.py`, `app/config.py`, `app/deps.py`, `app/lean_main.py`. Test: `tests/app/test_routes_summary.py`.

**Interfaces:**
- Consumes: `maybe_summarize_conversation`/`make_complete`; `Settings.summary_*`, `project_context`; `AppState`.
- Produces: after a stored turn (sync + stream + background), if `state.summary_enabled`, schedule `asyncio.create_task(maybe_summarize_conversation(store, conversation_id, make_complete(llm), keep_recent, batch))`. `AppState.summary_enabled`, `summary_keep_recent`, `summary_batch`, `project_context`; route passes `project_context=state.project_context` to `build_context`; lean_main wires settings.

- [ ] **Step 1: Write the failing test** — `tests/app/test_routes_summary.py`

```python
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.routes.responses import router as responses_router
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import CompletionUsage


class _SumLLM:
    async def astream(self, messages, tools=None, **kwargs):
        prompt = ""
        for m in messages:
            if m.get("role") == "user":
                prompt = m.get("content") or ""
        usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        is_summary = "Updated summary:" in prompt

        async def gen():
            yield TextChunk(delta=("ROLLED-UP SUMMARY" if is_summary else "ok"), usage=None)
            yield TextChunk(delta="", usage=usage)
        return gen()


def _client():
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(), llm=_SumLLM(), default_model="m",
        summary_enabled=True, summary_keep_recent=1, summary_batch=1,
    )
    app.include_router(responses_router)
    return TestClient(app)


def test_long_conversation_gets_rolling_summary():
    with TestClient(_client().app) as c:
        conv_id = None
        for i in range(4):
            body = c.post("/v1/responses", json={
                "input": f"message {i}", "stream": False,
                **({"conversation": conv_id, "previous_response_id": rid} if conv_id else {}),
            }).json()
            conv_id = body["conversation"]["id"]
            rid = body["id"]
        # background summarization runs on the app loop; poll the conversation summary
        summary = None
        for _ in range(100):
            detail = c.get(f"/v1/conversations/{conv_id}").json()
            # latest_response_id present means persisted; check the store summary via detail? use store
            conv = None
            import asyncio
            async def _read():
                return await c.app.state.app_state.store.get_conversation(conv_id)
            conv = asyncio.run(_read())
            if conv and conv.summary:
                summary = conv.summary
                break
            time.sleep(0.02)
        assert summary == "ROLLED-UP SUMMARY"


def test_summary_disabled_writes_no_summary():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_SumLLM(), default_model="m",
                                   summary_enabled=False)
    app.include_router(responses_router)
    import asyncio
    with TestClient(app) as c:
        conv_id = None
        rid = None
        for i in range(4):
            body = c.post("/v1/responses", json={"input": f"m{i}", "stream": False,
                          **({"conversation": conv_id, "previous_response_id": rid} if conv_id else {})}).json()
            conv_id = body["conversation"]["id"]; rid = body["id"]
        time.sleep(0.2)
        conv = asyncio.run(c.app.state.app_state.store.get_conversation(conv_id))
        assert conv.summary is None
```

NOTE: this end-to-end test uses `with TestClient(...) as c:` so the `create_task` summarization runs on the app loop; it polls the store. `summary_keep_recent=1, summary_batch=1` makes 4 turns (≈8 items) cross the threshold quickly.

- [ ] **Step 2: Run to verify failure** — `AppState` has no `summary_enabled` etc.

- [ ] **Step 3: Settings + AppState**

`app/config.py`:
```python
    summary_enabled: bool = False
    summary_keep_recent: int = 20
    summary_batch: int = 20
    project_context: str = ""   # (added in Task 1 if not already)
```
`app/deps.py` `AppState` (after `memory_model`):
```python
    summary_enabled: bool = False
    summary_keep_recent: int = 20
    summary_batch: int = 20
    project_context: str = ""
```

- [ ] **Step 4: Schedule summarization** in `app/routes/responses.py`

Add `from app.summarizer import maybe_summarize_conversation` (and `make_complete` is already imported). Add a helper near `_schedule_memory_update`:
```python
def _schedule_summary(state, request, conversation_id):
    if not (getattr(state, "summary_enabled", False) and conversation_id):
        return
    llm = None
    if state.router is not None:
        try:
            llm = state.router.get_llm(getattr(state, "memory_model", "") or request.model)
        except Exception:
            llm = None
    llm = llm or state.llm
    if llm is None:
        return
    asyncio.create_task(maybe_summarize_conversation(
        state.store, conversation_id, make_complete(llm),
        keep_recent=getattr(state, "summary_keep_recent", 20),
        batch=getattr(state, "summary_batch", 20),
    ))
```
Call `_schedule_summary(state, request, conversation_id)` right after each `_schedule_memory_update(...)` call (sync + stream + background paths).

Also pass project context to `build_context`:
```python
        ctx, conversation_id = await build_context(
            request, state.store, soul=state.soul, registry=(state.registry if tools_ok else None),
            project_context=getattr(state, "project_context", ""),
        )
```

- [ ] **Step 5: lean_main wiring** (`app/lean_main.py`)

Pass the new settings to `AppState(...)`:
```python
        summary_enabled=settings.summary_enabled,
        summary_keep_recent=settings.summary_keep_recent,
        summary_batch=settings.summary_batch,
        project_context=settings.project_context,
```

- [ ] **Step 6: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_routes_summary.py -q` → 2 pass.

- [ ] **Step 7: Full suite + boot/isolation** — `cd backend && python -m pytest ../tests/app -q` → all pass.

- [ ] **Step 8: Commit**

```bash
git add backend/app/routes/responses.py backend/app/config.py backend/app/deps.py backend/app/lean_main.py tests/app/test_routes_summary.py
git commit -m "feat(app): async post-turn rolling summarization + project-context wiring"
```

---

## Self-Review (against the spec)

- **③ cacheable split** (stable `render_stable_system_prompt` front + volatile `render_context_block` as a trailing system message; `AgentContext.context_block`; `build_messages` ordering) → Task 1. Project-context layer (static, no retrieval) → Task 1 + Task 5 wiring.
- **① rolling summary** (Conversation.summary/summarized_seq + store method → Task 2; `ConversationSummarizer`/`maybe_summarize_conversation` → Task 3; `build_context` serves summary + trims history → Task 4; async gated post-turn scheduling → Task 5).
- **② kept, not retrieved** — user memory still injected via `render_context_block(memories=...)` capped at `MEMORY_INJECT_LIMIT`; no retrieval added. (Future TODO.)
- **`fit_to_budget` unchanged** (in-call fallback). **Async/guarded** memory+summary never break a turn. **Both stores lockstep.** **Gated/back-compat:** summary_enabled default off; build_context serves full history when no summary; existing tests updated for the prompt split.

No placeholders; signatures (`render_stable_system_prompt`, `render_context_block`, `AgentContext.context_block`, `update_conversation_summary`, `ConversationSummarizer.summarize`, `maybe_summarize_conversation(...keep_recent,batch)`, `build_context(..., project_context=)`) consistent across tasks.
