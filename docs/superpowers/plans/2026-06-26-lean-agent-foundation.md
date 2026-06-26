# Lean Agent — Foundation (schema + store + lean agent core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `agent/` boot with zero heavy deps, and stand up the lean service's persistence layer — a fresh SQLModel schema (`create_all`, no migrations) and a `ResponseStore` (in-memory + SQLite/Postgres).

**Architecture:** New `backend/app/` package for the lean service. This foundation phase: (1) make `agent/` trace + tokenizer optional, (2) `app/config.py` + `app/db.py` + `app/models.py` (3 tables), (3) `app/store/` (protocol + `InMemoryStore` + `SqlStore`). The running endpoints (`LeanLLM`, responses serializer, routes) and legacy relocation are later plans.

**Tech Stack:** Python 3.11, SQLModel + SQLAlchemy async, aiosqlite (default) / asyncpg, pydantic-settings, pytest, tenacity.

**Reference spec:** `docs/superpowers/specs/2026-06-26-standalone-lean-agent-design.md` (steps 1–3). Branch: `personal/yfei/agent-core`.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/agent/agent.py` (modify) | trace imports become optional no-ops |
| `backend/agent/tools/base.py` (modify) | per-tool tracer becomes optional no-op |
| `backend/agent/budgeting.py` (modify) | tokenizer load becomes optional (length fallback) |
| `backend/app/__init__.py` (new) | package marker |
| `backend/app/config.py` (new) | `Settings` from env |
| `backend/app/db.py` (new) | async engine + `create_all()` |
| `backend/app/models.py` (new) | `Conversation`, `ConversationItem`, `ResponseRow` SQLModel tables |
| `backend/app/store/base.py` (new) | `ResponseStore` protocol + dataclasses |
| `backend/app/store/memory.py` (new) | `InMemoryStore` |
| `backend/app/store/sql.py` (new) | `SqlStore` |
| `tests/app/test_*` (new) | unit tests |

---

## Task 1: make `agent/` boot with zero heavy deps

Make tracing + tokenizer optional so the agent core imports without `extensions/trace`, `opentelemetry`, or tokenizer files. Behavior is unchanged when those ARE present. READ `backend/agent/agent.py` (top imports + `run`), `backend/agent/tools/base.py` (`_call_with_retry`), `backend/agent/budgeting.py` (`__init__` + `estimate_msg_tokens`).

**Files:** Modify `backend/agent/agent.py`, `backend/agent/tools/base.py`, `backend/agent/budgeting.py`. Test: `tests/app/test_lean_agent_core.py`.

- [ ] **Step 1: Write the failing test**

```text
# tests/app/test_lean_agent_core.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))


def test_agent_core_imports_without_trace(monkeypatch):
    # Simulate the trace extension / opentelemetry being absent.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("extensions.trace") or name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError(f"simulated-absent: {name}")
        return real_import(name, *a, **k)

    for m in list(sys.modules):
        if m.startswith("agent.agent") or m.startswith("agent.tools") or m == "agent.budgeting":
            sys.modules.pop(m, None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    import importlib
    import agent.agent as agent_mod  # must import despite missing trace deps
    importlib.reload(agent_mod)
    assert agent_mod.Agent is not None


def test_budgeting_without_tokenizer(monkeypatch):
    import agent.budgeting as b
    monkeypatch.setattr(b, "get_tokenizer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no tokenizer")))
    mgr = b.AgentMessageManager(context_window=110000, max_output_tokens=8000)
    # estimate still returns a positive int via the length fallback
    n = mgr.estimate_msg_tokens({"role": "user", "content": "hello world this is some text"})
    assert isinstance(n, int) and n > 0
```

- [ ] **Step 2: Run to verify failure** — `cd backend && python -m pytest ../tests/app/test_lean_agent_core.py -v` (create `tests/app/__init__.py` empty). Expected: FAIL (trace import raises at module load / tokenizer fallback missing).

- [ ] **Step 3: Make `agent/agent.py` trace imports optional**

Replace the three trace import lines at the top of `agent/agent.py`:
```text
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from extensions.trace.base import use_current_span
from opentelemetry import trace
```
with:
```text
try:
    from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
    from extensions.trace.base import use_current_span
    from opentelemetry import trace
except Exception:  # lean mode: no trace extension / opentelemetry
    def pai_agent_wrapper(func):           # passthrough decorator
        return func

    def use_current_span(_span):           # passthrough decorator factory
        def _deco(fn):
            return fn
        return _deco

    class _NoTrace:
        @staticmethod
        def get_current_span():
            return None

    trace = _NoTrace()
```
The existing `@pai_agent_wrapper` on `run` and `@use_current_span(trace.get_current_span())` on `gen()` now work unchanged in both modes (real tracing if installed; no-ops otherwise).

- [ ] **Step 4: Make `agent/tools/base.py` tracer optional**

In `_call_with_retry`, replace the body that does `from extensions.trace.tracer import get_tracer` + `with get_tracer().start_as_current_span(...)` with:
```text
async def _call_with_retry(tool: "Tool", args: dict) -> str:
    try:
        from extensions.trace.tracer import get_tracer
        span_cm = get_tracer().start_as_current_span(f"tool {tool.name}")
    except Exception:
        from contextlib import nullcontext
        span_cm = nullcontext()
    with span_cm as span:
        if span is not None:
            try:
                span.set_attribute("tool.name", tool.name)
            except Exception:
                pass
        result = await tool.fn(**args)
        return result if isinstance(result, str) else str(result)
```

- [ ] **Step 5: Make `agent/budgeting.py` tokenizer optional**

In `AgentMessageManager.__init__`, wrap the tokenizer load:
```text
        try:
            self.tokenizer = get_tokenizer()
        except Exception:
            logger.warning("Tokenizer unavailable; using length-based token estimate.")
            self.tokenizer = None
```
Then guard every place that calls `estimate_tokens_in_text(text, tokenizer=self.tokenizer)` and `truncate(...)` so a `None` tokenizer falls back to a char heuristic. Add a small helper near the top of the class file:
```text
def _estimate_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return max(1, len(text) // 4)   # ~4 chars/token heuristic
    return estimate_tokens_in_text(text, tokenizer=tokenizer)
```
and replace the `estimate_tokens_in_text(..., tokenizer=self.tokenizer)` calls in `estimate_msg_tokens` (and anywhere else in the class) with `_estimate_tokens(text, self.tokenizer)`. For `cap_tool_result`/`truncate` with a `None` tokenizer, fall back to a char cap (`content[: self.max_tool_result_tokens * 4]` + the truncation marker). Read the file and apply consistently.

- [ ] **Step 6: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_lean_agent_core.py ../tests/agent/ -q` → all pass (the agent suite must stay green WITH the trace extension present).

- [ ] **Step 7: Commit**

```bash
git add backend/agent/agent.py backend/agent/tools/base.py backend/agent/budgeting.py tests/app/
git commit -m "feat(agent): optional tracing + tokenizer so agent core boots lean"
```

---

## Task 2: config + db + schema

**Files:** Create `backend/app/__init__.py`, `backend/app/config.py`, `backend/app/db.py`, `backend/app/models.py`. Test: `tests/app/test_models.py`.

- [ ] **Step 1: Write the failing test**

```text
# tests/app/test_models.py
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.db import make_engine, create_all
from app.models import Conversation, ConversationItem, ResponseRow
from sqlmodel.ext.asyncio.session import AsyncSession


def test_create_all_and_insert_roundtrip():
    async def run():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        async with AsyncSession(engine) as s:
            conv = Conversation(id="conv_1")
            s.add(conv)
            s.add(ConversationItem(id="item_1", conversation_id="conv_1", seq=0,
                                   type="message", role="user", content={"text": "hi"}))
            s.add(ResponseRow(id="resp_1", conversation_id="conv_1", model="m", status="completed"))
            await s.commit()
            got = await s.get(ResponseRow, "resp_1")
            assert got.conversation_id == "conv_1" and got.status == "completed"
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: app.db`.

- [ ] **Step 3: Implement `backend/app/config.py`**

```text
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    default_model: str = "gpt-4o-mini"
    db_url: str = "sqlite+aiosqlite:///./data/agent.db"
    store_backend: str = "sql"   # "sql" | "memory"


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Implement `backend/app/db.py`**

```text
from __future__ import annotations
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import SQLModel
import app.models  # noqa: F401  (register tables on SQLModel.metadata)


def make_engine(db_url: str) -> AsyncEngine:
    return create_async_engine(db_url, future=True)


async def create_all(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
```

- [ ] **Step 5: Implement `backend/app/models.py`**

```text
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON, BigInteger, Text


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"
    id: str = Field(primary_key=True, max_length=64)
    user_id: Optional[str] = Field(default=None, index=True, max_length=64)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class ConversationItem(SQLModel, table=True):
    __tablename__ = "conversation_items"
    id: str = Field(primary_key=True, max_length=64)
    conversation_id: str = Field(index=True, max_length=64)
    seq: int = Field(sa_column=Column(BigInteger))
    type: str = Field(max_length=32)        # message|reasoning|function_call|function_call_output
    role: Optional[str] = Field(default=None, max_length=16)
    content: dict = Field(default_factory=dict, sa_column=Column(JSON))
    response_id: Optional[str] = Field(default=None, index=True, max_length=64)
    created_at: datetime = Field(default_factory=_now)


class ResponseRow(SQLModel, table=True):
    __tablename__ = "responses"
    id: str = Field(primary_key=True, max_length=64)
    conversation_id: Optional[str] = Field(default=None, index=True, max_length=64)
    previous_response_id: Optional[str] = Field(default=None, index=True, max_length=64)
    model: str = Field(max_length=128)
    status: str = Field(max_length=32)
    usage: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    error: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
```

Create empty `backend/app/__init__.py`.

- [ ] **Step 6: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_models.py -v` → 1 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/app/__init__.py backend/app/config.py backend/app/db.py backend/app/models.py tests/app/test_models.py
git commit -m "feat(app): lean Settings + async engine/create_all + clean schema"
```

---

## Task 3: `ResponseStore` protocol + `InMemoryStore`

**Files:** Create `backend/app/store/__init__.py`, `backend/app/store/base.py`, `backend/app/store/memory.py`. Test: `tests/app/test_store_memory.py`.

- [ ] **Step 1: Write failing tests**

```text
# tests/app/test_store_memory.py
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.store.memory import InMemoryStore
from app.store.base import Item, StoredResponse


def _store(): return InMemoryStore()


def test_create_conversation_and_append_items_ordered():
    async def run():
        st = _store()
        conv = await st.create_conversation(user_id="u1")
        await st.append_items(conv.id, [Item(type="message", role="user", content={"text": "q1"})])
        await st.append_items(conv.id, [Item(type="message", role="assistant", content={"text": "a1"})])
        items = await st.get_conversation_items(conv.id)
        assert [i.seq for i in items] == [0, 1]
        assert items[0].content["text"] == "q1"
    asyncio.run(run())


def test_save_get_delete_response():
    async def run():
        st = _store()
        conv = await st.create_conversation()
        r = await st.save_response(StoredResponse(id="resp_1", conversation_id=conv.id,
                                                  model="m", status="completed"))
        assert (await st.get_response("resp_1")).id == "resp_1"
        await st.delete_response("resp_1")
        assert await st.get_response("resp_1") is None
    asyncio.run(run())


def test_resolve_history_by_previous_response_id():
    async def run():
        st = _store()
        conv = await st.create_conversation()
        await st.append_items(conv.id, [Item(type="message", role="user", content={"text": "q1"}, response_id="resp_1")])
        await st.save_response(StoredResponse(id="resp_1", conversation_id=conv.id, model="m", status="completed"))
        hist = await st.resolve_history(previous_response_id="resp_1", conversation=None)
        assert any(i.content.get("text") == "q1" for i in hist)
    asyncio.run(run())


def test_resolve_history_conflicting_ids_raises():
    async def run():
        st = _store()
        c1 = await st.create_conversation(); c2 = await st.create_conversation()
        await st.save_response(StoredResponse(id="resp_x", conversation_id=c1.id, model="m", status="completed"))
        import pytest
        with pytest.raises(ValueError):
            await st.resolve_history(previous_response_id="resp_x", conversation=c2.id)
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: app.store.memory`.

- [ ] **Step 3: Implement `backend/app/store/base.py`**

```text
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Protocol


def _uuid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass
class Item:
    type: str
    content: dict
    role: Optional[str] = None
    response_id: Optional[str] = None
    id: str = field(default_factory=lambda: _uuid("item"))
    seq: int = 0


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: _uuid("conv"))
    user_id: Optional[str] = None


@dataclass
class StoredResponse:
    id: str
    model: str
    status: str
    conversation_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[dict] = None


class ResponseStore(Protocol):
    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation: ...
    async def append_items(self, conversation_id: str, items: List[Item]) -> List[Item]: ...
    async def get_conversation_items(self, conversation_id: str) -> List[Item]: ...
    async def save_response(self, response: StoredResponse) -> StoredResponse: ...
    async def get_response(self, response_id: str) -> Optional[StoredResponse]: ...
    async def delete_response(self, response_id: str) -> None: ...
    async def resolve_history(self, previous_response_id: Optional[str],
                              conversation: Optional[str]) -> List[Item]: ...
```

- [ ] **Step 4: Implement `backend/app/store/memory.py`**

```text
from __future__ import annotations
from typing import Dict, List, Optional
from app.store.base import Conversation, Item, StoredResponse, _uuid


class InMemoryStore:
    def __init__(self):
        self._convs: Dict[str, Conversation] = {}
        self._items: Dict[str, List[Item]] = {}
        self._responses: Dict[str, StoredResponse] = {}

    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        conv = Conversation(user_id=user_id)
        self._convs[conv.id] = conv
        self._items[conv.id] = []
        return conv

    async def append_items(self, conversation_id: str, items: List[Item]) -> List[Item]:
        log = self._items.setdefault(conversation_id, [])
        for it in items:
            it.seq = len(log)
            log.append(it)
        return items

    async def get_conversation_items(self, conversation_id: str) -> List[Item]:
        return list(self._items.get(conversation_id, []))

    async def save_response(self, response: StoredResponse) -> StoredResponse:
        self._responses[response.id] = response
        return response

    async def get_response(self, response_id: str) -> Optional[StoredResponse]:
        return self._responses.get(response_id)

    async def delete_response(self, response_id: str) -> None:
        self._responses.pop(response_id, None)

    async def resolve_history(self, previous_response_id, conversation) -> List[Item]:
        conv_id = conversation
        if previous_response_id:
            resp = self._responses.get(previous_response_id)
            if resp is None:
                return []
            if conversation and resp.conversation_id != conversation:
                raise ValueError("previous_response_id does not belong to conversation")
            conv_id = resp.conversation_id
        if not conv_id:
            return []
        return list(self._items.get(conv_id, []))
```

Create empty `backend/app/store/__init__.py`.

- [ ] **Step 5: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_store_memory.py -v` → 4 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/store/__init__.py backend/app/store/base.py backend/app/store/memory.py tests/app/test_store_memory.py
git commit -m "feat(app): ResponseStore protocol + InMemoryStore (history resolution + precedence)"
```

---

## Task 4: `SqlStore`

Same `ResponseStore` protocol, backed by SQLModel over `DB_URL`. READ `app/models.py` (Task 2) for the table classes.

**Files:** Create `backend/app/store/sql.py`. Test: `tests/app/test_store_sql.py`.

- [ ] **Step 1: Write failing tests** (mirror the memory tests against a real in-memory SQLite)

```text
# tests/app/test_store_sql.py
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.db import make_engine, create_all
from app.store.sql import SqlStore
from app.store.base import Item, StoredResponse


def _fresh_store():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    return engine


def test_sql_append_order_and_response_crud_and_resolve():
    async def run():
        engine = _fresh_store()
        await create_all(engine)
        st = SqlStore(engine)
        conv = await st.create_conversation(user_id="u1")
        await st.append_items(conv.id, [Item(type="message", role="user", content={"text": "q1"}, response_id="resp_1")])
        await st.append_items(conv.id, [Item(type="message", role="assistant", content={"text": "a1"})])
        items = await st.get_conversation_items(conv.id)
        assert [i.seq for i in items] == [0, 1] and items[0].content["text"] == "q1"
        await st.save_response(StoredResponse(id="resp_1", conversation_id=conv.id, model="m", status="completed"))
        assert (await st.get_response("resp_1")).status == "completed"
        hist = await st.resolve_history(previous_response_id="resp_1", conversation=None)
        assert any(i.content.get("text") == "q1" for i in hist)
        await st.delete_response("resp_1")
        assert await st.get_response("resp_1") is None
    asyncio.run(run())


def test_sql_conflicting_ids_raise():
    async def run():
        engine = _fresh_store()
        await create_all(engine)
        st = SqlStore(engine)
        c1 = await st.create_conversation(); c2 = await st.create_conversation()
        await st.save_response(StoredResponse(id="resp_x", conversation_id=c1.id, model="m", status="completed"))
        import pytest
        with pytest.raises(ValueError):
            await st.resolve_history(previous_response_id="resp_x", conversation=c2.id)
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: app.store.sql`.

- [ ] **Step 3: Implement `backend/app/store/sql.py`**

```text
from __future__ import annotations
from typing import List, Optional
from sqlalchemy import select, func, delete
from sqlmodel.ext.asyncio.session import AsyncSession
from app.models import Conversation as ConvRow, ConversationItem as ItemRow, ResponseRow
from app.store.base import Conversation, Item, StoredResponse


def _to_item(row: ItemRow) -> Item:
    return Item(id=row.id, type=row.type, role=row.role, content=row.content or {},
                response_id=row.response_id, seq=row.seq)


class SqlStore:
    def __init__(self, engine):
        self._engine = engine

    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        conv = Conversation(user_id=user_id)
        async with AsyncSession(self._engine) as s:
            s.add(ConvRow(id=conv.id, user_id=user_id))
            await s.commit()
        return conv

    async def append_items(self, conversation_id: str, items: List[Item]) -> List[Item]:
        async with AsyncSession(self._engine) as s:
            base = (await s.exec_(select(func.coalesce(func.max(ItemRow.seq), -1)).where(
                ItemRow.conversation_id == conversation_id))).scalar_one()
            n = int(base) + 1
            for it in items:
                it.seq = n
                s.add(ItemRow(id=it.id, conversation_id=conversation_id, seq=n, type=it.type,
                              role=it.role, content=it.content, response_id=it.response_id))
                n += 1
            await s.commit()
        return items

    async def get_conversation_items(self, conversation_id: str) -> List[Item]:
        async with AsyncSession(self._engine) as s:
            rows = (await s.exec_(select(ItemRow).where(
                ItemRow.conversation_id == conversation_id).order_by(ItemRow.seq))).scalars().all()
        return [_to_item(r) for r in rows]

    async def save_response(self, r: StoredResponse) -> StoredResponse:
        async with AsyncSession(self._engine) as s:
            s.add(ResponseRow(id=r.id, conversation_id=r.conversation_id,
                              previous_response_id=r.previous_response_id, model=r.model,
                              status=r.status, usage=r.usage, error=r.error))
            await s.commit()
        return r

    async def get_response(self, response_id: str) -> Optional[StoredResponse]:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ResponseRow, response_id)
        if row is None:
            return None
        return StoredResponse(id=row.id, model=row.model, status=row.status,
                              conversation_id=row.conversation_id,
                              previous_response_id=row.previous_response_id,
                              usage=row.usage, error=row.error)

    async def delete_response(self, response_id: str) -> None:
        async with AsyncSession(self._engine) as s:
            await s.exec_(delete(ResponseRow).where(ResponseRow.id == response_id))
            await s.commit()

    async def resolve_history(self, previous_response_id, conversation) -> List[Item]:
        conv_id = conversation
        if previous_response_id:
            resp = await self.get_response(previous_response_id)
            if resp is None:
                return []
            if conversation and resp.conversation_id != conversation:
                raise ValueError("previous_response_id does not belong to conversation")
            conv_id = resp.conversation_id
        if not conv_id:
            return []
        return await self.get_conversation_items(conv_id)
```

NOTE on the session API: `AsyncSession.exec_`/`exec`/`execute` naming varies by SQLModel version. If `s.exec_(...)` is wrong, use the version's async query API (likely `await s.exec(select(...))` for SQLModel ≥0.0.14, or `await s.execute(...)` then `.scalars()`). Verify against the installed SQLModel and adapt; keep the query semantics (max(seq), ordered items, get/delete by id).

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_store_sql.py -v` → 2 passed.

- [ ] **Step 5: Full foundation check + commit**

Run: `cd backend && python -m pytest ../tests/app/ ../tests/agent/ -q` → all pass.
```bash
git add backend/app/store/sql.py tests/app/test_store_sql.py
git commit -m "feat(app): SqlStore (SQLite/Postgres) with same history-resolution semantics"
```

---

## Notes for the implementer

- This foundation phase produces **no HTTP endpoint** — it's the agent-core lean adjustments + schema + store. The `LeanLLM`, responses serializer, and routes are the NEXT plan. Each task here is independently testable.
- `agent/` must stay green WITH the trace extension installed (the no-op fallbacks only fire when imports fail). Run the full `tests/agent/` suite after Task 1.
- The `metadata` column name: SQLModel/SQLAlchemy reserves the attribute name `metadata` on declarative classes, so the model field is `meta` mapped to DB column `"metadata"` via `Column("metadata", JSON)`. Keep that mapping.
- SQLite JSON: SQLAlchemy's `JSON` type works on SQLite (stored as text) and Postgres (native JSON) — portable. `BigInteger` for `seq` is fine on both.
- Do NOT wire `app/` into any server entrypoint yet, and do NOT touch the legacy backend — this phase only adds the `app/` foundation + the three `agent/` edits.
