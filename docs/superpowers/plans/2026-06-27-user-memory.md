# User Memory (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the agent a per-user long-term memory: a `User` identity aligned to OpenAI's `user`/`safety_identifier`, messages associated to the user, user memories injected into the system prompt, and a background extraction+consolidation pipeline that learns facts from each stored turn — with list/delete controls.

**Architecture:** `User` + `MemoryItem` join the store (both `InMemoryStore` and `SqlStore`). `build_context` injects the user's (capped, recency-ordered) memories via `render_system_prompt`. After a stored turn, an async, fault-tolerant `update_user_memory` extracts salient facts (LLM) and consolidates them (ADD/UPDATE/DELETE/NOOP). Reuses the detached-run/persist machinery; gated by settings + a per-request flag.

**Tech Stack:** Python 3.11, pydantic/SQLModel, FastAPI + TestClient, pytest. Backend tests from `backend/`: `cd backend && python -m pytest ../tests/app -q`. Async tests use `asyncio.run(run())`.

**Reference spec:** `docs/superpowers/specs/2026-06-27-user-memory-design.md`. Branch: `personal/yfei/agent-core`.

## Global Constraints

- **Import-lean:** new code imports only stdlib, pydantic/SQLModel, loguru, and existing `app.*`/`agent.*`. Boot/import-isolation gates stay green.
- **Backward compatible:** all features are additive and gated. With memory disabled / no user, behavior is unchanged. Existing backend tests (132) stay green. `AppState` keeps working when constructed without new fields (defaults).
- **Memory never breaks a turn:** extraction is async + fully exception-guarded; a failure logs and is dropped.
- **Both stores in lockstep:** every new `ResponseStore` method is implemented in `InMemoryStore` and `SqlStore` and tested against both.
- Run the full app suite at the end of each task.

---

## Key existing contracts (verified, do not re-derive)

- **`app/store/base.py`** — `_uuid(prefix)`, `_now()` (utc), `new_conversation_id()`. Dataclasses: `Item(type, content, role=None, response_id=None, id=<auto>, seq=0)`, `Conversation(id, user_id=None, title=None, last_response_id=None, created_at, updated_at)`, `StoredResponse(...)`. `ResponseStore` Protocol lists async store methods.
- **`app/models.py`** — `Conversation`, `ConversationItem(id, conversation_id idx, seq BigInteger, type, role, content JSON, response_id idx, created_at)`, `ResponseRow`. `_now()`.
- **`app/store/memory.py`** (`InMemoryStore`) holds `_convs`, `_items`, `_responses` dicts; conversation methods (`ensure_conversation`, `touch_conversation`, `list_conversations`, `get_conversation`, `list_conversation_responses`, `delete_conversation`).
- **`app/store/sql.py`** (`SqlStore(engine)`) opens `AsyncSession(self._engine)`; `_to_item`, `_to_conv`; imports `Conversation as ConvRow, ConversationItem as ItemRow, ResponseRow`.
- **`app/schemas.py`** — `ResponsesRequest(extra="ignore")` with `model, input, instructions, previous_response_id, conversation, user_id, store, stream, metadata, tools, soul, background`.
- **`app/builder.py`** — `build_context(request, store, *, soul=DEFAULT_SOUL, registry=None)`; composes `effective_soul`, selects tools, `system_prompt = render_system_prompt(effective_soul, tool_names=...)`.
- **`agent/soul.py`** — `render_system_prompt(soul, *, tool_names, extra="")`; sections Identity/Personality/Operating principles/Tools/Safety/Additional instructions; `_bullets(items)`.
- **`app/routes/responses.py`** — `_persist(state, request, current_turn, response_id, conversation_id, store_items, status, usage, error=None)` calls `ensure_conversation(conversation_id, user_id=request.user_id, title=...)` then builds `Item`s (user item + store items) and `append_items`, `save_response`, `touch_conversation`. `create_response` resolves model (router), `build_context(..., soul=state.soul, registry=...)`, runs the agent; background branch uses `_persist_run`; sync path calls `_persist`.
- **`app/deps.py`** — `@dataclass AppState(store, llm, default_model, context_window, max_output_tokens, soul, registry, router)`; `make_agent(...)`; `get_state`.
- **`app/llm.py`** — `LeanLLM.astream(messages, tools=None, **kwargs)` yields chunks with `.delta`.
- **`app/config.py`** — `Settings(env_prefix="")`; `get_settings()`.
- **`app/lean_main.py`** — lifespan builds `AppState`; mounts routers.
- **Test scaffolding** — `import sys, os[, asyncio]; sys.path.insert(0, ".../backend")`; in-memory Sql engine `make_engine("sqlite+aiosqlite:///:memory:")` + `await create_all(engine)`; route tests via `FastAPI()` + `app.state.app_state = AppState(...)` + `TestClient`. Echo LLM returns an async generator (`startswith("echo:")`).

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/app/store/base.py` (modify) | `User`, `MemoryItem` dataclasses; `Item.user_id`; Protocol additions |
| `backend/app/models.py` (modify) | `UserRow`, `MemoryRow`; `ConversationItem.user_id` |
| `backend/app/schemas.py` (modify) | `ResponsesRequest.user`/`safety_identifier` + `resolved_user_id` |
| `backend/app/store/memory.py` (modify) | user + memory + item.user_id (InMemoryStore) |
| `backend/app/store/sql.py` (modify) | user + memory + item.user_id (SqlStore) |
| `backend/agent/soul.py` (modify) | `render_system_prompt(memories=...)` |
| `backend/app/builder.py` (modify) | inject user memories |
| `backend/app/memory.py` (new) | `MemoryExtractor`, `apply_memory_ops`, `update_user_memory`, `make_complete` |
| `backend/app/routes/responses.py` (modify) | stamp user_id; schedule memory update |
| `backend/app/routes/users.py` (new) | memory management API |
| `backend/app/config.py` (modify) | memory settings |
| `backend/app/lean_main.py` (modify) | mount users router |
| `tests/app/test_user_identity.py`, `test_memory_store.py`, `test_memory_inject.py`, `test_memory_extractor.py`, `test_routes_memory.py` (new) | |

---

## Task 1: User identity (record + request aliases)

**Files:** Modify `backend/app/store/base.py`, `backend/app/models.py`, `backend/app/schemas.py`, `backend/app/store/memory.py`, `backend/app/store/sql.py`. Test: `tests/app/test_user_identity.py`.

**Interfaces:**
- Produces: `User(id, display_name=None, created_at=<utc>, meta={})`; `ResponsesRequest.user`/`safety_identifier` + `resolved_user_id` property; store `ensure_user(user_id, display_name=None) -> User` (idempotent), `get_user(user_id) -> Optional[User]` on both stores; `UserRow` table.

- [ ] **Step 1: Write the failing test** — `tests/app/test_user_identity.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import pytest
from app.schemas import ResponsesRequest
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


def test_resolved_user_id_precedence():
    assert ResponsesRequest(input="x", user_id="a", user="b").resolved_user_id == "a"
    assert ResponsesRequest(input="x", user="b").resolved_user_id == "b"
    assert ResponsesRequest(input="x", safety_identifier="s").resolved_user_id == "s"
    assert ResponsesRequest(input="x").resolved_user_id is None


@pytest.mark.parametrize("make_store", STORES)
def test_ensure_user_idempotent_and_get(make_store):
    async def run():
        st = await make_store()
        u1 = await st.ensure_user("u1", display_name="Alice")
        assert u1.id == "u1" and u1.display_name == "Alice"
        u2 = await st.ensure_user("u1", display_name="ignored")
        assert u2.display_name == "Alice"  # not overwritten
        assert (await st.get_user("u1")).id == "u1"
        assert await st.get_user("nope") is None
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `cd backend && python -m pytest ../tests/app/test_user_identity.py -q` → FAIL (no `user`/`resolved_user_id`/`ensure_user`).

- [ ] **Step 3: Add `User` + `Item.user_id`** in `backend/app/store/base.py`

Add after the `Conversation` dataclass:

```python
@dataclass
class User:
    id: str
    display_name: Optional[str] = None
    created_at: datetime = field(default_factory=_now)
    meta: dict = field(default_factory=dict)
```

Add `user_id` to the `Item` dataclass (after `response_id`):

```python
    user_id: Optional[str] = None
```

Add to the `ResponseStore` Protocol:

```python
    async def ensure_user(self, user_id: str, display_name: Optional[str] = None) -> User: ...
    async def get_user(self, user_id: str) -> Optional[User]: ...
```

- [ ] **Step 4: Add `UserRow` + `ConversationItem.user_id`** in `backend/app/models.py`

Add a table:

```python
class UserRow(SQLModel, table=True):
    __tablename__ = "users"
    id: str = Field(primary_key=True, max_length=64)
    display_name: Optional[str] = Field(default=None, max_length=200)
    created_at: datetime = Field(default_factory=_now)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
```

Add to `ConversationItem` (after `response_id`):

```python
    user_id: Optional[str] = Field(default=None, index=True, max_length=64)
```

- [ ] **Step 5: Add request aliases** in `backend/app/schemas.py`

Add fields (after `soul`):

```python
    user: Optional[str] = None
    safety_identifier: Optional[str] = None
```

Add a property to `ResponsesRequest`:

```python
    @property
    def resolved_user_id(self) -> Optional[str]:
        return self.user_id or self.user or self.safety_identifier
```

- [ ] **Step 6: Implement in `InMemoryStore`** (`backend/app/store/memory.py`)

Add `from app.store.base import User` to the imports. Add `self._users: Dict[str, User] = {}` in `__init__`. Add methods:

```python
    async def ensure_user(self, user_id, display_name=None) -> User:
        u = self._users.get(user_id)
        if u is not None:
            return u
        u = User(id=user_id, display_name=display_name)
        self._users[user_id] = u
        return u

    async def get_user(self, user_id) -> Optional[User]:
        return self._users.get(user_id)
```

- [ ] **Step 7: Implement in `SqlStore`** (`backend/app/store/sql.py`)

Add `UserRow` to the models import and `User` to the base import. Add methods:

```python
    async def ensure_user(self, user_id, display_name=None) -> User:
        async with AsyncSession(self._engine) as s:
            row = await s.get(UserRow, user_id)
            if row is None:
                row = UserRow(id=user_id, display_name=display_name)
                s.add(row)
                await s.commit()
                await s.refresh(row)
            return User(id=row.id, display_name=row.display_name,
                        created_at=row.created_at, meta=row.meta or {})

    async def get_user(self, user_id) -> Optional[User]:
        async with AsyncSession(self._engine) as s:
            row = await s.get(UserRow, user_id)
        if row is None:
            return None
        return User(id=row.id, display_name=row.display_name,
                    created_at=row.created_at, meta=row.meta or {})
```

- [ ] **Step 8: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_user_identity.py -q` → 3 pass.

- [ ] **Step 9: Full suite** — `cd backend && python -m pytest ../tests/app -q` → all pass (the new `Item.user_id`/`ConversationItem.user_id` default None; nothing reads them yet).

- [ ] **Step 10: Commit**

```bash
git add backend/app/store/base.py backend/app/models.py backend/app/schemas.py backend/app/store/memory.py backend/app/store/sql.py tests/app/test_user_identity.py
git commit -m "feat(app): User record + ResponsesRequest user/safety_identifier aliases (+ Item.user_id)"
```

---

## Task 2: Associate persisted items + conversation to the resolved user

**Files:** Modify `backend/app/store/memory.py`, `backend/app/store/sql.py` (persist `Item.user_id`), `backend/app/routes/responses.py`. Test: append to `tests/app/test_routes_responses.py`.

**Interfaces:**
- Consumes: `Item.user_id` (Task 1), `resolved_user_id`, `ensure_user`.
- Produces: `append_items` persists `Item.user_id`; the route stamps `resolved_user_id` on the user + store items, ensures the user, and uses `resolved_user_id` for `ensure_conversation`.

- [ ] **Step 1: Write the failing test** — append to `tests/app/test_routes_responses.py`

```python
def test_stored_items_and_conversation_carry_resolved_user_id():
    c = _client()
    body = c.post("/v1/responses", json={"input": "hello", "stream": False, "user": "u_alias"}).json()
    conv_id = body["conversation"]["id"]
    import asyncio

    async def _read():
        store = c.app.state.app_state.store
        conv = await store.get_conversation(conv_id)
        assert conv.user_id == "u_alias"           # alias resolved
        assert await store.get_user("u_alias") is not None  # user ensured
        items = await store.get_conversation_items(conv_id)
        assert items and all(it.user_id == "u_alias" for it in items)
    asyncio.run(_read())
```

- [ ] **Step 2: Run to verify failure** — `cd backend && python -m pytest ../tests/app/test_routes_responses.py -q` → FAIL (items have `user_id=None`; user not ensured; alias not used for conversation).

- [ ] **Step 3: Persist `Item.user_id` in `InMemoryStore.append_items`** (`backend/app/store/memory.py`)

The `Item` objects already carry `user_id`; `append_items` stores the `Item` objects as-is, so no change is needed there — confirm `append_items` appends the passed `Item` (it does). No code change for InMemoryStore append.

- [ ] **Step 4: Persist `Item.user_id` in `SqlStore.append_items`** (`backend/app/store/sql.py`)

In `append_items`, add `user_id=it.user_id` to the `ItemRow(...)` constructor, and in `_to_item` add `user_id=row.user_id`:

```python
            s.add(ItemRow(id=it.id, conversation_id=conversation_id, seq=n, type=it.type,
                          role=it.role, content=it.content, response_id=it.response_id,
                          user_id=it.user_id))
```
```python
def _to_item(row: ItemRow) -> Item:
    return Item(id=row.id, type=row.type, role=row.role, content=row.content or {},
                response_id=row.response_id, seq=row.seq, user_id=row.user_id)
```

- [ ] **Step 5: Stamp the user in the route** (`backend/app/routes/responses.py`)

In `_user_input_items`, accept + stamp a `user_id`:

```python
def _user_input_items(current_turn, response_id: str, user_id=None) -> list:
    text = current_turn.content if isinstance(current_turn.content, str) else ""
    return [Item(type="message", role="user", content={"text": text},
                 response_id=response_id, user_id=user_id)]
```

In `_persist`, resolve the user once, ensure it, stamp items, and use it for the conversation:

```python
async def _persist(state, request, current_turn, response_id, conversation_id,
                   store_items, status, usage, error=None):
    uid = request.resolved_user_id
    if uid:
        await state.store.ensure_user(uid)
    await state.store.ensure_conversation(
        conversation_id, user_id=uid, title=_title_from_turn(current_turn),
    )
    items = _user_input_items(current_turn, response_id, user_id=uid)
    for d in store_items:
        items.append(Item(type=d["type"], role=d.get("role"), content=d["content"],
                          response_id=response_id, user_id=uid))
    await state.store.append_items(conversation_id, items)
    await state.store.save_response(StoredResponse(
        id=response_id, conversation_id=conversation_id,
        model=request.model or state.default_model, status=status,
        usage=usage, error=error, previous_response_id=request.previous_response_id,
    ))
    await state.store.touch_conversation(conversation_id, last_response_id=response_id)
```

(Only the `uid`/`ensure_user`/`user_id=` additions change; the rest of `_persist` is as before.)

- [ ] **Step 6: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_routes_responses.py -q` → pass (incl. the new test).

- [ ] **Step 7: Full suite** — `cd backend && python -m pytest ../tests/app -q` → all pass.

- [ ] **Step 8: Commit**

```bash
git add backend/app/store/sql.py backend/app/routes/responses.py tests/app/test_routes_responses.py
git commit -m "feat(app): stamp resolved user_id on persisted items/conversation; ensure_user on stored turns"
```

---

## Task 3: `MemoryItem` store

**Files:** Modify `backend/app/store/base.py`, `backend/app/models.py`, `backend/app/store/memory.py`, `backend/app/store/sql.py`. Test: `tests/app/test_memory_store.py`.

**Interfaces:**
- Produces: `MemoryItem(id, user_id, text, kind="fact", source_response_id=None, status="active", created_at, updated_at)`; store methods `add_memory(item) -> MemoryItem`, `list_memories(user_id, limit=50) -> List[MemoryItem]` (active, newest `updated_at` first), `update_memory(memory_id, text) -> None`, `delete_memory(memory_id) -> None`, `delete_user_memories(user_id) -> None` — both stores; `MemoryRow` table.

- [ ] **Step 1: Write the failing test** — `tests/app/test_memory_store.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import pytest
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.db import make_engine, create_all
from app.store.base import MemoryItem


async def _mem():
    return InMemoryStore()


async def _sql():
    e = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(e)
    return SqlStore(e)


STORES = [_mem, _sql]


@pytest.mark.parametrize("make_store", STORES)
def test_add_list_update_delete(make_store):
    async def run():
        st = await make_store()
        a = await st.add_memory(MemoryItem(user_id="u1", text="likes tea"))
        await asyncio.sleep(0.01)
        b = await st.add_memory(MemoryItem(user_id="u1", text="lives in NYC"))
        await st.add_memory(MemoryItem(user_id="u2", text="other user"))
        mems = await st.list_memories("u1")
        assert [m.text for m in mems] == ["lives in NYC", "likes tea"]  # newest first
        await st.update_memory(b.id, "lives in Boston")
        assert (await st.list_memories("u1"))[0].text == "lives in Boston"
        await st.delete_memory(a.id)
        assert [m.text for m in await st.list_memories("u1")] == ["lives in Boston"]
    asyncio.run(run())


@pytest.mark.parametrize("make_store", STORES)
def test_limit_and_delete_user(make_store):
    async def run():
        st = await make_store()
        for i in range(5):
            await st.add_memory(MemoryItem(user_id="u1", text=f"f{i}"))
            await asyncio.sleep(0.001)
        assert len(await st.list_memories("u1", limit=3)) == 3
        await st.delete_user_memories("u1")
        assert await st.list_memories("u1") == []
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`/`ImportError: MemoryItem`.

- [ ] **Step 3: Add `MemoryItem` + Protocol** in `backend/app/store/base.py`

```python
@dataclass
class MemoryItem:
    user_id: str
    text: str
    kind: str = "fact"
    source_response_id: Optional[str] = None
    status: str = "active"
    id: str = field(default_factory=lambda: _uuid("mem"))
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
```

Add to `ResponseStore`:

```python
    async def add_memory(self, item: MemoryItem) -> MemoryItem: ...
    async def list_memories(self, user_id: str, limit: int = 50) -> List[MemoryItem]: ...
    async def update_memory(self, memory_id: str, text: str) -> None: ...
    async def delete_memory(self, memory_id: str) -> None: ...
    async def delete_user_memories(self, user_id: str) -> None: ...
```

- [ ] **Step 4: Add `MemoryRow`** in `backend/app/models.py`

```python
class MemoryRow(SQLModel, table=True):
    __tablename__ = "memory_items"
    id: str = Field(primary_key=True, max_length=64)
    user_id: str = Field(index=True, max_length=64)
    text: str = Field(sa_column=Column(JSON))  # store as text via JSON col for portability
    kind: str = Field(default="fact", max_length=32)
    source_response_id: Optional[str] = Field(default=None, max_length=64)
    status: str = Field(default="active", max_length=16)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
```

NOTE: use a normal string column for `text` if preferred; the JSON column avoids length limits across SQLite/Postgres. If `text` is declared `str` with `sa_column=Column(JSON)`, store/read the raw string (JSON-encodes a string fine). Simpler alternative: `text: str = Field(sa_column=Column("text", Text))` importing `from sqlalchemy import Text`. Pick one and keep `_to_mem` consistent.

- [ ] **Step 5: Implement in `InMemoryStore`** (`backend/app/store/memory.py`)

Add `from app.store.base import MemoryItem` + `from app.store.base import _now`. Add `self._memories: Dict[str, MemoryItem] = {}` in `__init__`. Methods:

```python
    async def add_memory(self, item: MemoryItem) -> MemoryItem:
        self._memories[item.id] = item
        return item

    async def list_memories(self, user_id: str, limit: int = 50) -> List[MemoryItem]:
        mems = [m for m in self._memories.values()
                if m.user_id == user_id and m.status == "active"]
        mems.sort(key=lambda m: m.updated_at, reverse=True)
        return mems[:limit]

    async def update_memory(self, memory_id: str, text: str) -> None:
        m = self._memories.get(memory_id)
        if m is not None:
            m.text = text
            m.updated_at = _now()

    async def delete_memory(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)

    async def delete_user_memories(self, user_id: str) -> None:
        for mid in [m.id for m in self._memories.values() if m.user_id == user_id]:
            self._memories.pop(mid, None)
```

- [ ] **Step 6: Implement in `SqlStore`** (`backend/app/store/sql.py`)

Add `MemoryRow` to the models import, `MemoryItem`/`_now` to the base import. Add a `_to_mem(row)` helper + methods:

```python
def _to_mem(row: MemoryRow) -> MemoryItem:
    return MemoryItem(id=row.id, user_id=row.user_id, text=row.text, kind=row.kind,
                      source_response_id=row.source_response_id, status=row.status,
                      created_at=row.created_at, updated_at=row.updated_at)
```
```python
    async def add_memory(self, item: MemoryItem) -> MemoryItem:
        async with AsyncSession(self._engine) as s:
            s.add(MemoryRow(id=item.id, user_id=item.user_id, text=item.text, kind=item.kind,
                            source_response_id=item.source_response_id, status=item.status,
                            created_at=item.created_at, updated_at=item.updated_at))
            await s.commit()
        return item

    async def list_memories(self, user_id: str, limit: int = 50) -> List[MemoryItem]:
        async with AsyncSession(self._engine) as s:
            rows = (await s.exec(
                select(MemoryRow).where(MemoryRow.user_id == user_id, MemoryRow.status == "active")
                .order_by(MemoryRow.updated_at.desc()).limit(limit))).all()
        return [_to_mem(r) for r in rows]

    async def update_memory(self, memory_id: str, text: str) -> None:
        async with AsyncSession(self._engine) as s:
            row = await s.get(MemoryRow, memory_id)
            if row is not None:
                row.text = text
                row.updated_at = _now()
                s.add(row)
                await s.commit()

    async def delete_memory(self, memory_id: str) -> None:
        async with AsyncSession(self._engine) as s:
            await s.exec(delete(MemoryRow).where(MemoryRow.id == memory_id))
            await s.commit()

    async def delete_user_memories(self, user_id: str) -> None:
        async with AsyncSession(self._engine) as s:
            await s.exec(delete(MemoryRow).where(MemoryRow.user_id == user_id))
            await s.commit()
```

- [ ] **Step 7: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_memory_store.py -q` → 4 pass (2×2). Then full suite green.

- [ ] **Step 8: Commit**

```bash
git add backend/app/store/base.py backend/app/models.py backend/app/store/memory.py backend/app/store/sql.py tests/app/test_memory_store.py
git commit -m "feat(store): MemoryItem + add/list/update/delete/delete_user_memories (both stores)"
```

---

## Task 4: Inject user memories into the system prompt

**Files:** Modify `backend/agent/soul.py`, `backend/app/builder.py`, `backend/app/config.py`. Test: `tests/app/test_memory_inject.py`.

**Interfaces:**
- Consumes: `list_memories` (Task 3), `resolved_user_id` (Task 1).
- Produces: `render_system_prompt(soul, *, tool_names, memories: Optional[List[str]] = None, extra="")` adds a `# Memory` section; `build_context` loads the user's memories (capped by `Settings.memory_inject_limit`, default 30) and injects them. `Settings.memory_inject_limit`.

- [ ] **Step 1: Write the failing test** — `tests/app/test_memory_inject.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.soul import DEFAULT_SOUL, render_system_prompt
from app.builder import build_context
from app.schemas import ResponsesRequest
from app.store.memory import InMemoryStore
from app.store.base import MemoryItem


def test_render_includes_memory_section_when_present():
    out = render_system_prompt(DEFAULT_SOUL, tool_names=[], memories=["likes tea", "in NYC"])
    assert "# Memory" in out and "likes tea" in out and "in NYC" in out
    assert "# Memory" not in render_system_prompt(DEFAULT_SOUL, tool_names=[], memories=[])


def test_build_context_injects_user_memories():
    async def run():
        st = InMemoryStore()
        await st.add_memory(MemoryItem(user_id="u1", text="prefers Python"))
        req = ResponsesRequest(model="m", input="hi", user="u1")
        ctx, _ = await build_context(req, st)
        assert "prefers Python" in ctx.system_prompt
        # no user -> no memory section
        ctx2, _ = await build_context(ResponsesRequest(model="m", input="hi"), st)
        assert "# Memory" not in ctx2.system_prompt
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — FAIL (no `memories` kwarg / no injection).

- [ ] **Step 3: Add the `# Memory` section** in `backend/agent/soul.py`

Change the signature and add the section (place it after `# Operating principles`, before `# Tools`):

```python
def render_system_prompt(
    soul: Soul, *, tool_names: List[str],
    memories: Optional[List[str]] = None, extra: str = ""
) -> str:
    ...
    parts.append("# Operating principles\n" + _bullets(soul.principles))

    if memories:
        parts.append(
            "# Memory\nWhat you remember about this user (use it naturally; "
            "do not recite it verbatim):\n" + _bullets(memories)
        )

    tools_section = "# Tools\n" + _TOOL_PROTOCOL
    ...
```

(Add `Optional` to the `typing` import if not present.)

- [ ] **Step 4: Inject in `build_context`** (`backend/app/builder.py`)

Add a constant and load memories before rendering:

```python
MEMORY_INJECT_LIMIT = 30
```

In `build_context`, after computing `effective_soul` and `tool_names`/`toolbox`, before `render_system_prompt`:

```python
    memories: List[str] = []
    uid = request.resolved_user_id
    if uid:
        memories = [m.text for m in await store.list_memories(uid, limit=MEMORY_INJECT_LIMIT)]
    system_prompt = render_system_prompt(
        effective_soul, tool_names=tool_names, memories=memories
    )
```

(Replace the existing `system_prompt = render_system_prompt(effective_soul, tool_names=tool_names)` line.)

- [ ] **Step 5: Add the setting** in `backend/app/config.py`

```python
    memory_inject_limit: int = 30
```

(Used by callers that want to override; `build_context` uses the module constant by default — keep both consistent, the constant is the default.)

- [ ] **Step 6: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_memory_inject.py -q` → 2 pass. Full suite green (existing builder/soul tests unaffected — `memories` defaults to None/[]).

- [ ] **Step 7: Commit**

```bash
git add backend/agent/soul.py backend/app/builder.py backend/app/config.py tests/app/test_memory_inject.py
git commit -m "feat(app): inject per-user memories into the system prompt (render_system_prompt + build_context)"
```

---

## Task 5: Extraction + consolidation pipeline (pure)

**Files:** Create `backend/app/memory.py`. Test: `tests/app/test_memory_extractor.py`.

**Interfaces:**
- Consumes: `MemoryItem` + store memory methods (Task 3).
- Produces:
  - `MemoryExtractor(complete: Callable[[str], Awaitable[str]])` with `async def extract(self, user_text, assistant_text, existing: List[MemoryItem]) -> List[dict]` returning ops `{op, text?, target_id?}`; tolerant JSON parsing (code-fence stripped; bad JSON → `[]`).
  - `async def apply_memory_ops(store, user_id, ops, source_response_id=None) -> None` — ADD/UPDATE/DELETE/NOOP.
  - `async def update_user_memory(store, user_id, user_text, assistant_text, complete, source_response_id=None) -> None` — list existing → extract → apply; fully guarded.
  - `make_complete(llm) -> Callable[[str], Awaitable[str]]` — wraps `LeanLLM.astream` single-shot.

- [ ] **Step 1: Write the failing test** — `tests/app/test_memory_extractor.py`

```python
import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.store.memory import InMemoryStore
from app.store.base import MemoryItem
from app.memory import MemoryExtractor, apply_memory_ops, update_user_memory


def _complete_returning(payload):
    async def complete(prompt):
        return payload
    return complete


def test_extract_parses_ops_and_tolerates_fences():
    ex = MemoryExtractor(_complete_returning(
        '```json\n[{"op":"ADD","text":"likes tea"},{"op":"NOOP"}]\n```'))
    ops = asyncio.run(ex.extract("I like tea", "Noted.", []))
    assert ops[0]["op"] == "ADD" and ops[0]["text"] == "likes tea"


def test_extract_bad_json_returns_empty():
    ex = MemoryExtractor(_complete_returning("not json at all"))
    assert asyncio.run(ex.extract("x", "y", [])) == []


def test_apply_ops_add_update_delete():
    async def run():
        st = InMemoryStore()
        existing = await st.add_memory(MemoryItem(user_id="u1", text="old"))
        await apply_memory_ops(st, "u1", [
            {"op": "ADD", "text": "new fact"},
            {"op": "UPDATE", "target_id": existing.id, "text": "updated"},
            {"op": "DELETE", "target_id": "missing"},  # no-op safe
            {"op": "NOOP"},
        ], source_response_id="resp_1")
        texts = {m.text for m in await st.list_memories("u1")}
        assert texts == {"new fact", "updated"}
    asyncio.run(run())


def test_update_user_memory_end_to_end_with_fake_complete():
    async def run():
        st = InMemoryStore()
        complete = _complete_returning('[{"op":"ADD","text":"works at Acme"}]')
        await update_user_memory(st, "u1", "I work at Acme", "Got it.", complete, "resp_9")
        mems = await st.list_memories("u1")
        assert [m.text for m in mems] == ["works at Acme"]
        assert mems[0].source_response_id == "resp_9"
    asyncio.run(run())


def test_update_user_memory_swallows_errors():
    async def run():
        st = InMemoryStore()
        async def boom(prompt):
            raise RuntimeError("llm down")
        # must not raise
        await update_user_memory(st, "u1", "x", "y", boom)
        assert await st.list_memories("u1") == []
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: app.memory`.

- [ ] **Step 3: Implement `backend/app/memory.py`**

```python
from __future__ import annotations
import json
import re
from typing import Awaitable, Callable, List, Optional
from loguru import logger
from app.store.base import MemoryItem

CompleteFn = Callable[[str], Awaitable[str]]

_EXTRACT_PROMPT = """You maintain a long-term memory about a specific user.
Given the latest exchange and the user's existing memories, decide what to change.
Return ONLY a JSON array of operations, each one of:
  {{"op":"ADD","text":"<new durable fact about the user>"}}
  {{"op":"UPDATE","target_id":"<id>","text":"<revised fact>"}}
  {{"op":"DELETE","target_id":"<id>"}}
  {{"op":"NOOP"}}
Only record durable, user-specific facts (preferences, identity, context). Do NOT
record transient chit-chat, the assistant's words, or anything sensitive the user
didn't volunteer. If nothing is worth changing, return [].

Existing memories:
{existing}

Latest exchange:
User: {user_text}
Assistant: {assistant_text}

JSON operations:"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    return m.group(1).strip() if m else s


class MemoryExtractor:
    def __init__(self, complete: CompleteFn):
        self._complete = complete

    async def extract(self, user_text: str, assistant_text: str,
                      existing: List[MemoryItem]) -> List[dict]:
        existing_str = "\n".join(f'- (id={m.id}) {m.text}' for m in existing) or "(none)"
        prompt = _EXTRACT_PROMPT.format(
            existing=existing_str, user_text=user_text, assistant_text=assistant_text)
        try:
            raw = await self._complete(prompt)
        except Exception:
            logger.exception("memory extract: completion failed")
            return []
        try:
            parsed = json.loads(_strip_fences(raw))
        except Exception:
            logger.warning(f"memory extract: non-JSON output: {raw[:200]!r}")
            return []
        if not isinstance(parsed, list):
            return []
        return [op for op in parsed if isinstance(op, dict) and "op" in op]


async def apply_memory_ops(store, user_id: str, ops: List[dict],
                           source_response_id: Optional[str] = None) -> None:
    for op in ops:
        kind = op.get("op")
        try:
            if kind == "ADD" and op.get("text"):
                await store.add_memory(MemoryItem(
                    user_id=user_id, text=op["text"], source_response_id=source_response_id))
            elif kind == "UPDATE" and op.get("target_id") and op.get("text"):
                await store.update_memory(op["target_id"], op["text"])
            elif kind == "DELETE" and op.get("target_id"):
                await store.delete_memory(op["target_id"])
            # NOOP / unknown: skip
        except Exception:
            logger.exception(f"memory apply: op failed: {op}")


async def update_user_memory(store, user_id: str, user_text: str, assistant_text: str,
                             complete: CompleteFn,
                             source_response_id: Optional[str] = None) -> None:
    """List existing -> extract -> apply. Fully guarded; never raises."""
    try:
        existing = await store.list_memories(user_id)
        ops = await MemoryExtractor(complete).extract(user_text, assistant_text, existing)
        if ops:
            await apply_memory_ops(store, user_id, ops, source_response_id)
    except Exception:
        logger.exception("update_user_memory failed")


def make_complete(llm) -> CompleteFn:
    """Single-shot completion over a LeanLLM-style astream (collects deltas)."""
    async def complete(prompt: str) -> str:
        chunks: List[str] = []
        async for ch in llm.astream(messages=[{"role": "user", "content": prompt}], tools=[]):
            delta = getattr(ch, "delta", "") or ""
            if delta:
                chunks.append(delta)
        return "".join(chunks)
    return complete
```

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_memory_extractor.py -q` → 5 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/memory.py tests/app/test_memory_extractor.py
git commit -m "feat(app): memory extraction + consolidation pipeline (extractor/apply/update, pure)"
```

---

## Task 6: Wire background extraction + management API

**Files:** Modify `backend/app/routes/responses.py`, `backend/app/config.py`, `backend/app/lean_main.py`; create `backend/app/routes/users.py`. Test: `tests/app/test_routes_memory.py`.

**Interfaces:**
- Consumes: `update_user_memory`/`make_complete` (Task 5); the router/llm from `AppState`.
- Produces:
  - After a stored turn (sync path + background run finally), if `Settings.memory_enabled` and `request.memory` and a resolved user_id, schedule `asyncio.create_task(update_user_memory(...))` with `complete = make_complete(<the turn's model client>)`. Guarded.
  - `ResponsesRequest.memory: bool = True`; `Settings.memory_enabled: bool = False`, `memory_model: str = ""`.
  - `GET /v1/users/{id}/memories`, `DELETE /v1/users/{id}/memories`, `DELETE /v1/users/{id}/memories/{memory_id}`.

- [ ] **Step 1: Write the failing test** — `tests/app/test_routes_memory.py`

```python
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.routes.responses import router as responses_router
from app.routes.users import router as users_router
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import CompletionUsage


class _MemoryLLM:
    """Echoes for the answer turn; returns a fixed ops JSON for the extraction prompt."""
    async def astream(self, messages, tools=None, **kwargs):
        prompt = ""
        for m in messages:
            if m.get("role") == "user":
                prompt = m.get("content") or ""
        usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        is_extract = "JSON operations:" in prompt

        async def gen():
            if is_extract:
                yield TextChunk(delta='[{"op":"ADD","text":"likes hiking"}]', usage=None)
            else:
                yield TextChunk(delta="ok", usage=None)
            yield TextChunk(delta="", usage=usage)
        return gen()


def _app(memory_enabled=True):
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(), llm=_MemoryLLM(), default_model="m",
        memory_enabled=memory_enabled,
    )
    app.include_router(responses_router)
    app.include_router(users_router)
    return app


def test_stored_turn_populates_user_memory_then_injects_it():
    with TestClient(_app()) as c:
        c.post("/v1/responses", json={"input": "I like hiking", "stream": False, "user": "u1"})
        # background extraction runs on the app loop; poll the memory API
        mems = []
        for _ in range(100):
            mems = c.get("/v1/users/u1/memories").json()["data"]
            if mems:
                break
            time.sleep(0.02)
        assert any(m["text"] == "likes hiking" for m in mems)


def test_memory_disabled_writes_nothing():
    with TestClient(_app(memory_enabled=False)) as c:
        c.post("/v1/responses", json={"input": "hi", "stream": False, "user": "u1"})
        time.sleep(0.2)
        assert c.get("/v1/users/u1/memories").json()["data"] == []


def test_delete_user_memories():
    with TestClient(_app()) as c:
        c.post("/v1/responses", json={"input": "I like hiking", "stream": False, "user": "u1"})
        for _ in range(100):
            if c.get("/v1/users/u1/memories").json()["data"]:
                break
            time.sleep(0.02)
        assert c.delete("/v1/users/u1/memories").status_code == 200
        assert c.get("/v1/users/u1/memories").json()["data"] == []
```

- [ ] **Step 2: Run to verify failure** — `AppState` has no `memory_enabled` / `users` router missing.

- [ ] **Step 3: Settings + request flag**

`backend/app/config.py` (add):
```python
    memory_enabled: bool = False
    memory_model: str = ""
```
`backend/app/schemas.py` (add to `ResponsesRequest`):
```python
    memory: bool = True
```
`backend/app/deps.py` — add to `AppState` (after `router`):
```python
    memory_enabled: bool = False
```

- [ ] **Step 4: Schedule the memory update** in `backend/app/routes/responses.py`

Add imports:
```python
import asyncio
from app.memory import update_user_memory, make_complete
```

Add a helper near `_persist`:
```python
def _schedule_memory_update(state, request, current_turn, store_items, response_id):
    uid = request.resolved_user_id
    if not (getattr(state, "memory_enabled", False) and request.memory and uid):
        return
    user_text = current_turn.content if isinstance(current_turn.content, str) else ""
    assistant_text = ""
    for d in store_items:
        if d.get("type") == "message" and d.get("role") == "assistant":
            assistant_text = (d.get("content") or {}).get("text", "")
    # Resolve the LLM used for memory extraction: memory_model (if routed) else the
    # request's model client (router) else state.llm.
    llm = None
    if state.router is not None:
        model_id = getattr(state, "memory_model", "") or request.model
        try:
            llm = state.router.get_llm(model_id)
        except Exception:
            llm = None
    if llm is None:
        llm = state.llm
    if llm is None:
        return
    asyncio.create_task(update_user_memory(
        state.store, uid, user_text, assistant_text, make_complete(llm),
        source_response_id=response_id,
    ))
```

Call it right after each `_persist(...)` in both the streaming `gen()` (after persist) and the sync path, and inside the background `_persist_run` (after `_persist`). Example for the sync path:
```python
    if request.store:
        await _persist(state, request, ctx.current_turn, response_id, conversation_id,
                       store_items, resp_dict["status"], resp_dict.get("usage"), resp_dict.get("error"))
        _schedule_memory_update(state, request, ctx.current_turn, store_items, response_id)
```
For the streaming path use `sink["items"]`; for `_persist_run` use `sink["items"]` likewise.

NOTE: `_schedule_memory_update` is fire-and-forget; `update_user_memory` is itself fully guarded, so a failed extraction never affects the turn. `getattr(state, "memory_enabled", False)` keeps it safe if an older `AppState` lacks the field.

- [ ] **Step 5: Implement `backend/app/routes/users.py`**

```python
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.deps import AppState, get_state

router = APIRouter()


def _iso(dt):
    return dt.isoformat() if dt is not None else None


@router.get("/v1/users/{user_id}/memories")
async def list_user_memories(user_id: str, limit: int = 100, state: AppState = Depends(get_state)):
    mems = await state.store.list_memories(user_id, limit=limit)
    return JSONResponse({"data": [
        {"id": m.id, "text": m.text, "kind": m.kind,
         "created_at": _iso(m.created_at), "updated_at": _iso(m.updated_at)}
        for m in mems
    ]})


@router.delete("/v1/users/{user_id}/memories")
async def clear_user_memories(user_id: str, state: AppState = Depends(get_state)):
    await state.store.delete_user_memories(user_id)
    return JSONResponse({"deleted": True})


@router.delete("/v1/users/{user_id}/memories/{memory_id}")
async def delete_user_memory(user_id: str, memory_id: str, state: AppState = Depends(get_state)):
    await state.store.delete_memory(memory_id)
    return JSONResponse({"id": memory_id, "deleted": True})
```

- [ ] **Step 6: Wire at boot** in `backend/app/lean_main.py`

Add `from app.routes.users import router as users_router`; pass `memory_enabled=settings.memory_enabled` to `AppState(...)`; `app.include_router(users_router)`.

- [ ] **Step 7: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_routes_memory.py -q` → 3 pass.

- [ ] **Step 8: Full suite + boot/isolation gates** — `cd backend && python -m pytest ../tests/app -q` → all pass.

- [ ] **Step 9: Commit**

```bash
git add backend/app/routes/responses.py backend/app/config.py backend/app/schemas.py backend/app/deps.py backend/app/routes/users.py backend/app/lean_main.py tests/app/test_routes_memory.py
git commit -m "feat(app): background user-memory extraction after stored turns + memory management API"
```

---

## Self-Review (against the spec)

- **Identity + OpenAI alignment** (`user`/`safety_identifier` → `resolved_user_id`, `User` record, `ensure_user`) → Tasks 1–2.
- **Messages associated to user** (`Item.user_id`/`ConversationItem.user_id`, stamped in the route; conversation `user_id` from the resolved id) → Tasks 1–2.
- **User memory store** (`MemoryItem` + CRUD, both stores) → Task 3.
- **Read/inject** (`render_system_prompt(memories=)` + `build_context` capped injection) → Task 4.
- **Write/extract** (mem0-style extractor + ADD/UPDATE/DELETE/NOOP consolidation; async, guarded, gated) → Tasks 5–6.
- **Management API** (list/clear/delete-one) → Task 6.
- **Gating/back-compat** (`memory_enabled` default off, per-request `memory`, `AppState`/route degrade when absent; existing tests green) → Tasks 4–6.
- **v2 deferrals honored** (no system blocks, no vector retrieval, no memory tools, no upstream forwarding, no UI).

No placeholders; signatures (`resolved_user_id`, `ensure_user`/`get_user`, `MemoryItem`, `add_memory`/`list_memories`/`update_memory`/`delete_memory`/`delete_user_memories`, `render_system_prompt(..., memories=)`, `MemoryExtractor.extract`, `apply_memory_ops`, `update_user_memory`, `make_complete`) are consistent across tasks. Both stores kept in lockstep; memory is async + exception-guarded so it can never break a turn.
