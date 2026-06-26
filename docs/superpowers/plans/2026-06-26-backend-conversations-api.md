# Backend Conversations API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the lean service's `conversations` table real and listable so a frontend sidebar has a data source — add `title`/`last_response_id` columns, `user_id` on the request, conversation persistence on stored turns, store list/get/delete methods, and `GET/GET-one/DELETE /v1/conversations` routes that reconstruct UI turns from the item log.

**Architecture:** Builds on the already-implemented lean service (`backend/app/`, `/v1/responses` with `reasoning.summary` streaming). Today the `conversations` table is never written: `build_context` mints a `conversation_id` and the route appends items + a `StoredResponse` under it, but no `Conversation` row is created. This plan makes conversations first-class rows (created/touched on stored turns), adds store query methods, a pure item→UI-message grouping function, and a new `routes/conversations.py` wired into `lean_main.py`. Everything stays import-lean (no RAG/llamaindex/tokenizer deps) and green against both `InMemoryStore` and `SqlStore`.

**Tech Stack:** Python 3.11, FastAPI + Starlette, SQLModel + SQLAlchemy async, pydantic, pytest. Tests run from `backend/` with paths relative to it (`cd backend && python -m pytest ../tests/app/... -q`). Async tests use the existing repo pattern: a plain `def test_*` that calls `asyncio.run(run())` over an inner `async def run()`.

**Reference spec:** `docs/superpowers/specs/2026-06-26-new-agent-chat-frontend-design.md` ("Backend additions"). Branch: `personal/yfei/agent-core`. The `newfrontend/` SPA is the SEPARATE next plan (`2026-06-26-newfrontend-vite-spa.md`) and depends on this one.

## Global Constraints

- **Import-lean:** the lean service must boot with zero RAG/llamaindex/tokenizer/trace deps. Do not add imports outside `app/`, `api/protocol/`, `agent/`, `common/`, and stdlib + already-used third parties (FastAPI, SQLModel, SQLAlchemy, pydantic, openai). The boot/isolation tests `tests/app/test_lean_main_boot.py` and `tests/app/test_lean_import_isolation.py` must stay green.
- **No DB migration:** new columns are added to the SQLModel and picked up by `create_all` only (the spec explicitly says "no migration"). Do not touch `alembic/`.
- **Both stores stay in lockstep:** every `ResponseStore` method added to the Protocol in `app/store/base.py` MUST be implemented in BOTH `app/store/memory.py` (`InMemoryStore`) and `app/store/sql.py` (`SqlStore`), and tested against both.
- **Store-item content shapes are a shared contract** with the serializer (`api/protocol/responses_serializer.py`) and `app/builder.py`. The shapes are: user/assistant `message` → `{"text": str}`; `reasoning` → `{"text": str}`; `function_call` → `{"call_id","name","arguments"}`; `function_call_output` → `{"call_id","output"}`. Every item of one turn shares one `response_id`. Do not change these shapes.
- **Routes use the `/v1/...` prefix** and return `JSONResponse`, matching `app/routes/responses.py`.
- Run the whole suite at the end of every task: `cd backend && python -m pytest ../tests/app -q`.

---

## Key existing contracts (verified, do not re-derive)

- **`app/store/base.py`** — dataclasses `Item(type, content, role=None, response_id=None, id=<auto>, seq=0)`, `Conversation(id=<auto>, user_id=None)`, `StoredResponse(id, model, status, conversation_id=None, previous_response_id=None, usage=None, error=None)`; helper `new_conversation_id()`. `ResponseStore` is a `typing.Protocol` listing the async store methods.
- **`app/store/memory.py`** — `InMemoryStore` holds `self._convs: Dict[str, Conversation]`, `self._items: Dict[str, List[Item]]`, `self._responses: Dict[str, StoredResponse]`. `append_items` assigns `seq = max(seq)+1`.
- **`app/store/sql.py`** — `SqlStore(engine)` opens `AsyncSession(self._engine)` per call; helper `_to_item(row)`. Uses `select(...)` from `sqlmodel`, `func`/`delete` from `sqlalchemy`.
- **`app/models.py`** — table models `Conversation` (`id`, `user_id` indexed, `created_at`, `updated_at`, `meta` JSON col named `"metadata"`), `ConversationItem` (`id`, `conversation_id` indexed, `seq` BigInteger, `type`, `role`, `content` JSON, `response_id` indexed, `created_at`), `ResponseRow` (`id`, `conversation_id`, `previous_response_id`, `model`, `status`, `usage` JSON, `error` JSON, `created_at`, `meta`). `_now()` returns `datetime.now(timezone.utc)`.
- **`app/routes/responses.py`** — `POST /v1/responses` builds context, runs the agent, serializes (stream or sync), and on `store=true` calls `_persist(...)` which `append_items` (user item + serializer store items, all sharing `response_id`) then `save_response(...)`. `build_context` returns `(ctx, conversation_id)` with `conversation_id` ALWAYS non-None.
- **`app/builder.py`** — `build_context(request, store)`; `items_to_messages(items)`; `_input_to_turn(req_input)`. `_item_text(content)` reads `content["text"]`.
- **`app/deps.py`** — `AppState(store, llm, default_model, context_window=110000, max_output_tokens=8000)`, `make_agent()`, `get_state(request)`.
- **`app/lean_main.py`** — builds `AppState` in lifespan and `app.include_router(responses_router)` + `chat_router`.
- **Serializer store-item shapes** (`api/protocol/responses_serializer.py`): `{"type":"message","role":"assistant","content":{"text":...}}`, `{"type":"reasoning","role":None,"content":{"text":...}}`, `{"type":"function_call","role":None,"content":{"call_id","name","arguments"}}`, `{"type":"function_call_output","role":None,"content":{"call_id","output"}}`. The route prepends the user item `{"type":"message","role":"user","content":{"text":...}}`.
- **Test scaffolding pattern** (all existing `tests/app/*`): first two lines
  ```python
  import sys, os, asyncio
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
  ```
  In-memory SqlStore engine: `make_engine("sqlite+aiosqlite:///:memory:")` then `await create_all(engine)`. Route tests assemble a `FastAPI()` in-test, set `app.state.app_state = AppState(store=..., llm=_EchoLLM(), default_model="m")`, `app.include_router(router)`, and use `fastapi.testclient.TestClient`. The echo LLM is defined in `tests/app/test_routes_responses.py` — copy it where needed.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/app/store/base.py` (modify) | `Conversation` dataclass gains `title`/`last_response_id`/`created_at`/`updated_at`; `ResponseStore` Protocol gains `ensure_conversation`/`touch_conversation`/`list_conversations`/`get_conversation`/`delete_conversation`/`list_conversation_responses` |
| `backend/app/models.py` (modify) | `Conversation` table gains nullable `title` + `last_response_id` columns |
| `backend/app/schemas.py` (modify) | `ResponsesRequest` gains optional `user_id` |
| `backend/app/store/memory.py` (modify) | `InMemoryStore` implements the six new methods |
| `backend/app/store/sql.py` (modify) | `SqlStore` implements the six new methods |
| `backend/app/routes/responses.py` (modify) | persist path: `ensure_conversation` before append, `touch_conversation` after save; thread `user_id` + first-user-message title |
| `backend/app/conversations_view.py` (new) | PURE `group_conversation_messages(items, responses)` → UI message dicts |
| `backend/app/routes/conversations.py` (new) | `GET /v1/conversations`, `GET /v1/conversations/{id}`, `DELETE /v1/conversations/{id}` |
| `backend/app/lean_main.py` (modify) | mount `conversations_router` |
| `tests/app/test_store_conversations.py` (new) | store methods against InMemoryStore + SqlStore |
| `tests/app/test_conversations_view.py` (new) | grouping algorithm unit tests |
| `tests/app/test_routes_conversations.py` (new) | endpoint tests via TestClient |
| `tests/app/test_routes_responses.py` (modify) | assert a stored turn creates a listable conversation with title + user_id |

---

## Task 1: Data model — `title`/`last_response_id`/`user_id`

Add the new columns/fields with no behavior change yet, so the rest of the plan has types to depend on.

**Files:**
- Modify: `backend/app/store/base.py` (the `Conversation` dataclass)
- Modify: `backend/app/models.py` (the `Conversation` table)
- Modify: `backend/app/schemas.py` (`ResponsesRequest`)
- Test: `tests/app/test_models.py` (append one test) — verify existing first

**Interfaces:**
- Produces: `Conversation(id, user_id=None, title=None, last_response_id=None, created_at=<utc now>, updated_at=<utc now>)` dataclass; SQLModel `Conversation` table with nullable `title: Optional[str]` and `last_response_id: Optional[str]`; `ResponsesRequest.user_id: Optional[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/app/test_models.py`:

```python
def test_conversation_table_has_title_and_last_response_id():
    from app.models import Conversation as ConvRow
    cols = ConvRow.__table__.columns.keys()
    assert "title" in cols and "last_response_id" in cols


def test_conversation_dataclass_defaults():
    from app.store.base import Conversation
    c = Conversation()
    assert c.title is None and c.last_response_id is None
    assert c.created_at is not None and c.updated_at is not None


def test_responses_request_accepts_user_id():
    from app.schemas import ResponsesRequest
    req = ResponsesRequest(model="m", input="hi", user_id="u_123")
    assert req.user_id == "u_123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_models.py -q`
Expected: FAIL — `AssertionError` on `"title" in cols` (and/or `AttributeError`/`TypeError` for the dataclass/request fields).

- [ ] **Step 3: Add the dataclass fields** in `backend/app/store/base.py`

Add the import at the top (after `import uuid`):

```python
from datetime import datetime, timezone
```

Add a module helper above the dataclasses (next to `new_conversation_id`):

```python
def _now() -> datetime:
    return datetime.now(timezone.utc)
```

Replace the `Conversation` dataclass with:

```python
@dataclass
class Conversation:
    id: str = field(default_factory=lambda: _uuid("conv"))
    user_id: Optional[str] = None
    title: Optional[str] = None
    last_response_id: Optional[str] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
```

- [ ] **Step 4: Add the SQLModel columns** in `backend/app/models.py`

In the `Conversation` table class, add two fields after `updated_at`:

```python
    title: Optional[str] = Field(default=None, max_length=200)
    last_response_id: Optional[str] = Field(default=None, max_length=64)
```

- [ ] **Step 5: Add the request field** in `backend/app/schemas.py`

Add to `ResponsesRequest` after `conversation`:

```python
    user_id: Optional[str] = None
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_models.py -q`
Expected: PASS (all tests, including the three new ones).

- [ ] **Step 7: Run the full app suite (no regressions)**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (existing behavior unchanged — `create_conversation` still works; new columns default to NULL).

- [ ] **Step 8: Commit**

```bash
git add backend/app/store/base.py backend/app/models.py backend/app/schemas.py tests/app/test_models.py
git commit -m "feat(app): Conversation gains title/last_response_id + ResponsesRequest.user_id"
```

---

## Task 2: Store methods — ensure/touch/list/get/delete

Add the six conversation methods to the `ResponseStore` Protocol and implement them identically in both stores. These power the persist path (Task 3) and the routes (Task 5).

**Files:**
- Modify: `backend/app/store/base.py` (`ResponseStore` Protocol)
- Modify: `backend/app/store/memory.py` (`InMemoryStore`)
- Modify: `backend/app/store/sql.py` (`SqlStore`)
- Test: `tests/app/test_store_conversations.py` (new)

**Interfaces:**
- Consumes: `Conversation` / `Item` / `StoredResponse` dataclasses (Task 1); existing `append_items`/`save_response`/`get_conversation_items`.
- Produces (added to `ResponseStore` and both stores):
  - `async def ensure_conversation(conversation_id: str, user_id: Optional[str], title: Optional[str]) -> Conversation` — create-if-absent; idempotent; NEVER overwrites an existing row's `title` or `user_id`.
  - `async def touch_conversation(conversation_id: str, last_response_id: str) -> None` — set `last_response_id` and bump `updated_at`; no-op if absent.
  - `async def list_conversations(user_id: Optional[str], limit: int = 50, offset: int = 0) -> List[Conversation]` — filter by `user_id` when given; newest `updated_at` first; sliced by offset/limit.
  - `async def get_conversation(conversation_id: str) -> Optional[Conversation]`.
  - `async def list_conversation_responses(conversation_id: str) -> List[StoredResponse]` — all responses for the conversation (any order; the view sorts).
  - `async def delete_conversation(conversation_id: str) -> None` — remove the conversation row, its items, and its responses.

- [ ] **Step 1: Write the failing test** — `tests/app/test_store_conversations.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import pytest
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.db import make_engine, create_all
from app.store.base import Item, StoredResponse


async def _mem():
    return InMemoryStore()


async def _sql():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    return SqlStore(engine)


# Run every test against both store backends.
STORES = [_mem, _sql]


def _run(coro_factory):
    asyncio.run(coro_factory())


@pytest.mark.parametrize("make_store", STORES)
def test_ensure_conversation_is_idempotent_and_keeps_title(make_store):
    async def run():
        st = await make_store()
        c1 = await st.ensure_conversation("conv_1", user_id="u1", title="first title")
        assert c1.id == "conv_1" and c1.title == "first title" and c1.user_id == "u1"
        # second call must NOT overwrite title or user_id
        c2 = await st.ensure_conversation("conv_1", user_id="u2", title="second title")
        assert c2.title == "first title" and c2.user_id == "u1"
        got = await st.get_conversation("conv_1")
        assert got is not None and got.title == "first title"
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_touch_sets_last_response_id_and_bumps_updated_at(make_store):
    async def run():
        st = await make_store()
        c = await st.ensure_conversation("conv_1", user_id=None, title="t")
        before = c.updated_at
        await asyncio.sleep(0.01)
        await st.touch_conversation("conv_1", last_response_id="resp_9")
        got = await st.get_conversation("conv_1")
        assert got.last_response_id == "resp_9"
        assert got.updated_at >= before
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_list_conversations_filters_by_user_and_orders_newest_first(make_store):
    async def run():
        st = await make_store()
        await st.ensure_conversation("conv_a", user_id="u1", title="a")
        await asyncio.sleep(0.01)
        await st.ensure_conversation("conv_b", user_id="u1", title="b")
        await asyncio.sleep(0.01)
        await st.ensure_conversation("conv_c", user_id="u2", title="c")
        # touch conv_a so it becomes the most-recently-updated for u1
        await asyncio.sleep(0.01)
        await st.touch_conversation("conv_a", last_response_id="resp_1")
        u1 = await st.list_conversations(user_id="u1", limit=50, offset=0)
        assert [c.id for c in u1] == ["conv_a", "conv_b"]
        # pagination
        page = await st.list_conversations(user_id="u1", limit=1, offset=1)
        assert [c.id for c in page] == ["conv_b"]
        # no filter -> all three, newest first
        allc = await st.list_conversations(user_id=None, limit=50, offset=0)
        assert allc[0].id == "conv_a" and len(allc) == 3
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_list_conversation_responses(make_store):
    async def run():
        st = await make_store()
        await st.ensure_conversation("conv_1", user_id=None, title="t")
        await st.save_response(StoredResponse(id="resp_1", conversation_id="conv_1", model="m", status="completed"))
        await st.save_response(StoredResponse(id="resp_2", conversation_id="conv_1", model="m", status="failed"))
        await st.save_response(StoredResponse(id="resp_x", conversation_id="conv_other", model="m", status="completed"))
        rs = await st.list_conversation_responses("conv_1")
        assert {r.id for r in rs} == {"resp_1", "resp_2"}
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_delete_conversation_cascades(make_store):
    async def run():
        st = await make_store()
        await st.ensure_conversation("conv_1", user_id=None, title="t")
        await st.append_items("conv_1", [Item(type="message", role="user", content={"text": "q"}, response_id="resp_1")])
        await st.save_response(StoredResponse(id="resp_1", conversation_id="conv_1", model="m", status="completed"))
        await st.delete_conversation("conv_1")
        assert await st.get_conversation("conv_1") is None
        assert await st.get_conversation_items("conv_1") == []
        assert await st.get_response("resp_1") is None
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_get_conversation_absent_returns_none(make_store):
    async def run():
        st = await make_store()
        assert await st.get_conversation("nope") is None
    _run(run)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_store_conversations.py -q`
Expected: FAIL — `AttributeError: 'InMemoryStore' object has no attribute 'ensure_conversation'`.

- [ ] **Step 3: Extend the Protocol** in `backend/app/store/base.py`

Add these signatures inside `class ResponseStore(Protocol):` (after `resolve_history`):

```python
    async def ensure_conversation(self, conversation_id: str, user_id: Optional[str],
                                  title: Optional[str]) -> Conversation: ...
    async def touch_conversation(self, conversation_id: str, last_response_id: str) -> None: ...
    async def list_conversations(self, user_id: Optional[str], limit: int = 50,
                                 offset: int = 0) -> List[Conversation]: ...
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]: ...
    async def list_conversation_responses(self, conversation_id: str) -> List[StoredResponse]: ...
    async def delete_conversation(self, conversation_id: str) -> None: ...
```

- [ ] **Step 4: Implement in `InMemoryStore`** (`backend/app/store/memory.py`)

Add the import at the top (after the existing imports):

```python
from app.store.base import _now
```

Add these methods to `InMemoryStore`:

```python
    async def ensure_conversation(self, conversation_id, user_id, title) -> Conversation:
        existing = self._convs.get(conversation_id)
        if existing is not None:
            return existing
        conv = Conversation(id=conversation_id, user_id=user_id, title=title)
        self._convs[conversation_id] = conv
        self._items.setdefault(conversation_id, [])
        return conv

    async def touch_conversation(self, conversation_id, last_response_id) -> None:
        conv = self._convs.get(conversation_id)
        if conv is None:
            return
        conv.last_response_id = last_response_id
        conv.updated_at = _now()

    async def list_conversations(self, user_id, limit=50, offset=0) -> List[Conversation]:
        convs = [c for c in self._convs.values()
                 if user_id is None or c.user_id == user_id]
        convs.sort(key=lambda c: c.updated_at, reverse=True)
        return convs[offset:offset + limit]

    async def get_conversation(self, conversation_id) -> Optional[Conversation]:
        return self._convs.get(conversation_id)

    async def list_conversation_responses(self, conversation_id) -> List[StoredResponse]:
        return [r for r in self._responses.values()
                if r.conversation_id == conversation_id]

    async def delete_conversation(self, conversation_id) -> None:
        self._convs.pop(conversation_id, None)
        self._items.pop(conversation_id, None)
        for rid in [r.id for r in self._responses.values()
                    if r.conversation_id == conversation_id]:
            self._responses.pop(rid, None)
```

NOTE: `create_conversation` (the legacy minting method) does not create a row through `ensure_conversation`; leave it untouched. `ensure_conversation` is the path the route now uses.

- [ ] **Step 5: Implement in `SqlStore`** (`backend/app/store/sql.py`)

Add imports at the top (extend the existing lines):

```python
from app.store.base import Conversation, Item, StoredResponse, _now
```

Add a row→dataclass helper next to `_to_item`:

```python
def _to_conv(row: ConvRow) -> Conversation:
    return Conversation(id=row.id, user_id=row.user_id, title=row.title,
                        last_response_id=row.last_response_id,
                        created_at=row.created_at, updated_at=row.updated_at)
```

Add these methods to `SqlStore`:

```python
    async def ensure_conversation(self, conversation_id, user_id, title) -> Conversation:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ConvRow, conversation_id)
            if row is None:
                row = ConvRow(id=conversation_id, user_id=user_id, title=title)
                s.add(row)
                await s.commit()
                await s.refresh(row)
            return _to_conv(row)

    async def touch_conversation(self, conversation_id, last_response_id) -> None:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ConvRow, conversation_id)
            if row is None:
                return
            row.last_response_id = last_response_id
            row.updated_at = _now()
            s.add(row)
            await s.commit()

    async def list_conversations(self, user_id, limit=50, offset=0) -> List[Conversation]:
        async with AsyncSession(self._engine) as s:
            stmt = select(ConvRow)
            if user_id is not None:
                stmt = stmt.where(ConvRow.user_id == user_id)
            stmt = stmt.order_by(ConvRow.updated_at.desc()).offset(offset).limit(limit)
            rows = (await s.exec(stmt)).all()
        return [_to_conv(r) for r in rows]

    async def get_conversation(self, conversation_id) -> Optional[Conversation]:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ConvRow, conversation_id)
        return _to_conv(row) if row is not None else None

    async def list_conversation_responses(self, conversation_id) -> List[StoredResponse]:
        async with AsyncSession(self._engine) as s:
            rows = (await s.exec(
                select(ResponseRow).where(
                    ResponseRow.conversation_id == conversation_id))).all()
        return [StoredResponse(id=r.id, model=r.model, status=r.status,
                               conversation_id=r.conversation_id,
                               previous_response_id=r.previous_response_id,
                               usage=r.usage, error=r.error) for r in rows]

    async def delete_conversation(self, conversation_id) -> None:
        async with AsyncSession(self._engine) as s:
            await s.exec(delete(ItemRow).where(ItemRow.conversation_id == conversation_id))
            await s.exec(delete(ResponseRow).where(ResponseRow.conversation_id == conversation_id))
            await s.exec(delete(ConvRow).where(ConvRow.id == conversation_id))
            await s.commit()
```

NOTE: `ConvRow`, `ItemRow`, `ResponseRow` are already imported at the top of `sql.py` (`from app.models import Conversation as ConvRow, ConversationItem as ItemRow, ResponseRow`). `ResponseRow.updated_at` does not exist — order responses is not needed here (the view sorts by item seq), so no ORDER BY on responses.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_store_conversations.py -q`
Expected: PASS (12 — 6 tests × 2 stores).

- [ ] **Step 7: Run the full app suite**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/store/base.py backend/app/store/memory.py backend/app/store/sql.py tests/app/test_store_conversations.py
git commit -m "feat(store): conversation ensure/touch/list/get/delete on both stores"
```

---

## Task 3: Persist conversations on stored turns

Wire conversation creation/touch into the `/v1/responses` persist path so every stored turn produces a real, listable `Conversation` row with a title from the first user message and the request's `user_id`.

**Files:**
- Modify: `backend/app/routes/responses.py` (the `_persist` helper)
- Test: `tests/app/test_routes_responses.py` (append assertions)

**Interfaces:**
- Consumes: `store.ensure_conversation`, `store.touch_conversation` (Task 2); `ResponsesRequest.user_id` (Task 1).
- Produces: after a `store=true` turn, `store.get_conversation(conversation_id)` returns a row whose `title` = first user message trimmed to ≤80 chars (set only on creation) and `last_response_id` = this turn's `response_id`; `user_id` from the request.

- [ ] **Step 1: Write the failing test** — append to `tests/app/test_routes_responses.py`

```python
def test_stored_turn_creates_listable_conversation_with_title_and_user():
    c = _client()
    body = c.post("/v1/responses", json={
        "input": "what is the capital of France?",
        "stream": False,
        "user_id": "u_42",
    }).json()
    conv_id = body["conversation"]["id"]
    # the conversation is now a real, retrievable row via the responses' app_state store
    # (assert through a second turn that continues it AND via a direct store read)
    import asyncio
    from app.deps import get_state

    async def _read():
        # reach the store the TestClient app is using
        state = c.app.state.app_state
        conv = await state.store.get_conversation(conv_id)
        assert conv is not None
        assert conv.title == "what is the capital of France?"
        assert conv.user_id == "u_42"
        assert conv.last_response_id == body["id"]
    asyncio.run(_read())


def test_title_is_trimmed_to_80_chars():
    c = _client()
    long_q = "x" * 200
    body = c.post("/v1/responses", json={"input": long_q, "stream": False}).json()
    conv_id = body["conversation"]["id"]
    import asyncio

    async def _read():
        conv = await c.app.state.app_state.store.get_conversation(conv_id)
        assert len(conv.title) == 80
    asyncio.run(_read())


def test_store_false_creates_no_conversation():
    c = _client()
    body = c.post("/v1/responses", json={"input": "ephemeral", "stream": False, "store": False}).json()
    conv_id = body["conversation"]["id"]
    import asyncio

    async def _read():
        assert await c.app.state.app_state.store.get_conversation(conv_id) is None
    asyncio.run(_read())
```

NOTE: `TestClient` exposes the underlying app at `c.app`, so `c.app.state.app_state.store` is the same `InMemoryStore` the route used. `_client()` already exists in this file.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_routes_responses.py -q`
Expected: FAIL — `get_conversation` returns `None` (the row is never created today).

- [ ] **Step 3: Add a title helper + update `_persist`** in `backend/app/routes/responses.py`

Add a module-level helper near `_user_input_items`:

```python
def _title_from_turn(current_turn) -> str:
    text = current_turn.content if isinstance(current_turn.content, str) else ""
    return text.strip()[:80]
```

In `_persist`, BEFORE the existing `await state.store.append_items(...)` call, ensure the conversation row; AFTER `save_response`, touch it. The new `_persist` body:

```python
async def _persist(
    state: AppState,
    request: ResponsesRequest,
    current_turn,
    response_id: str,
    conversation_id: str,
    store_items: list,
    status: str,
    usage: dict | None,
    error: dict | None = None,
):
    # Create the conversation row if absent (title from the first user message,
    # set only on creation; user_id from the request). Idempotent on later turns.
    await state.store.ensure_conversation(
        conversation_id,
        user_id=request.user_id,
        title=_title_from_turn(current_turn),
    )
    items = _user_input_items(current_turn, response_id)
    for d in store_items:
        items.append(
            Item(
                type=d["type"],
                role=d.get("role"),
                content=d["content"],
                response_id=response_id,
            )
        )
    await state.store.append_items(conversation_id, items)
    await state.store.save_response(
        StoredResponse(
            id=response_id,
            conversation_id=conversation_id,
            model=request.model or state.default_model,
            status=status,
            usage=usage,
            error=error,
            previous_response_id=request.previous_response_id,
        )
    )
    # Bump updated_at + record the latest response as the continuation anchor.
    await state.store.touch_conversation(conversation_id, last_response_id=response_id)
```

NOTE: `_persist` is called from both the streaming `gen()` and the sync path; both already guard on `request.store`, so no conversation is written when `store=false`. Nothing else in the route changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_routes_responses.py -q`
Expected: PASS (existing tests + the three new ones).

- [ ] **Step 5: Run the full app suite**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/routes/responses.py tests/app/test_routes_responses.py
git commit -m "feat(app): persist Conversation rows on stored turns (title + user_id + touch)"
```

---

## Task 4: Item → UI-message grouping (pure function)

The trickiest read-side logic, isolated as a pure, fully unit-tested function so the route (Task 5) is thin.

**Files:**
- Create: `backend/app/conversations_view.py`
- Test: `tests/app/test_conversations_view.py`

**Interfaces:**
- Consumes: `Item` (with `type`/`role`/`content`/`response_id`/`seq`) and `StoredResponse` (with `id`/`status`/`previous_response_id`) from `app/store/base.py`.
- Produces: `group_conversation_messages(items: List[Item], responses: List[StoredResponse]) -> List[dict]`. Output messages:
  - user: `{"role": "user", "text": str, "response_id": str}`
  - assistant: `{"role": "assistant", "text": str, "reasoning": Optional[str], "response_id": str, "previous_response_id": Optional[str], "status": str}`

- [ ] **Step 1: Write the failing test** — `tests/app/test_conversations_view.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.store.base import Item, StoredResponse
from app.conversations_view import group_conversation_messages


def _items(seq_triples):
    # seq_triples: list of (type, role, content, response_id)
    out = []
    for i, (t, role, content, rid) in enumerate(seq_triples):
        out.append(Item(type=t, role=role, content=content, response_id=rid, seq=i))
    return out


def test_single_text_turn():
    items = _items([
        ("message", "user", {"text": "hi"}, "resp_1"),
        ("message", "assistant", {"text": "hello"}, "resp_1"),
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="completed",
                            conversation_id="c", previous_response_id=None)]
    msgs = group_conversation_messages(items, resps)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0] == {"role": "user", "text": "hi", "response_id": "resp_1"}
    a = msgs[1]
    assert a["text"] == "hello" and a["reasoning"] is None
    assert a["status"] == "completed" and a["previous_response_id"] is None
    assert a["response_id"] == "resp_1"


def test_reasoning_surfaced_on_assistant():
    items = _items([
        ("message", "user", {"text": "q"}, "resp_1"),
        ("reasoning", None, {"text": "let me think"}, "resp_1"),
        ("message", "assistant", {"text": "a"}, "resp_1"),
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="completed", conversation_id="c")]
    msgs = group_conversation_messages(items, resps)
    assert msgs[1]["reasoning"] == "let me think"


def test_multi_turn_chronological_with_previous_response_id():
    items = _items([
        ("message", "user", {"text": "q1"}, "resp_1"),
        ("message", "assistant", {"text": "a1"}, "resp_1"),
        ("message", "user", {"text": "q2"}, "resp_2"),
        ("message", "assistant", {"text": "a2"}, "resp_2"),
    ])
    resps = [
        StoredResponse(id="resp_1", model="m", status="completed", conversation_id="c"),
        StoredResponse(id="resp_2", model="m", status="completed", conversation_id="c",
                       previous_response_id="resp_1"),
    ]
    msgs = group_conversation_messages(items, resps)
    assert [m.get("text") for m in msgs] == ["q1", "a1", "q2", "a2"]
    assert msgs[3]["previous_response_id"] == "resp_1"


def test_failed_turn_renders_empty_assistant_with_failed_status():
    items = _items([
        ("message", "user", {"text": "boom?"}, "resp_1"),
        # no assistant message item on a failed turn
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="failed", conversation_id="c")]
    msgs = group_conversation_messages(items, resps)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["text"] == "" and msgs[1]["status"] == "failed"


def test_function_call_items_are_skipped():
    items = _items([
        ("message", "user", {"text": "use a tool"}, "resp_1"),
        ("function_call", None, {"call_id": "c1", "name": "get", "arguments": "{}"}, "resp_1"),
        ("function_call_output", None, {"call_id": "c1", "output": "42"}, "resp_1"),
        ("message", "assistant", {"text": "done"}, "resp_1"),
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="completed", conversation_id="c")]
    msgs = group_conversation_messages(items, resps)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["text"] == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_conversations_view.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.conversations_view'`.

- [ ] **Step 3: Implement `backend/app/conversations_view.py`**

```python
from __future__ import annotations
from typing import Dict, List, Optional
from app.store.base import Item, StoredResponse


def _text(content: dict) -> str:
    return (content or {}).get("text", "") or ""


def group_conversation_messages(
    items: List[Item], responses: List[StoredResponse]
) -> List[Dict]:
    """Reconstruct UI messages from the per-turn item log.

    Items are appended per turn; every item of a turn shares one ``response_id``
    (user ``message`` + optional ``reasoning`` + ``function_call``/output +
    assistant ``message``). We group by ``response_id`` preserving first-seen
    ``seq`` order, then yield up to two UI messages per group: a ``user`` message
    and an ``assistant`` message carrying reasoning + the turn's response status.
    ``function_call``/``function_call_output`` items are skipped (tool UI is out
    of scope for v1).
    """
    by_id: Dict[str, StoredResponse] = {r.id: r for r in responses}

    # Group items by response_id, preserving chronological (seq) first-seen order.
    order: List[str] = []
    groups: Dict[str, List[Item]] = {}
    for it in sorted(items, key=lambda i: i.seq):
        key = it.response_id or ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(it)

    messages: List[Dict] = []
    for key in order:
        group = groups[key]
        user_item = next(
            (i for i in group if i.type == "message" and i.role == "user"), None
        )
        assistant_item = next(
            (i for i in group if i.type == "message" and i.role == "assistant"), None
        )
        reasoning_item = next((i for i in group if i.type == "reasoning"), None)
        resp = by_id.get(key)

        if user_item is not None:
            messages.append(
                {"role": "user", "text": _text(user_item.content), "response_id": key}
            )

        # Emit an assistant message whenever the group corresponds to a response
        # turn (has a response row, or any assistant/reasoning content). A failed
        # turn has a response row but no assistant message item -> empty text.
        if resp is not None or assistant_item is not None or reasoning_item is not None:
            messages.append(
                {
                    "role": "assistant",
                    "text": _text(assistant_item.content) if assistant_item else "",
                    "reasoning": _text(reasoning_item.content) if reasoning_item else None,
                    "response_id": key,
                    "previous_response_id": resp.previous_response_id if resp else None,
                    "status": resp.status if resp else "completed",
                }
            )
    return messages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_conversations_view.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/conversations_view.py tests/app/test_conversations_view.py
git commit -m "feat(app): pure item->UI message grouping for conversation history"
```

---

## Task 5: Conversations routes + wiring

The HTTP surface: list, detail (with reconstructed messages), delete. Thin over the store + the grouping function.

**Files:**
- Create: `backend/app/routes/conversations.py`
- Modify: `backend/app/lean_main.py` (mount the router)
- Test: `tests/app/test_routes_conversations.py`

**Interfaces:**
- Consumes: `store.list_conversations`/`get_conversation`/`get_conversation_items`/`list_conversation_responses`/`delete_conversation` (Task 2); `group_conversation_messages` (Task 4); `AppState`/`get_state` (`app/deps.py`).
- Produces routes:
  - `GET /v1/conversations?user_id=&limit=&offset=` → `{"data": [{"id","title","created_at","updated_at","last_response_id"}]}`.
  - `GET /v1/conversations/{id}` → `{"id","title","created_at","updated_at","latest_response_id","messages":[...]}`; 404 if absent.
  - `DELETE /v1/conversations/{id}` → 200 `{"id","object":"conversation.deleted","deleted":true}` / 404 if absent.

- [ ] **Step 1: Write the failing test** — `tests/app/test_routes_conversations.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.routes.responses import router as responses_router
from app.routes.conversations import router as conversations_router
from common.llm.models import TextChunk


class _EchoLLM:
    async def astream(self, messages, tools=None, **kwargs):
        last = ""
        for m in messages:
            if m.get("role") == "user":
                last = m.get("content") or ""

        class _U:
            prompt_tokens, completion_tokens, total_tokens = 1, 1, 2

        yield TextChunk(delta=f"echo:{last}", usage=None)
        yield TextChunk(delta="", usage=_U())


def _client():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_EchoLLM(), default_model="m")
    app.include_router(responses_router)
    app.include_router(conversations_router)
    return TestClient(app)


def test_list_conversations_filtered_by_user():
    c = _client()
    c.post("/v1/responses", json={"input": "alpha", "stream": False, "user_id": "u1"})
    c.post("/v1/responses", json={"input": "beta", "stream": False, "user_id": "u2"})
    data = c.get("/v1/conversations", params={"user_id": "u1"}).json()["data"]
    assert len(data) == 1
    assert data[0]["title"] == "alpha"
    assert "last_response_id" in data[0] and "updated_at" in data[0]


def test_get_conversation_reconstructs_messages_and_latest_id():
    c = _client()
    first = c.post("/v1/responses", json={"input": "q1", "stream": False, "user_id": "u1"}).json()
    conv_id = first["conversation"]["id"]
    second = c.post("/v1/responses", json={
        "input": "q2", "stream": False, "user_id": "u1",
        "previous_response_id": first["id"], "conversation": conv_id,
    }).json()
    detail = c.get(f"/v1/conversations/{conv_id}").json()
    assert detail["id"] == conv_id
    assert detail["latest_response_id"] == second["id"]
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][0]["text"] == "q1"
    assert detail["messages"][1]["text"] == "echo:q1"
    assert detail["messages"][3]["previous_response_id"] == first["id"]


def test_get_conversation_404():
    c = _client()
    assert c.get("/v1/conversations/nope").status_code == 404


def test_delete_conversation():
    c = _client()
    body = c.post("/v1/responses", json={"input": "x", "stream": False, "user_id": "u1"}).json()
    conv_id = body["conversation"]["id"]
    assert c.delete(f"/v1/conversations/{conv_id}").status_code == 200
    assert c.get(f"/v1/conversations/{conv_id}").status_code == 404
    assert c.delete(f"/v1/conversations/{conv_id}").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_routes_conversations.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.routes.conversations'`.

- [ ] **Step 3: Implement `backend/app/routes/conversations.py`**

```python
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.deps import AppState, get_state
from app.conversations_view import group_conversation_messages

router = APIRouter()


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


@router.get("/v1/conversations")
async def list_conversations(
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    state: AppState = Depends(get_state),
):
    convs = await state.store.list_conversations(user_id=user_id, limit=limit, offset=offset)
    return JSONResponse(
        {
            "data": [
                {
                    "id": c.id,
                    "title": c.title,
                    "created_at": _iso(c.created_at),
                    "updated_at": _iso(c.updated_at),
                    "last_response_id": c.last_response_id,
                }
                for c in convs
            ]
        }
    )


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, state: AppState = Depends(get_state)):
    conv = await state.store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    items = await state.store.get_conversation_items(conversation_id)
    responses = await state.store.list_conversation_responses(conversation_id)
    messages = group_conversation_messages(items, responses)
    return JSONResponse(
        {
            "id": conv.id,
            "title": conv.title,
            "created_at": _iso(conv.created_at),
            "updated_at": _iso(conv.updated_at),
            "latest_response_id": conv.last_response_id,
            "messages": messages,
        }
    )


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, state: AppState = Depends(get_state)):
    conv = await state.store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    await state.store.delete_conversation(conversation_id)
    return JSONResponse(
        {"id": conversation_id, "object": "conversation.deleted", "deleted": True}
    )
```

- [ ] **Step 4: Wire the router into `backend/app/lean_main.py`**

Add the import after the existing route imports:

```python
from app.routes.conversations import router as conversations_router
```

Add the mount after `app.include_router(chat_router)`:

```python
app.include_router(conversations_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_routes_conversations.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the full app suite + import-isolation/boot gates**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (everything, including `test_lean_main_boot.py` and `test_lean_import_isolation.py`).

- [ ] **Step 7: Commit**

```bash
git add backend/app/routes/conversations.py backend/app/lean_main.py tests/app/test_routes_conversations.py
git commit -m "feat(app): Conversations API routes (list/get/delete) wired into lean_main"
```

---

## Self-Review (completed against the spec)

- **`Conversation.title` + `last_response_id` columns** → Task 1 (no migration; `create_all` picks them up).
- **`ResponsesRequest.user_id`** → Task 1; persisted on the conversation → Task 3.
- **Persist conversation rows; `ensure_conversation` before append, title from first user message ≤80 chars set only on creation; `last_response_id` + `updated_at` bump after save** → Task 3.
- **Store protocol additions `ensure_conversation`/`touch_conversation`/`list_conversations`/`get_conversation`/`delete_conversation`** (both stores) → Task 2. `list_conversation_responses` added (the detail route needs per-turn status/previous_response_id; the spec's detail algorithm step 1 says "load all responses for the conversation").
- **`store=false` writes no conversation** → Task 3 (`_persist` only runs under `request.store`), tested.
- **Item → UI message grouping algorithm** (group by response_id, chronological, user+assistant per group, reasoning surfaced, status/previous_response_id from response row, function_call skipped, failed turn → empty assistant + failed status) → Task 4.
- **Routes `GET /v1/conversations`, `GET /v1/conversations/{id}` (latest_response_id == last_response_id), `DELETE`** with the exact response shapes + 404s → Task 5; wired into `lean_main.py` → Task 5.
- **Tests against both `InMemoryStore` and `SqlStore`** (parametrized) + endpoint TestClient tests including the `user_id` round-trip → Tasks 2, 3, 5.

No placeholders; types are consistent across tasks (`group_conversation_messages` signature, store method names, store-item shapes).
