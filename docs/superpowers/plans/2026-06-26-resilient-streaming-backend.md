# Resilient Streaming — Backend Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detach streaming agent runs from the HTTP connection so a turn finishes and persists server-side regardless of disconnects, and add server-side cancel + resumable streaming over the existing OpenAI-Responses serializer.

**Architecture:** A new in-memory `RunManager` (in `AppState`) owns each streaming run as a detached `asyncio.Task` that pumps the serializer's SSE chunks into a per-`response_id` buffer; HTTP connections become *subscribers* to that buffer. The serializer gains a `cancel` event so a cancelled run finalizes a partial as a `response.incomplete` event carrying `Response.status="cancelled"`. Persistence moves into the run's `finally`, so completion, failure, and cancel all persist and advance conversation anchors. Non-background requests keep today's exact inline path.

**Tech Stack:** Python 3.11, FastAPI + Starlette `StreamingResponse`, `asyncio` (Task/Event/Condition), `openai` 2.30.0 types, SQLModel, pytest. Tests run from `backend/`: `cd backend && python -m pytest ../tests/app/... -q`. Async tests use the repo pattern: a plain `def test_*` calling `asyncio.run(run())` over an inner `async def run()`.

**Reference spec:** `docs/superpowers/specs/2026-06-26-resilient-streaming-design.md`. Branch: `personal/yfei/agent-core`. The frontend (reconnect + cancel UI) is the SEPARATE next plan `2026-06-26-resilient-streaming-frontend.md` and depends on this one.

## Global Constraints

- **Import-lean:** the lean service must keep booting with zero RAG/llamaindex/tokenizer deps. New code imports only stdlib (`asyncio`, `time`), `openai` types, FastAPI, and existing `app.*`/`api.protocol.*` modules. `tests/app/test_lean_main_boot.py` and `tests/app/test_lean_import_isolation.py` must stay green.
- **Backward compatible:** when `background` is false, `POST /v1/responses` behaves EXACTLY as today (same inline code path). All existing backend tests (71) stay green.
- **Both stores:** persistence is store-agnostic (reuses the existing `_persist`), so behavior is identical for `InMemoryStore` and `SqlStore`; assert against both where persistence is checked.
- **Single source of emitted events:** every SSE chunk the run emits is appended to the buffer in order; `sequence_number` is monotonic from 1 with no gaps, so **buffer index `i` holds the event with `sequence_number == i+1`** — `starting_after=N` ⇒ resume from buffer index `N`. Do not break this invariant.
- **Status vocabulary:** run/Response status is one of `in_progress | completed | failed | cancelled`. `Response.status` literal already permits all of these (verified). The `incomplete`/max-tokens status is OUT of scope.
- **Cancel wire representation:** a cancelled run's terminal stream event is `ResponseIncompleteEvent` (`type="response.incomplete"`) whose embedded `Response` has `status="cancelled"`. There is no `response.cancelled` SDK event.
- Run the full app suite at the end of every task: `cd backend && python -m pytest ../tests/app -q`.

---

## Key existing contracts (verified, do not re-derive)

- **`api/protocol/responses_serializer.py`** — `async def serialize_response_stream(events, *, model, response_id, conversation_id, sink) -> AsyncIterator[str]`. Internals: a `_sse(event)` helper (`f"data: {event.model_dump_json()}\n\n"`); a `nxt()` monotonic `sequence_number` counter starting at 1; a `_Assembler` (`asm`) with `.status` (default `"completed"`), `.error`, `.text`, `.reasoning`, `.to_response()`, `.finalize(usage)`, `.on_failed(msg)`. The main loop is `async for ev in events:` over `AgentEvent`s. After the loop it closes any open reasoning/message items, calls `asm.finalize(usage)`, then emits a terminal: `ResponseFailedEvent` if `asm.status=="failed"` else `ResponseCompletedEvent`. Finally sets `sink["response"] = final.model_dump(mode="json")` and `sink["items"] = asm.store_items`.
- **`openai.types.responses`** — `ResponseIncompleteEvent(response: Response, sequence_number: int, type="response.incomplete")` (verified). `Response.status` literal includes `"cancelled"` (verified). `ResponseStreamEvent` `TypeAdapter` parses all emitted events (used in tests).
- **`AgentEvent`** (`agent/core/events.py`): `RunStarted(response_id, conversation_id=None)`, `TextDelta(text)`, `ReasoningDelta(text)`, `ToolStarted/ToolCompleted/ToolResult`, `RunCompleted(usage, finish_reason)`, `RunFailed(message, error_type)`, `Usage(input, output, total)`.
- **`app/routes/responses.py`** — `_rid()` mints `resp_*`. `_persist(state, request, current_turn, response_id, conversation_id, store_items, status, usage, error=None)` does `ensure_conversation` → `append_items` (user item + store items) → `save_response(status=...)` → `touch_conversation`. `POST /v1/responses` builds context (`build_context`), `agent = state.make_agent()`, `events = await agent.run(ctx)`, then streams via `serialize_response_stream` in a `gen()` that calls `_persist` AFTER the loop (only if `request.store`), or sync via `serialize_response_sync`. `GET /v1/responses/{id}` returns stored JSON; `DELETE` removes it.
- **`app/deps.py`** — `@dataclass AppState(store, llm, default_model, context_window=110000, max_output_tokens=8000)` + `make_agent()` + `get_state(request)`.
- **`app/schemas.py`** — `ResponsesRequest(BaseModel, extra="ignore")` with `model, input, instructions, previous_response_id, conversation, user_id, store=True, stream=False, metadata, tools`.
- **`app/lean_main.py`** — builds `AppState(store=, llm=, default_model=)` in lifespan; mounts `responses_router`, `chat_router`, `conversations_router`.
- **Test scaffolding** (`tests/app/*`): first two lines
  ```python
  import sys, os, asyncio
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
  ```
  Route tests assemble a `FastAPI()` in-test, set `app.state.app_state = AppState(store=InMemoryStore(), llm=_EchoLLM(), default_model="m")`, `app.include_router(...)`, and use `fastapi.testclient.TestClient`. The shared `_EchoLLM` (in `tests/app/test_routes_responses.py`) is an async fn returning an async generator: `async def astream(self, messages, tools=None, **kwargs): ...; async def gen(): yield TextChunk(...); yield TextChunk(delta="", usage=CompletionUsage(...)); return gen()`. The agent prepends a `[System Time: ...]` header to the user turn, so echo assertions use `.startswith("echo:")`, not exact equality.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/api/protocol/responses_serializer.py` (modify) | `serialize_response_stream` gains `cancel: Optional[asyncio.Event]`; cancelled runs end with a `response.incomplete`/`status="cancelled"` terminal |
| `backend/app/runs.py` (new) | `Run` + `RunManager`: detached run task, in-memory event buffer, broadcast subscribe, cancel, TTL eviction |
| `backend/app/deps.py` (modify) | `AppState` gains `runs: RunManager` (default factory) |
| `backend/app/schemas.py` (modify) | `ResponsesRequest` gains `background: bool = False` |
| `backend/app/routes/responses.py` (modify) | background POST (detached, stream + non-stream); resume `GET ?stream=&starting_after=`; `POST .../cancel` |
| `tests/app/test_serializer_cancel.py` (new) | cancel-aware serializer |
| `tests/app/test_runs.py` (new) | `RunManager` unit tests |
| `tests/app/test_routes_resilient.py` (new) | background/resume/cancel endpoint tests |

---

## Task 1: Cancel-aware serializer

Add a `cancel` event to `serialize_response_stream`: when set mid-stream, stop consuming agent events, finalize the partial, and emit a `response.incomplete` terminal whose `Response.status="cancelled"`. No `cancel` → behavior is byte-for-byte unchanged.

**Files:**
- Modify: `backend/api/protocol/responses_serializer.py`
- Test: `tests/app/test_serializer_cancel.py`

**Interfaces:**
- Consumes: existing `_Assembler`, `_sse`, the `AgentEvent` types.
- Produces: `serialize_response_stream(events, *, model, response_id, conversation_id, sink, cancel: Optional[asyncio.Event] = None)`. On cancel, `sink["response"]["status"] == "cancelled"` and the last SSE event is `type="response.incomplete"` with `response.status=="cancelled"`.

- [ ] **Step 1: Write the failing test** — `tests/app/test_serializer_cancel.py`

```python
import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.core.events import RunStarted, TextDelta, RunCompleted, Usage
from api.protocol.responses_serializer import serialize_response_stream
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

_ADAPTER = TypeAdapter(ResponseStreamEvent)


def _parse(lines):
    evs = []
    for ln in lines:
        for part in ln.splitlines():
            if part.startswith("data:"):
                payload = part[len("data:"):].strip()
                if payload and payload != "[DONE]":
                    evs.append(_ADAPTER.validate_python(json.loads(payload)))
    return evs


async def _slow_events(cancel: asyncio.Event):
    """Yield one text delta, then block forever — until cancel fires."""
    yield RunStarted(response_id="resp_c", conversation_id="conv_c")
    yield TextDelta(text="partial")
    await asyncio.Event().wait()  # never returns; the serializer must abandon us on cancel


async def _normal_events():
    yield RunStarted(response_id="resp_n", conversation_id="conv_n")
    yield TextDelta(text="hello")
    yield RunCompleted(usage=Usage(input=1, output=1, total=2))


def test_cancel_set_midstream_emits_incomplete_cancelled_terminal():
    async def run():
        cancel = asyncio.Event()
        sink = {}
        gen = serialize_response_stream(
            _slow_events(cancel), model="m", response_id="resp_c",
            conversation_id="conv_c", sink=sink, cancel=cancel,
        )
        out = []
        # consume the created/in_progress/text events, then cancel, then drain.
        agen = gen.__aiter__()
        # pull the first few events until we've seen the text delta
        async def pull_until_text():
            while True:
                chunk = await agen.__anext__()
                out.append(chunk)
                if "response.output_text.delta" in chunk:
                    return
        await pull_until_text()
        cancel.set()
        async for chunk in agen:
            out.append(chunk)
        evs = _parse(out)
        assert evs[-1].type == "response.incomplete"
        assert evs[-1].response.status == "cancelled"
        # partial text was preserved on the way out
        text = "".join(e.delta for e in evs if e.type == "response.output_text.delta")
        assert text == "partial"
        assert sink["response"]["status"] == "cancelled"
        assert any(it["type"] == "message" for it in sink["items"])
    asyncio.run(run())


def test_no_cancel_still_completes_normally():
    async def run():
        sink = {}
        out = [c async for c in serialize_response_stream(
            _normal_events(), model="m", response_id="resp_n",
            conversation_id="conv_n", sink=sink, cancel=asyncio.Event(),
        )]
        evs = _parse(out)
        assert evs[-1].type == "response.completed"
        assert sink["response"]["status"] == "completed"
    asyncio.run(run())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_serializer_cancel.py -q`
Expected: FAIL — `serialize_response_stream() got an unexpected keyword argument 'cancel'`.

- [ ] **Step 3: Add `asyncio` import + a cancel-race helper** in `backend/api/protocol/responses_serializer.py`

At the top of the file, add to the imports:

```python
import asyncio
```

Add this module-level sentinel + helper just above `serialize_response_stream` (after `_sse`):

```python
class _Cancelled(Exception):
    """Raised inside the stream loop when the cancel event fires."""


async def _anext_or_cancel(aiter, cancel):
    """Return the next event, or raise _Cancelled if the cancel event fires first.
    Without a cancel event this is a plain anext."""
    if cancel is None:
        return await aiter.__anext__()
    next_task = asyncio.ensure_future(aiter.__anext__())
    cancel_task = asyncio.ensure_future(cancel.wait())
    try:
        done, _pending = await asyncio.wait(
            {next_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
    except BaseException:
        next_task.cancel()
        cancel_task.cancel()
        raise
    if next_task in done:
        cancel_task.cancel()
        return next_task.result()  # may raise StopAsyncIteration
    next_task.cancel()
    raise _Cancelled()
```

- [ ] **Step 4: Thread `cancel` through `serialize_response_stream`**

Change the signature:

```python
async def serialize_response_stream(
    events: AsyncIterator,
    *,
    model: str,
    response_id: str,
    conversation_id: Optional[str],
    sink: Dict,
    cancel: Optional[asyncio.Event] = None,
) -> AsyncIterator[str]:
```

Replace the main consumption loop `async for ev in events:` (the big block that handles `TextDelta`/`ReasoningDelta`/`ToolStarted`/`ToolCompleted`/`ToolResult`/`RunCompleted`/`RunFailed`) with a manual loop that races cancel. Keep the per-event handling body identical — only the loop scaffold changes:

```python
    cancelled = False
    aiter = events.__aiter__()
    while True:
        try:
            ev = await _anext_or_cancel(aiter, cancel)
        except StopAsyncIteration:
            break
        except _Cancelled:
            cancelled = True
            break
        if isinstance(ev, TextDelta):
            # ... (unchanged body from the existing `if isinstance(ev, TextDelta):` branch)
        elif isinstance(ev, ReasoningDelta):
            # ... (unchanged)
        elif isinstance(ev, ToolStarted):
            # ... (unchanged)
        elif isinstance(ev, ToolCompleted):
            # ... (unchanged)
        elif isinstance(ev, ToolResult):
            # ... (unchanged)
        elif isinstance(ev, RunCompleted):
            usage = ev.usage
        elif isinstance(ev, RunFailed):
            asm.on_failed(ev.message)
```

NOTE: copy the existing branch bodies verbatim into the new `if/elif` chain — do not rewrite them. The ONLY change is `async for ev in events:` → the `while True:` + `_anext_or_cancel` scaffold above.

After the loop (the existing "close an open reasoning item" / "close an open message item" / `asm.finalize(usage)` / terminal block stays), set the status for cancellation BEFORE choosing the terminal. Replace the terminal selection block:

```python
    asm.finalize(usage)
    if msg_open:
        # ... (unchanged: emit the message output_item.done) ...

    if cancelled:
        asm.status = "cancelled"

    final = asm.to_response()
    if asm.status == "failed":
        yield _sse(
            ResponseFailedEvent(
                response=final, sequence_number=nxt(), type="response.failed"
            )
        )
    elif asm.status == "cancelled":
        yield _sse(
            ResponseIncompleteEvent(
                response=final, sequence_number=nxt(), type="response.incomplete"
            )
        )
    else:
        yield _sse(
            ResponseCompletedEvent(
                response=final, sequence_number=nxt(), type="response.completed"
            )
        )

    sink["response"] = final.model_dump(mode="json")
    sink["items"] = asm.store_items
```

Add `ResponseIncompleteEvent` to the existing `from openai.types.responses import (...)` block.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_serializer_cancel.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the existing serializer + stream regression tests**

Run: `cd backend && python -m pytest ../tests/app/test_responses_serializer_stream.py ../tests/app/test_responses_serializer_sync.py -q`
Expected: PASS (the no-cancel path is unchanged).

- [ ] **Step 7: Commit**

```bash
git add backend/api/protocol/responses_serializer.py tests/app/test_serializer_cancel.py
git commit -m "feat(protocol): cancel-aware streaming serializer (response.incomplete + status cancelled)"
```

---

## Task 2: `RunManager` + `Run`

The detached-run registry: a run pumps the serializer into an in-memory buffer; subscribers replay-from-cursor then tail; cancel stops it; finished runs persist regardless of subscribers and evict after a TTL.

**Files:**
- Create: `backend/app/runs.py`
- Test: `tests/app/test_runs.py`

**Interfaces:**
- Consumes: `serialize_response_stream(..., cancel=...)` (Task 1).
- Produces:
  - `class Run` with attributes `response_id, conversation_id, model, status, events: List[str], cancel: asyncio.Event, done: asyncio.Event, finished_at: Optional[float], task`.
  - `class RunManager`:
    - `start(self, *, events, model, response_id, conversation_id, persist: Callable[[dict, str], Awaitable[None]]) -> Run` — register + spawn the pump task; sweeps expired runs first.
    - `get(self, response_id) -> Optional[Run]`.
    - `async def cancel(self, response_id) -> bool` — set the run's cancel event; `False` if no live run.
    - `async def subscribe(self, run: Run, starting_after: int = 0) -> AsyncIterator[str]` — yield buffered chunks from index `starting_after`, then tail until done.
  - `persist(sink: dict, status: str)` is the callback the pump invokes in `finally` when `sink` has a response.

- [ ] **Step 1: Write the failing test** — `tests/app/test_runs.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.runs import RunManager
from agent.core.events import RunStarted, TextDelta, RunCompleted, Usage


async def _events(texts):
    yield RunStarted(response_id="resp_1", conversation_id="conv_1")
    for t in texts:
        yield TextDelta(text=t)
    yield RunCompleted(usage=Usage(input=1, output=1, total=2))


async def _blocking_events():
    yield RunStarted(response_id="resp_1", conversation_id="conv_1")
    yield TextDelta(text="partial")
    await asyncio.Event().wait()


def test_run_persists_completed_with_no_subscriber():
    async def run():
        rm = RunManager()
        seen = {}

        async def persist(sink, status):
            seen["status"] = status
            seen["items"] = sink["items"]

        r = rm.start(events=_events(["he", "llo"]), model="m",
                     response_id="resp_1", conversation_id="conv_1", persist=persist)
        await asyncio.wait_for(r.done.wait(), timeout=2)
        assert r.status == "completed"
        assert seen["status"] == "completed"
        assert any(it["type"] == "message" and it["content"]["text"] == "hello"
                   for it in seen["items"])
    asyncio.run(run())


def test_subscribe_from_cursor_replays_then_tails():
    async def run():
        rm = RunManager()

        async def persist(sink, status):
            pass

        r = rm.start(events=_events(["a", "b", "c"]), model="m",
                     response_id="resp_1", conversation_id="conv_1", persist=persist)
        await asyncio.wait_for(r.done.wait(), timeout=2)
        # full replay from 0
        full = [c async for c in rm.subscribe(r, starting_after=0)]
        # resume from a cursor: only events with sequence_number > N (buffer index >= N)
        n = 3
        tail = [c async for c in rm.subscribe(r, starting_after=n)]
        assert len(full) > len(tail) and len(tail) == len(full) - n
        # the tail still ends with the terminal completed event
        assert "response.completed" in tail[-1]
    asyncio.run(run())


def test_live_subscriber_receives_streaming_events():
    async def run():
        rm = RunManager()

        async def persist(sink, status):
            pass

        r = rm.start(events=_events(["x", "y"]), model="m",
                     response_id="resp_1", conversation_id="conv_1", persist=persist)
        got = [c async for c in rm.subscribe(r, starting_after=0)]
        assert "response.created" in got[0]
        assert "response.completed" in got[-1]
    asyncio.run(run())


def test_cancel_finalizes_cancelled_and_persists_partial():
    async def run():
        rm = RunManager()
        seen = {}

        async def persist(sink, status):
            seen["status"] = status
            seen["items"] = sink["items"]

        r = rm.start(events=_blocking_events(), model="m",
                     response_id="resp_1", conversation_id="conv_1", persist=persist)
        # wait until the partial text has been buffered
        for _ in range(200):
            if any("response.output_text.delta" in c for c in r.events):
                break
            await asyncio.sleep(0.01)
        assert await rm.cancel("resp_1") is True
        await asyncio.wait_for(r.done.wait(), timeout=2)
        assert r.status == "cancelled"
        assert seen["status"] == "cancelled"
        assert any(it["content"].get("text") == "partial"
                   for it in seen["items"] if it["type"] == "message")
    asyncio.run(run())


def test_cancel_unknown_or_finished_returns_false():
    async def run():
        rm = RunManager()
        assert await rm.cancel("nope") is False
    asyncio.run(run())


def test_eviction_drops_finished_runs_after_ttl():
    async def run():
        rm = RunManager(retention_seconds=0)  # evict immediately on next start

        async def persist(sink, status):
            pass

        r1 = rm.start(events=_events(["a"]), model="m", response_id="resp_1",
                      conversation_id="conv_1", persist=persist)
        await asyncio.wait_for(r1.done.wait(), timeout=2)
        # next start sweeps expired finished runs
        r2 = rm.start(events=_events(["b"]), model="m", response_id="resp_2",
                      conversation_id="conv_1", persist=persist)
        await asyncio.wait_for(r2.done.wait(), timeout=2)
        assert rm.get("resp_1") is None
        assert rm.get("resp_2") is not None
    asyncio.run(run())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_runs.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.runs'`.

- [ ] **Step 3: Implement `backend/app/runs.py`**

```python
from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional
from loguru import logger
from api.protocol.responses_serializer import serialize_response_stream

RETENTION_SECONDS = 300

PersistFn = Callable[[Dict, str], Awaitable[None]]


class Run:
    """A single in-flight (or recently finished) streaming response.

    Holds the ordered SSE buffer (``events``); subscribers replay from a cursor
    then tail. ``sequence_number`` is monotonic from 1 with no gaps, so buffer
    index ``i`` holds the event with ``sequence_number == i+1`` — a resume cursor
    ``starting_after=N`` maps directly to buffer index ``N``.
    """

    def __init__(self, response_id: str, conversation_id: str, model: str):
        self.response_id = response_id
        self.conversation_id = conversation_id
        self.model = model
        self.status = "in_progress"
        self.events: List[str] = []
        self.cancel = asyncio.Event()
        self.done = asyncio.Event()
        self.finished_at: Optional[float] = None
        self.task: Optional[asyncio.Task] = None
        self._cond = asyncio.Condition()

    async def append(self, chunk: str) -> None:
        async with self._cond:
            self.events.append(chunk)
            self._cond.notify_all()

    async def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = time.monotonic()
        async with self._cond:
            self.done.set()
            self._cond.notify_all()


class RunManager:
    """In-memory, single-process registry of detached streaming runs."""

    def __init__(self, retention_seconds: int = RETENTION_SECONDS):
        self._runs: Dict[str, Run] = {}
        self._retention = retention_seconds

    def get(self, response_id: str) -> Optional[Run]:
        return self._runs.get(response_id)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        stale = [
            rid for rid, r in self._runs.items()
            if r.finished_at is not None and now - r.finished_at > self._retention
        ]
        for rid in stale:
            self._runs.pop(rid, None)

    def start(self, *, events, model: str, response_id: str,
              conversation_id: str, persist: PersistFn) -> Run:
        self._evict_expired()
        run = Run(response_id, conversation_id, model)
        self._runs[response_id] = run
        run.task = asyncio.create_task(self._pump(run, events, persist))
        return run

    async def _pump(self, run: Run, events, persist: PersistFn) -> None:
        sink: Dict = {}
        status = "failed"
        try:
            async for chunk in serialize_response_stream(
                events, model=run.model, response_id=run.response_id,
                conversation_id=run.conversation_id, sink=sink, cancel=run.cancel,
            ):
                await run.append(chunk)
            status = (sink.get("response") or {}).get("status", "completed")
        except Exception:  # noqa: BLE001 — a broken run must still finalize
            logger.exception(f"run {run.response_id} pump failed")
            status = "failed"
        finally:
            try:
                if sink.get("response"):
                    await persist(sink, status)
            except Exception:  # noqa: BLE001 — persistence failure must not hang the run
                logger.exception(f"run {run.response_id} persist failed")
            await run.finish(status)

    async def cancel(self, response_id: str) -> bool:
        run = self._runs.get(response_id)
        if run is None or run.done.is_set():
            return False
        run.cancel.set()
        return True

    async def subscribe(self, run: Run, starting_after: int = 0) -> AsyncIterator[str]:
        i = max(0, starting_after)
        while True:
            async with run._cond:
                while i >= len(run.events) and not run.done.is_set():
                    await run._cond.wait()
                new = run.events[i:]
                i = len(run.events)
                finished = run.done.is_set() and i >= len(run.events)
            for chunk in new:
                yield chunk
            if finished:
                break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_runs.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full app suite**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (no regressions; `loguru` is already a dependency).

- [ ] **Step 6: Commit**

```bash
git add backend/app/runs.py tests/app/test_runs.py
git commit -m "feat(app): RunManager (detached runs, buffered subscribe, cancel, TTL eviction)"
```

---

## Task 3: Background POST + `AppState.runs`

Wire the registry into the app and add the opt-in `background` path to `POST /v1/responses`: detach the run, return either a live SSE subscription (stream) or the `in_progress` response (non-stream). Non-background stays inline.

**Files:**
- Modify: `backend/app/deps.py` (`AppState.runs`)
- Modify: `backend/app/schemas.py` (`ResponsesRequest.background`)
- Modify: `backend/app/routes/responses.py` (background branch)
- Test: `tests/app/test_routes_resilient.py` (background cases)

**Interfaces:**
- Consumes: `RunManager.start`/`subscribe` (Task 2); existing `_persist`, `build_context`, `make_agent`.
- Produces: `AppState.runs: RunManager`; `ResponsesRequest.background: bool`; `POST /v1/responses` honoring `background:true`.

- [ ] **Step 1: Write the failing test** — `tests/app/test_routes_resilient.py`

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
from openai.types.chat.chat_completion_chunk import CompletionUsage


class _EchoLLM:
    async def astream(self, messages, tools=None, **kwargs):
        last = ""
        for m in messages:
            if m.get("role") == "user":
                last = m.get("content") or ""
        usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)

        async def gen():
            yield TextChunk(delta=f"echo:{last}", usage=None)
            yield TextChunk(delta="", usage=usage)
        return gen()


def _client():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_EchoLLM(), default_model="m")
    app.include_router(responses_router)
    app.include_router(conversations_router)
    return TestClient(app)


def test_background_stream_emits_events_and_persists():
    c = _client()
    with c.stream("POST", "/v1/responses",
                  json={"input": "hi", "stream": True, "background": True, "user_id": "u1"}) as r:
        assert r.status_code == 200
        raw = "".join(chunk for chunk in r.iter_text())
    assert "response.created" in raw and "response.completed" in raw
    assert "response.output_text.delta" in raw
    # the detached run persisted the conversation (listable)
    data = c.get("/v1/conversations", params={"user_id": "u1"}).json()["data"]
    assert len(data) == 1 and data[0]["title"] == "hi"


def test_background_non_stream_returns_in_progress_immediately():
    c = _client()
    body = c.post("/v1/responses",
                  json={"input": "hello", "stream": False, "background": True, "user_id": "u1"}).json()
    assert body["status"] == "in_progress"
    assert body["object"] == "response"
    assert body["id"].startswith("resp_")
    assert body["conversation"]["id"].startswith("conv_")


def test_non_background_path_unchanged():
    c = _client()
    body = c.post("/v1/responses", json={"input": "hi", "stream": False}).json()
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"].startswith("echo:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_routes_resilient.py -q`
Expected: FAIL — `AppState.__init__() ... 'runs'` is missing / `background` ignored so non-stream returns `completed` not `in_progress`.

- [ ] **Step 3: Add `AppState.runs`** in `backend/app/deps.py`

Add imports at the top:

```python
from dataclasses import dataclass, field
from app.runs import RunManager
```

Add the field to `AppState` (after `max_output_tokens`):

```python
    runs: RunManager = field(default_factory=RunManager)
```

NOTE: keep the existing `@dataclass` decorator and `make_agent`/`get_state` unchanged. The default factory means existing `AppState(store=, llm=, default_model=)` construction keeps working.

- [ ] **Step 4: Add `background`** in `backend/app/schemas.py`

Add to `ResponsesRequest` (after `stream`):

```python
    background: bool = False
```

- [ ] **Step 5: Add the background branch** in `backend/app/routes/responses.py`

Add imports at the top:

```python
import time
from fastapi.responses import StreamingResponse  # already imported; keep
```

In `create_response`, AFTER `response_id = _rid()` / `agent = state.make_agent()` / `events = await agent.run(ctx)`, and BEFORE the existing `if request.stream:` block, insert the background branch:

```python
    if request.background:
        async def _persist_run(sink: dict, status: str):
            if not request.store:
                return
            r = sink["response"]
            await _persist(
                state, request, ctx.current_turn, response_id, conversation_id,
                sink["items"], status, r.get("usage"), r.get("error"),
            )

        run = state.runs.start(
            events=events, model=request.model, response_id=response_id,
            conversation_id=conversation_id, persist=_persist_run,
        )
        if request.stream:
            return StreamingResponse(
                state.runs.subscribe(run, starting_after=0),
                media_type="text/event-stream",
            )
        return JSONResponse(
            {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": request.model,
                "conversation": {"id": conversation_id},
                "created_at": time.time(),
            }
        )
```

NOTE: the existing non-background `if request.stream:` / sync blocks below stay exactly as they are. `_persist_run` reuses the same `_persist` (so cancelled/failed/completed all persist + touch the conversation). The `background` run owns persistence; the subscriber connection does not persist.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_routes_resilient.py -q`
Expected: PASS (3 tests).

- [ ] **Step 7: Run the full app suite**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (non-background tests in `test_routes_responses.py` unchanged).

- [ ] **Step 8: Commit**

```bash
git add backend/app/deps.py backend/app/schemas.py backend/app/routes/responses.py tests/app/test_routes_resilient.py
git commit -m "feat(app): background detached runs on POST /v1/responses (+ AppState.runs)"
```

---

## Task 4: Resume + cancel endpoints

Expose resumable streaming (`GET /v1/responses/{id}?stream=true&starting_after=N`) and server-side cancel (`POST /v1/responses/{id}/cancel`), then wire the conversations router stays as-is.

**Files:**
- Modify: `backend/app/routes/responses.py` (extend GET; add cancel route)
- Test: `tests/app/test_routes_resilient.py` (append resume + cancel cases)

**Interfaces:**
- Consumes: `RunManager.get`/`subscribe`/`cancel` (Task 2); the background POST (Task 3).
- Produces:
  - `GET /v1/responses/{id}` gains query params `stream: bool = False`, `starting_after: int = 0`. When `stream`: resume the run if present (SSE) else 409. Without `stream`: existing JSON.
  - `POST /v1/responses/{id}/cancel` → 200 `{"id", "object":"response.cancel", "status":"cancelling"}` if a live run was signalled; else fall back to the stored response's status, or 404.

- [ ] **Step 1: Write the failing test** — append to `tests/app/test_routes_resilient.py`

```python
def _seq(line_block):
    import json
    seqs = []
    for part in line_block.splitlines():
        if part.startswith("data:"):
            payload = part[len("data:"):].strip()
            if payload:
                seqs.append(json.loads(payload).get("sequence_number"))
    return [s for s in seqs if s is not None]


def test_resume_replays_only_events_after_cursor():
    c = _client()
    # start + fully drain a background stream to populate the run buffer
    with c.stream("POST", "/v1/responses",
                  json={"input": "hi", "stream": True, "background": True, "user_id": "u1"}) as r:
        first_raw = "".join(chunk for chunk in r.iter_text())
    rid = None
    import json
    for part in first_raw.splitlines():
        if part.startswith("data:") and '"response.created"' in part:
            rid = json.loads(part[len("data:"):].strip())["response"]["id"]
            break
    assert rid is not None
    # resume from cursor N=3 -> only events with sequence_number > 3
    with c.stream("GET", f"/v1/responses/{rid}",
                  params={"stream": "true", "starting_after": 3}) as r:
        assert r.status_code == 200
        resume_raw = "".join(chunk for chunk in r.iter_text())
    seqs = _seq(resume_raw)
    assert seqs and min(seqs) > 3
    assert "response.completed" in resume_raw


def test_resume_unknown_run_returns_409():
    c = _client()
    r = c.get("/v1/responses/resp_missing", params={"stream": "true", "starting_after": 0})
    assert r.status_code == 409


def test_cancel_persists_cancelled_partial_and_lists_conversation():
    import threading, time as _t

    # an LLM that yields one chunk then blocks, so the run is live long enough to cancel
    class _SlowLLM:
        async def astream(self, messages, tools=None, **kwargs):
            import asyncio
            async def gen():
                from common.llm.models import TextChunk
                yield TextChunk(delta="partial", usage=None)
                await asyncio.Event().wait()
            return gen()

    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_SlowLLM(), default_model="m")
    app.include_router(responses_router)
    app.include_router(conversations_router)
    c = TestClient(app)

    rid_holder = {}

    def _consume():
        import json
        with c.stream("POST", "/v1/responses",
                      json={"input": "go", "stream": True, "background": True, "user_id": "u1"}) as r:
            for part in r.iter_lines():
                if part and part.startswith("data:") and '"response.created"' in part:
                    rid_holder["rid"] = json.loads(part[len("data:"):].strip())["response"]["id"]
                    # keep reading until the server closes the stream (after cancel)
            # stream ends after cancel terminal
    th = threading.Thread(target=_consume, daemon=True)
    th.start()
    # wait for the response id to appear, then cancel
    for _ in range(200):
        if "rid" in rid_holder:
            break
        _t.sleep(0.02)
    assert "rid" in rid_holder
    cr = c.post(f"/v1/responses/{rid_holder['rid']}/cancel")
    assert cr.status_code == 200
    th.join(timeout=5)
    # the cancelled turn was persisted as a real, listable conversation
    data = c.get("/v1/conversations", params={"user_id": "u1"}).json()["data"]
    assert len(data) == 1
    detail = c.get(f"/v1/conversations/{data[0]['id']}").json()
    assert detail["latest_response_id"] == rid_holder["rid"]


def test_cancel_unknown_returns_404():
    c = _client()
    assert c.post("/v1/responses/resp_missing/cancel").status_code == 404
```

NOTE: `test_cancel_*` drives the streaming endpoint from a background thread because `TestClient.stream` is a context manager held open while the run is live; the main thread fires the cancel. If `iter_lines` semantics differ in this Starlette version, switch to reading `r.iter_text()` incrementally — the assertion that matters is that after cancel the conversation is persisted with `latest_response_id == rid`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_routes_resilient.py -q`
Expected: FAIL — resume returns the JSON body (not SSE) / 200 instead of 409; cancel route 405 (not defined).

- [ ] **Step 3: Extend `GET /v1/responses/{id}`** in `backend/app/routes/responses.py`

Replace the existing `get_response` handler signature + body to accept the resume query params:

```python
@router.get("/v1/responses/{response_id}")
async def get_response(
    response_id: str,
    stream: bool = False,
    starting_after: int = 0,
    state: AppState = Depends(get_state),
):
    if stream:
        run = state.runs.get(response_id)
        if run is None:
            raise HTTPException(status_code=409, detail="run not resumable")
        return StreamingResponse(
            state.runs.subscribe(run, starting_after=starting_after),
            media_type="text/event-stream",
        )
    stored = await state.store.get_response(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="response not found")
    return JSONResponse(
        {
            "id": stored.id,
            "object": "response",
            "status": stored.status,
            "model": stored.model,
            "conversation": (
                {"id": stored.conversation_id} if stored.conversation_id else None
            ),
            "usage": stored.usage,
            "error": stored.error,
            "previous_response_id": stored.previous_response_id,
        }
    )
```

NOTE: the non-stream branch body is identical to the existing handler — only the new `stream`/`starting_after` params + the resume branch are added.

- [ ] **Step 4: Add the cancel route** in `backend/app/routes/responses.py`

Add after the `delete_response` handler:

```python
@router.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str, state: AppState = Depends(get_state)):
    if await state.runs.cancel(response_id):
        return JSONResponse(
            {"id": response_id, "object": "response.cancel", "status": "cancelling"}
        )
    # no live run: report the stored status if we have it, else 404
    stored = await state.store.get_response(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="response not found")
    return JSONResponse(
        {"id": response_id, "object": "response.cancel", "status": stored.status}
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_routes_resilient.py -q`
Expected: PASS (7 tests total in the file).

- [ ] **Step 6: Run the full app suite + boot/isolation gates**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (everything, including `test_lean_main_boot.py` and `test_lean_import_isolation.py`).

- [ ] **Step 7: Commit**

```bash
git add backend/app/routes/responses.py tests/app/test_routes_resilient.py
git commit -m "feat(app): resume (GET ?stream=&starting_after) + POST /v1/responses/{id}/cancel"
```

---

## Self-Review (completed against the spec)

- **Detach run from connection; persistence in the run's `finally`** → Task 2 (`RunManager._pump` persists in `finally`) + Task 3 (`_persist_run` wired). The no-subscriber-still-persists guarantee is tested (`test_run_persists_completed_with_no_subscriber`).
- **Cancel-aware serializer → `response.incomplete` / `status="cancelled"`** → Task 1.
- **`background: true` opt-in; stream → live SSE, non-stream → in_progress; non-background unchanged** → Task 3.
- **Resume `GET ?stream=true&starting_after=N` (seq>N), 409 when not resumable** → Task 4 (cursor maps to buffer index; tested for `min(seq)>N`).
- **`POST .../cancel` → cancelled persisted partial + conversation touched + 404 unknown** → Tasks 1+2+4 (end-to-end cancel test asserts `latest_response_id`).
- **TTL eviction; single source of events; monotonic seq invariant** → Task 2 (eviction tested; cursor-as-index documented).
- **Status vocabulary `in_progress|completed|failed|cancelled`; `incomplete`/max-tokens out of scope** → honored (no producer emits `incomplete`).
- **Import-lean + both stores + backward compatibility** → Global Constraints; non-background regression covered by existing tests + `test_non_background_path_unchanged`.

No placeholders; types/signatures (`RunManager.start(*, events, model, response_id, conversation_id, persist)`, `subscribe(run, starting_after)`, `cancel(response_id)`, `serialize_response_stream(..., cancel=)`, `_persist_run(sink, status)`) are consistent across tasks.
