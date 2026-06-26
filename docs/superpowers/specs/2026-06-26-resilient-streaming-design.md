# Resilient Streaming — Background Runs, Resume, and Server-Side Cancel — Design

**Date:** 2026-06-26
**Status:** Approved (pending spec review)
**Branch:** `personal/yfei/agent-core`
**Builds on:** the lean agent service (`backend/app/`, `/v1/responses` OpenAI-Responses serializer with `sequence_number` on every stream event) and the new Agent Chat frontend (`newfrontend/`, pure stream reducer + `useResponsesChat`). See `docs/superpowers/specs/2026-06-26-new-agent-chat-frontend-design.md` and the two implemented plans.

## Problem

Today the agent runs **inside the HTTP request task**: `routes/responses.py` calls `agent.run(ctx)` and consumes the event stream in the streaming response's `gen()`. Two consequences:

1. **Any disconnect discards the turn.** The route persists only *after* the `async for` loop finishes (`gen()` calls `_persist` after the stream completes). If the client disconnects — a page reload, a network blip, a closed laptop — Starlette tears down the response task, the generator is cancelled, and nothing is persisted: no conversation row, no items, no response. The work is lost.
2. **There is no way to stop a run server-side, and no way to reconnect to one.** v1 "Stop" is a client-only `AbortController.abort()`; the partial answer lives only in the browser, is not persisted, and is not continuable. There is no `cancel` endpoint and no resume.

The lean service's serializer already stamps `sequence_number` on every `response.*` event — the foundation for resumable streams — but nothing uses it.

We want **resilience**: an in-flight answer must survive the connection. A reload or network drop should **reconnect and resume the same run from where it left off** without losing tokens, the run should **finish and persist server-side regardless of the connection**, and **Stop should be a real server-side cancel** that persists a continuable partial.

## Goals

- **Detach the run from the connection.** Streaming runs execute in a registry-managed background task; the HTTP connection is a *subscriber*, not the owner. The run finishes and persists even if every subscriber disconnects.
- **Resume.** `GET /v1/responses/{id}?stream=true&starting_after=N` replays buffered events after sequence `N`, then tails the live run to its terminal.
- **Server-side cancel.** `POST /v1/responses/{id}/cancel` stops the run cooperatively, persists the partial with `status="cancelled"`, and emits a terminal `response.incomplete` event whose embedded `Response` carries `status="cancelled"`.
- **Frontend delivers the resilience:** chat turns run in background; reload/visibility/online triggers reconnect-and-resume; Stop calls cancel (the partial persists and the conversation continues from it).
- Backward compatible: non-background requests keep today's exact inline behavior; all existing backend (71) and frontend (33) tests stay green.

## Scope

**In scope:**
- Backend `RunManager` (in `AppState`): a per-`response_id` registry holding the detached `asyncio.Task`, an ordered in-memory event buffer, status, a cancel signal, a broadcast notify, and a finished-at timestamp for TTL eviction.
- A **cancel-aware** extension to `serialize_response_stream` (`backend/api/protocol/responses_serializer.py`): an optional `cancel: asyncio.Event`; when set, the serializer finalizes the partial and emits a terminal `response.incomplete` with `Response.status="cancelled"` (and `sink` status `cancelled`) instead of `response.completed`.
- `POST /v1/responses` gains `background: bool = false`. With `background:true`: spawn the detached run; `stream:true` → SSE subscribed from seq 0 tailing live; `stream:false` → return the `in_progress` `Response` immediately.
- `GET /v1/responses/{id}?stream=true&starting_after=N` (resume) and `POST /v1/responses/{id}/cancel`.
- Persistence moves into the detached run's `finally` (reusing the existing `_persist`), so completion, failure, **and** cancel all persist and advance conversation anchors.
- Frontend: reducer tracks `lastSequenceNumber` and handles `response.incomplete`; a plain `fetch`+SSE-reader resume util; `cancelResponse` API; `useResponsesChat` sends `background:true`, Stop→cancel, and reconnect-on-resume; the assistant bubble renders a `cancelled` state.

**Out of scope (this iteration):**
- The `incomplete`/`max_output_tokens` status (length-truncation). Cheap to add later; deferred to keep scope tight. (`Response.status` and the reducer will *tolerate* it, but no producer emits it.)
- Multi-worker / multi-process resume. The registry is single-process, single-event-loop. A run lives in exactly one worker; cross-worker live-attach would need a shared bus (Redis pub/sub) and is explicitly deferred.
- Durable (restart-surviving) resume. A server restart kills in-flight runs; the *final* response is only persisted on finalize, so a run killed mid-flight by a restart is lost (same as today). Event-log persistence is deferred.
- A real job queue / `queued` status. Background runs start immediately as `in_progress`.
- Tenant/auth; tool/RAG UI.

## Decisions

1. **Detach by default for background; opt-in via `background:true`.** Non-background keeps the inline path verbatim (backward compatibility, existing tests untouched). The frontend sets `background:true` for chat turns, so chat is resilient; other callers are unaffected.
2. **In-memory, single-process registry.** No locks needed (cooperative single event loop). Finished runs are retained for a TTL (default **300 s**) so a late reconnect can still replay, then evicted; the final `Response` is always in the store regardless, so eviction ends *resume-ability*, not data availability.
3. **Persistence belongs to the run, not the connection.** The detached task persists in its `finally` via the existing `_persist`, with the final status from the run. This is what makes a plain disconnect non-lossy and a cancel continuable.
4. **Cancel signals over the wire as `response.incomplete` carrying `status="cancelled"`.** There is no `response.cancelled` stream event in the OpenAI SDK, but `ResponseIncompleteEvent` (`type:"response.incomplete"`) exists and `Response.status` permits `"cancelled"` — so this is SDK-parseable on the POST path and on our own reducer. The store/`Response` status is `cancelled`.
5. **A cancelled turn IS a real, continuable response.** It persists with a `response_id`, the conversation's `last_response_id`/`updated_at` are touched, and the frontend advances its anchors — so the next turn continues from the cancelled one. This is the upgrade over v1's throwaway local "stopped".
6. **Resume uses a plain `fetch`+SSE reader on the frontend, not the OpenAI SDK.** The SDK has no first-class "resume from `starting_after`" we want to depend on; our reducer consumes plain event objects, so a tiny SSE reader keeps us in control of the cursor. The initial POST keeps using the SDK (`client.responses.create`).

## Architecture

### Backend

#### `RunManager` and `Run` (`backend/app/runs.py`, new)

```text
Run:
  response_id: str
  conversation_id: str
  model: str
  status: str                      # "in_progress" | "completed" | "failed" | "cancelled"
  events: list[str]                # SSE chunk strings, in emission order (each carries sequence_number)
  cancel: asyncio.Event            # set by cancel(); the serializer races against it
  done: asyncio.Event              # set when the run task finishes
  finished_at: float | None        # for TTL eviction
  task: asyncio.Task
  _waiters: asyncio.Condition      # broadcast: notified after each append / on done

RunManager:
  start(run_inputs, persist) -> Run         # build a Run, spawn the pump task, register it
  get(response_id) -> Run | None
  cancel(response_id) -> bool               # set run.cancel; True if a live run was signalled
  subscribe(run, starting_after: int) -> AsyncIterator[str]   # replay seq>N then tail to done
  _evict_expired()                          # drop runs finished > TTL ago (swept on each start)
```

- **The pump** (`RunManager.start`'s task body) iterates `serialize_response_stream(agent_events, ..., sink=sink, cancel=run.cancel)`, appending every yielded SSE chunk to `run.events` and `notify_all`-ing the condition. On loop end it sets `run.status = sink["response"]["status"]`. In `finally` it calls the `persist` callback with `(sink, run.status)`, sets `run.done`, stamps `finished_at`, and notifies. A failure inside the agent surfaces as a `response.failed` terminal from the serializer (existing behavior) → status `failed`, still persisted.
- **`subscribe(run, starting_after)`** yields buffered chunks whose `sequence_number > starting_after`, then waits on the condition for new appends, yielding as they arrive, until `run.done` and the buffer is drained. Multiple subscribers (the original POST + any number of resumes) read the same buffer independently. Subscribers never mutate the run.
- **Broadcast pattern:** appenders hold the `Condition`, append, `notify_all`. Each subscriber loop: drain `events[i:]` from its own index `i`, then `async with cond: await cond.wait()` (re-checking `done` and `len(events)` to avoid lost-wakeup).

#### Cancel-aware serializer (`responses_serializer.py`, modify)

`serialize_response_stream(..., cancel: Optional[asyncio.Event] = None)`. The main `async for ev in events` loop races each `anext(events)` against `cancel.wait()`. If cancel wins: stop consuming, run the same finalize path, but emit a terminal **`ResponseIncompleteEvent`** (`type:"response.incomplete"`) whose `response` has `status="cancelled"`, and set `sink["response"]` accordingly (status `cancelled`). If the stream ends normally, the existing `response.completed`/`response.failed` terminal is emitted unchanged. The `sink` contract (`{"response": <dump>, "items": [...]}`) is unchanged otherwise, so `_persist` works as-is.

#### Routes (`routes/responses.py`, modify)

- `POST /v1/responses` — when `request.background`:
  - Mint `response_id`, build context (as today), build the agent + event stream.
  - Construct a `persist` closure capturing `state`, `request`, `ctx.current_turn`, `response_id`, `conversation_id` → calls the existing `_persist(...)` with the sink's items/status/usage/error.
  - `run = state.runs.start(run_inputs, persist)`.
  - `stream:true` → `StreamingResponse(state.runs.subscribe(run, starting_after=0), media_type="text/event-stream")`.
  - `stream:false` → return the `in_progress` `Response` JSON immediately (id, model, status `in_progress`, conversation id).
  - When `background` is false → **unchanged** inline path (existing code, existing tests).
- `GET /v1/responses/{id}` — when `?stream=true`: look up the run; if present, `StreamingResponse(subscribe(run, starting_after))`; if absent → **409** (`{"error":"run not resumable"}`) so the client falls back to the non-stream `GET` for the final. Without `stream`, the existing JSON lookup is unchanged.
- `POST /v1/responses/{id}/cancel` — `state.runs.cancel(id)`; if it signalled a live run → 200 `{"id", "object":"response.cancel", "status":"cancelling"}`; if no live run (already finished/unknown) → look up the stored response and return its status, or 404 if unknown.

#### `AppState` (`deps.py`, modify)

Add `runs: RunManager` (constructed in `lean_main.py` lifespan and in test `AppState(...)`). Default-construct a `RunManager()` so existing test `AppState(store=..., llm=..., default_model=...)` keeps working via a default factory.

### Frontend (`newfrontend/`)

- **Reducer (`stream/reducer.ts`):** `StreamState` gains `lastSequenceNumber: number`; every branch captures `event.sequence_number` when present (max-monotonic). New case `response.incomplete` → read `response.status`; map `"cancelled"` → message `status:"cancelled"` (and capture ids/usage like `completed`). `MessageStatus` gains `"cancelled"`.
- **API (`api/`):** `cancelResponse(id)` → `POST /v1/responses/{id}/cancel`. `streamResume(id, startingAfter, signal)` → `fetch('/v1/responses/{id}?stream=true&starting_after=N', {signal})`, parse the SSE body with a small reader util (`lib/sse.ts`: split on `\n\n`, strip `data: `, `JSON.parse`) yielding plain event objects.
- **Hook (`useResponsesChat.ts`):**
  - `send` sets `background:true` in the stream params.
  - `stop()` → if the in-flight message has a `responseId`, call `cancelResponse(responseId)`; if the id isn't known yet, set a `cancelPending` ref and fire `cancelResponse` the moment `response.created` yields the id. (No local abort of the run — the run is server-owned.)
  - On the terminal: advance anchors on `completed` **or** `cancelled` (both are persisted, continuable turns) and refresh the sidebar.
  - **Reconnect:** expose `resumeIfInterrupted()`; the shell calls it on mount, `visibilitychange`→visible, and `online`. If the last message is `status:"streaming"` with a `responseId`, open `streamResume(responseId, lastSequenceNumber, signal)` and feed events through the reducer into `updateLast` (same loop as `send`). Guard against double-subscription with an in-flight ref.
- **Components:** `AssistantMessage` renders a subtle "cancelled" note for `status:"cancelled"` and still shows `MessageControls` (it's a real, continuable turn).

## Data flow

**Normal background turn:** POST `{background:true, stream:true, ...}` → route mints `resp_id`, starts a detached run, returns SSE subscribed from seq 0 → reducer drives the bubble; the run persists in `finally` on `completed`. Anchors advance; sidebar refreshes.

**Reload mid-answer:** the browser drops the SSE; the **run keeps going** server-side. On remount, `resumeIfInterrupted()` sees a `streaming` message with `responseId` + `lastSequenceNumber=N`, opens `GET ?stream=true&starting_after=N`, receives only the missed events then the terminal; the run had already (or will) persist. No tokens lost.

**Plain disconnect, no reconnect:** the run finishes and persists on its own; the conversation appears in the sidebar on next load, fully intact.

**Stop:** `cancelResponse(resp_id)` → `run.cancel.set()` → serializer finalizes a partial, emits terminal `response.incomplete (status cancelled)` → pump persists with status `cancelled`, touches the conversation. The bubble shows "cancelled"; the next send continues from `resp_id`.

## Error handling

- Agent/LLM error → serializer emits `response.failed` (existing) → run status `failed`, persisted with error; reducer shows the error state.
- Resume against an unknown/evicted run → 409; client falls back to `GET /v1/responses/{id}` (non-stream) for the final, or `GET /v1/conversations/{id}` to rebuild.
- Cancel of an already-finished/unknown run → idempotent: returns the stored status or 404; the client treats a 404/finished as "already done".
- `starting_after` beyond the buffer → yields nothing then the terminal once `done` (or closes if already done).

## Testing

- **Backend (`RunManager`):** append+`subscribe(starting_after)` replays only `seq>N` then tails to the live terminal; two concurrent subscribers both receive the full tail; cancel sets the event, the run finalizes `cancelled` and persists a partial (items + `status="cancelled"` response + conversation touched); a normal run persists `completed` even with **no subscriber attached** (the disconnect-is-non-lossy guarantee); TTL eviction drops a finished run after the window. Against both `InMemoryStore` and `SqlStore` where persistence is asserted.
- **Backend (serializer):** with a `cancel` event pre-set, the stream terminates with a `response.incomplete` event whose `response.status=="cancelled"` and `sink["response"]["status"]=="cancelled"`; without cancel, the terminal is `response.completed` (regression guard). Parse every event via the SDK `TypeAdapter(ResponseStreamEvent)`.
- **Backend (routes, `TestClient`):** `background:true, stream:true` streams `response.created`…terminal; `background:true, stream:false` returns an `in_progress` body immediately and the run still persists; `GET ?stream=true&starting_after=N` resumes (assert the first delivered event has `sequence_number>N`); `POST .../cancel` → the stored response ends `cancelled` and the conversation lists it; resume of an evicted/unknown run → 409. Non-background path tests stay unchanged and green.
- **Frontend:** reducer captures `sequence_number` (monotonic) and maps `response.incomplete (cancelled)` → `status:"cancelled"`; `lib/sse.ts` parses chunked SSE into events (incl. split across read boundaries); `streamResume` issues the right URL with `starting_after`; `useResponsesChat` Stop calls `cancelResponse` (and defers until `responseId` is known); reconnect feeds resume events into the existing bubble and advances anchors on the cancelled/completed terminal. Component: `AssistantMessage` renders the cancelled state.

## Implementation sequencing

Two plans, built in order, executed subagent-driven (as before):

1. **Backend resilient runs** — cancel-aware serializer; `RunManager`/`Run` (`app/runs.py`); `background` on `ResponsesRequest`; detached execution + persistence-in-`finally` in `routes/responses.py`; resume (`GET ?stream=true&starting_after`) and `POST .../cancel`; `AppState.runs` wired in `lean_main.py`. Backend stays import-lean and green against both stores.
2. **Frontend reconnect + server-side cancel** — reducer `sequence_number` + `response.incomplete`/`cancelled`; `lib/sse.ts` + `streamResume` + `cancelResponse`; `useResponsesChat` background + Stop→cancel + `resumeIfInterrupted`; shell wires the reconnect triggers; `AssistantMessage` cancelled state. Buildable/runnable against plan 1.

## Risks / open questions

- **Single-process assumption is load-bearing.** Resume and cancel only work when the request hits the same worker that owns the run. Acceptable for the lean/dev deployment; multi-worker needs a shared bus (deferred, noted).
- **Cancel latency.** The serializer races `anext(events)` against `cancel.wait()`, so cancel takes effect at the next event boundary (sub-second under streaming). A stalled upstream LLM call could delay it until the next chunk; acceptable.
- **`response.incomplete` for cancel is a slight semantic stretch** (the event type says "incomplete", the embedded `Response.status` says "cancelled"). Chosen because the SDK has no `response.cancelled` stream event and rejects unknown event types; the authoritative signal is `Response.status`. The reducer keys off `response.status`, not the event type, for the final state.
- **Registry memory growth** is bounded by the TTL sweep + (recommended) a max-runs cap; runaway long conversations don't accumulate because finished runs evict.
- **Persistence timing changes for background.** Background turns persist at finalize (not per-event), identical to today's stream path; sequence-number gaps never occur because the buffer is the single source of emitted events.
