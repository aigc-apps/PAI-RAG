# New Agent Chat Frontend — Design

**Date:** 2026-06-26
**Status:** Approved (pending spec review)
**Branch:** `personal/yfei/agent-core`
**Builds on:** the lean agent service (`backend/app/`, `/v1/responses` with `reasoning.summary` streaming) from `docs/superpowers/specs/2026-06-26-standalone-lean-agent-design.md` and the two implementation plans `2026-06-26-lean-agent-foundation.md` + `2026-06-26-lean-agent-service.md`.

## Problem

The existing chat UI (`frontend/`) is a Next.js app deeply coupled to `@assistant-ui/react` and the **legacy** custom chat protocol (`/v1/chat/completions` with bespoke `actions`/`observation`/`reasoning_content` SSE chunks). Its runtime adapter (`frontend/app/runtime/usePaiChatThreadRuntime.tsx`, ~778 lines) hand-parses that protocol. It cannot consume the new lean `/v1/responses` (OpenAI Responses, `reasoning.summary` events) without a substantial rewrite, and it carries RAG/KB/tenant machinery the lean service doesn't yet support.

We want a **brand-new, standalone frontend** — built from scratch, without assistant-ui — that talks to the lean `/v1/responses` service. It starts minimal (clean agent chat) and later absorbs the good parts of the old UI selectively ("取其精华").

## Goals

- A new, separate frontend app at **`newfrontend/`** (does not touch the existing `frontend/`).
- A clean agent chat over the lean **`/v1/responses`** stream: streaming Markdown answers + collapsible reasoning rendered from the `reasoning.summary` events.
- Multi-turn continuity via `previous_response_id`, a conversation-history sidebar, stop/regenerate/copy controls, and a model selector.
- A thin **backend Conversations read/list API** on the lean service to power the sidebar (the sidebar's data source — currently absent).

## Scope

**In scope:**
- `newfrontend/` — Vite + React + TypeScript SPA (no Next.js, no assistant-ui).
- Streaming consumption of `/v1/responses` via the official `openai` JS SDK (`responses.stream()`), including `reasoning.summary` events.
- v1 features: streaming text+Markdown, collapsible streaming reasoning, multi-turn (`previous_response_id`), conversation-history sidebar, stop/regenerate/copy, model selector.
- Backend additions to the lean service: add `title` + `last_response_id` columns to `Conversation`; accept an optional `user_id` on `ResponsesRequest` and persist it on the conversation; persist/`touch` `Conversation` rows on stored turns (title from first user message, bump `updated_at` + `last_response_id` each turn); `list_conversations`/`get_conversation`/`delete_conversation` store methods; `GET /v1/conversations`, `GET /v1/conversations/{id}`, `DELETE /v1/conversations/{id}` routes.

**Out of scope (v1; ported later from the old UI):**
- Attachments / multimodal input.
- Tool / RAG / KB / search / chatdb / MCP UI (the lean service has no tools yet — added later as plugins).
- Tenant / workspace auth (`X-TENANT-ID`), i18n, message editing/branching, title auto-generation via the model.
- Touching or migrating the existing `frontend/` app.

## Decisions

1. **Separate greenfield app** `newfrontend/`, not a route inside `frontend/`. The old app stays running and is migrated from selectively later.
2. **Vite + React + TS SPA** (not Next.js): the backend is a pure API, no SSR needed; Vite is lighter and faster to iterate, matching the lean ethos. Reuse the proven non-assistant-ui stack: Tailwind v4, Radix primitives, react-markdown + remark-gfm + syntax highlighter, lucide icons, sonner, zustand.
3. **Consume the stream with the official `openai` JS SDK** (`client.responses.stream(...)`), not hand-rolled SSE. The SDK yields strongly-typed `response.*` events (including the reasoning-summary family), eliminating the legacy hand-parser. Browser client: `baseURL` → the lean service via the Vite dev proxy, placeholder `apiKey` (the real key lives server-side in the lean service), `dangerouslyAllowBrowser: true` (acceptable: no auth layer yet, browser only ever talks to our own lean service).
4. **Reasoning via the `reasoning.summary` channel** — consistent with what the serializer now emits (`response.reasoning_summary_part.added` → `...summary_text.delta*` → `...summary_text.done` → `...part.done`, wrapped in reasoning `output_item.added/done`).
5. **Persistence/threading uses the lean service's native model** (`conversation` + `previous_response_id` + the new Conversations API), NOT the legacy `/api/threads/*` endpoints. **Continuation is anchored on `conversation`** (the client keeps the `conversation_id` and sends it on every follow-up turn), with `previous_response_id` as a secondary hint. This decouples continuation from any single response so a Stopped/failed/deleted response never breaks the thread.
6. **No tenant/auth, but per-browser isolation via a local `user_id`.** v1 has no real auth, but the client generates a stable `user_id` (UUID in `localStorage`), sends it on `/v1/responses` (persisted on the conversation) and as the `?user_id=` filter on `GET /v1/conversations`, so each browser sees only its own conversations. Tenant/workspace remains out of scope.
7. **Stop is local-only in v1.** Aborting a stream keeps the partial assistant message as a client-only, non-persisted, non-continuable bubble (the backend persists only on full stream completion — see Stop semantics under Data flow). Cancel/incomplete server-side persistence is deferred.

## Architecture

### Backend additions (lean service `backend/app/`)

The conversations table is currently never written (routes persist only items + responses under a minted conversation_id). To back a sidebar, conversations must become real, listable rows, and a continuation must be reconstructable after a history load.

- **`Conversation` model gains two columns:** `title: Optional[str]` and `last_response_id: Optional[str]` (both nullable; add to the SQLModel + `create_all`, no migration). `ResponsesRequest` gains an optional `user_id: Optional[str]` (persisted on the conversation).
- **Persist `Conversation` rows.** On a `store=true` turn, the route calls `ensure_conversation(conversation_id, user_id, title)` BEFORE appending items: create-if-absent with `title` = first user message text trimmed to ~80 chars (set only on creation, never overwritten) and `user_id` from the request. After saving the response, set the conversation's `last_response_id = response_id` and bump `updated_at`. `store=false` writes no conversation (unchanged).
- **Store protocol additions** (`ResponseStore`, implemented in both `InMemoryStore` and `SqlStore`):
  - `ensure_conversation(conversation_id: str, user_id: Optional[str], title: Optional[str]) -> Conversation` — create-if-absent (idempotent; does not overwrite an existing title).
  - `touch_conversation(conversation_id: str, last_response_id: str) -> None` — set `last_response_id` + bump `updated_at`.
  - `list_conversations(user_id: Optional[str], limit: int, offset: int) -> List[Conversation]` — filtered by `user_id` when given, newest `updated_at` first.
  - `get_conversation(conversation_id: str) -> Optional[Conversation]`.
  - `delete_conversation(conversation_id: str) -> None` — removes the conversation and its items/responses.
- **Item → UI message grouping algorithm** (used by `GET /v1/conversations/{id}`). The item log is appended per turn, every item of a turn sharing one `response_id` (user `message` + optional `reasoning` + `function_call`/`function_call_output` + assistant `message`). To reconstruct turns:
  1. Load all `conversation_items` ordered by `seq`, and all `responses` for the conversation (for per-turn `status`/`previous_response_id`).
  2. Group items by `response_id`, preserving first-seen `seq` order across groups (chronological).
  3. Each group yields up to two UI messages: a `user` message (text from the user `message` item) and an `assistant` message — `text` from the assistant `message` item (may be empty/absent on a failed turn), `reasoning` from the `reasoning` item's text. The assistant message carries `response_id`, `previous_response_id` and `status` (from the matching `responses` row).
  4. **`function_call`/`function_call_output` items are skipped in the v1 UI mapping** (tool UI is out of scope — see Risks).
- **Routes** (`backend/app/routes/conversations.py`):
  - `GET /v1/conversations?user_id=&limit=&offset=` → `{ "data": [ { id, title, created_at, updated_at, last_response_id } ] }`.
  - `GET /v1/conversations/{id}` → `{ id, title, created_at, updated_at, latest_response_id, messages: [ { role: "user", text, response_id } | { role: "assistant", text, reasoning, response_id, previous_response_id, status } ] }` where `latest_response_id == last_response_id` (the anchor for continuing the thread). 404 if absent.
  - `DELETE /v1/conversations/{id}` → 200 / 404.
- Wire the new router into `app/lean_main.py`.

### Frontend (`newfrontend/`)

```text
newfrontend/
  index.html
  vite.config.ts            # dev proxy /v1 -> lean service; tailwind plugin
  src/
    main.tsx                # React root
    api/
      client.ts            # openai SDK client (baseURL via proxy)
      conversations.ts     # GET/DELETE /v1/conversations[/{id}]
    store/
      chat.ts              # zustand: messages, status, model, last_response_id
      conversations.ts     # zustand: conversation list, selected id
    hooks/
      useResponsesChat.ts  # send a turn, consume the stream, stop/regen
    stream/
      reducer.ts           # PURE: (state, response.* event) -> message state
    components/
      App.tsx
      Sidebar.tsx          # conversation list + "New chat" + delete
      ChatView.tsx
      MessageList.tsx
      UserMessage.tsx
      AssistantMessage.tsx # Markdown body + CollapsibleReasoning + ToolCall placeholder
      CollapsibleReasoning.tsx
      Markdown.tsx         # react-markdown + gfm + code highlight
      Composer.tsx         # input + send/stop
      ModelSelector.tsx
      MessageControls.tsx  # copy / regenerate
```

**Streaming reducer (the testable core).** `stream/reducer.ts` is a pure function mapping each `response.*` event onto the in-flight assistant message: `response.created` captures the `response.id`; reasoning-summary events accumulate into the message's `reasoning` (with a streaming/`done` status driving the collapsible); `response.output_text.delta` appends to `text`; `response.completed`/`response.failed` set final status + usage/error. `useResponsesChat` owns the openai SDK stream, feeds events to the reducer, and exposes `send`, `stop` (abort the stream), `regenerate`.

### Data flow — one turn

The client keeps two anchors: `conversation_id` (the thread) and `last_response_id` (the latest *persisted* response). A stable `user_id` (localStorage UUID) is sent on every request.

1. User submits → optimistically append a `user` message → set status `streaming`.
2. `client.responses.stream({ model, input: text, user_id, store: true, conversation: conversation_id /* if set */, previous_response_id: last_response_id /* if set */ })`. The first turn of a fresh chat sends neither anchor.
3. Events drive the reducer: `response.created`/`completed` carry `response.conversation.id` and `response.id`. Reasoning streams first (collapsible, auto-collapses on done), then the Markdown answer. On `completed`: set `conversation_id = response.conversation.id` (first turn) and `last_response_id = response.id`, then refresh the sidebar (the backend created/touched the conversation).
4. **Stop** aborts the stream. Per Stop semantics (below), the partial assistant message is kept **client-only**: `conversation_id`/`last_response_id` are NOT advanced, so the next send continues from the last *persisted* response (or starts a fresh conversation if the stopped turn was the first). **Regenerate** re-sends the last user input with the same anchors that produced the turn being regenerated (`conversation` + the `previous_response_id` of that turn). **Copy** writes the message text to the clipboard.
5. **History:** selecting a conversation in the sidebar calls `GET /v1/conversations/{id}`, renders past turns from `messages`, and sets `conversation_id = id` and `last_response_id = latest_response_id` so new turns continue the thread.

## Error handling

- Stream/network error → the assistant message shows an error state (from `response.failed` or a thrown SDK error); a sonner toast surfaces the message; the input is re-enabled.
- `GET /v1/conversations/{id}` 404 → toast + clear selection.

### Stop semantics (v1)

The route persists a stored response only AFTER the serializer stream completes (`responses.py` persists in the stream `gen()` after the `async for` finishes). When the client aborts, that generator is cancelled and **nothing is persisted server-side** — no conversation row, no items, no response. v1 makes this explicit and consistent rather than papering over it:

- A Stopped turn's partial assistant message is kept **client-only**: visibly marked (e.g. "stopped"), **not** added to the sidebar, and **not** continuable.
- The client does NOT advance `conversation_id`/`last_response_id` on Stop, so the next send continues from the last persisted response — or, if the Stopped turn was the first turn of a fresh chat, the next send starts a brand-new conversation (the dangling local bubble is discarded on the next send or on conversation switch).
- Server-side cancel/`incomplete`-status persistence (so a Stopped turn could be saved and continued) is explicitly deferred beyond v1.

## Testing

- **Backend:** `ensure_conversation` idempotency (does not overwrite an existing title), `touch_conversation` sets `last_response_id` + bumps `updated_at`, `list_conversations` ordering + pagination + `user_id` filtering, `get_conversation` reconstructs turns via the grouping algorithm (multi-turn ordering, reasoning surfaced, per-assistant-turn `response_id`/`previous_response_id`/`status`, a failed turn rendered with empty/absent text + status `failed`, `function_call` items skipped), `latest_response_id` returned, title derivation (first user message, trimmed), `delete_conversation` cascade, `store=false` creates no conversation — against both `InMemoryStore` and `SqlStore`; endpoint tests via FastAPI `TestClient` including the `user_id` round-trip.
- **Frontend:** unit-test `stream/reducer.ts` against recorded `/v1/responses` event sequences (a plain text turn; a reasoning+text turn exercising the `reasoning.summary` family; a `response.failed` turn) — assert the resulting message state and that `conversation_id`/`last_response_id` are captured from `response.created`/`completed`. Continuation test: a second `send` carries the captured `conversation` anchor. Stop test: aborting does NOT advance `conversation_id`/`last_response_id` and the partial bubble is marked client-only. Component tests: `CollapsibleReasoning` (streaming vs done collapse), `Markdown` rendering + code copy, `MessageControls` (copy/regenerate), `Composer` (send/stop toggle). A light smoke test of `useResponsesChat` against a mocked stream.

## Implementation sequencing

Two plans, built in order (the backend API is the sidebar's prerequisite):

1. **Backend Conversations API** — `Conversation.title` column, store methods (`list_conversations`/`get_conversation`/`delete_conversation`/`ensure`+`touch`), the persist-path change (create/touch conversation on stored turns), `routes/conversations.py`, wiring into `lean_main.py`, tests. The lean service stays green and import-lean.
2. **`newfrontend/` Vite SPA** — scaffold, openai SDK client + conversations API client, the pure stream reducer + `useResponsesChat`, the component tree, zustand stores, Vite proxy, tests. Buildable/runnable against the lean service.

## Risks / open questions

- **`openai` SDK Responses streaming shape fidelity:** the SDK's `responses.stream()` must parse exactly the events our serializer emits (reasoning-summary family, `output_item` indices, `completed`/`failed`). Lock with the reducer's recorded-sequence tests; if the SDK rejects an event, the serializer (not the test) is the source of truth to reconcile.
- **`dangerouslyAllowBrowser` + local `user_id` (not auth):** the `user_id` is a client-generated localStorage UUID for per-browser sidebar isolation, NOT a security boundary — anyone can pass any `user_id`. Acceptable while the lean service is unauthenticated and browser-only-to-our-service; a real auth layer (server-side key, authenticated `user_id`) is deferred.
- **Title quality:** first-user-message truncation is a deliberate v1 simplification; model-generated titles are deferred.
- **Tool items: live/history parity is incomplete, and that's fine for v1.** newfrontend v1 **ignores** `function_call`/`function_call_output` items in both the live stream and the history mapping (tool UI is out of scope). A future tool UI will require **serializer parity first**: the streaming serializer currently writes `ToolResult` only into the store accumulator and does NOT emit `function_call_output` output-item stream events, so a tool turn renders differently live vs. on history reload. Closing that parity (emit `function_call_output` `output_item.added/done` in the stream) is a prerequisite for any tool UI and is tracked for the tool-support plan, not this one.
- **Conversation item → UI message mapping** must round-trip the same store-item shapes the serializer writes; the grouping algorithm (Backend additions) is the shared contract — keep it aligned with the serializer's `store_items` shapes.
