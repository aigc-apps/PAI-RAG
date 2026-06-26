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
- Backend additions to the lean service: persist `Conversation` rows (id, user_id, title, timestamps) on stored turns; `list_conversations`/`get_conversation`/`delete_conversation` store methods; `GET /v1/conversations`, `GET /v1/conversations/{id}`, `DELETE /v1/conversations/{id}` routes.

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
5. **Persistence/threading uses the lean service's native model** (`previous_response_id` + `conversation` + the new Conversations API), NOT the legacy `/api/threads/*` endpoints.
6. **No tenant/auth** in v1 — a deliberate simplification over the old UI.

## Architecture

### Backend additions (lean service `backend/app/`)

The conversations table is currently never written (routes persist only items + responses under a minted conversation_id). To back a sidebar, conversations must become real, listable rows.

- **Persist `Conversation` rows.** When `store=true` and the turn opens a NEW conversation (no `previous_response_id`/`conversation` linking), create a `Conversation` row with: `id` (the minted id), `user_id` (optional, from request), `title` (first user message text, trimmed to ~80 chars), `created_at`/`updated_at`. On every stored turn, bump the conversation's `updated_at`. Store the title in a dedicated `title` column (add to the `Conversation` SQLModel) — simplest and queryable. `store=false` writes no conversation (unchanged).
- **Store protocol additions** (`ResponseStore`, implemented in both `InMemoryStore` and `SqlStore`):
  - `list_conversations(user_id: Optional[str], limit: int, offset: int) -> List[Conversation]` — newest `updated_at` first.
  - `get_conversation(conversation_id: str) -> Optional[Conversation]`.
  - `delete_conversation(conversation_id: str) -> None` — removes the conversation and its items/responses.
  - `ensure_conversation(conversation_id, user_id, title)` / `touch_conversation(conversation_id)` — create-if-absent + bump `updated_at` (called from the route's persist path).
- **Routes** (`backend/app/routes/conversations.py`):
  - `GET /v1/conversations?user_id=&limit=&offset=` → `{ "data": [ { id, title, created_at, updated_at } ] }`.
  - `GET /v1/conversations/{id}` → the conversation plus its turns, mapped from `conversation_items` to a UI-friendly message list: `{ id, title, messages: [ { role, text, reasoning, tool_calls } ] }` (group `function_call`/`function_call_output` with their assistant turn; surface reasoning text).
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

1. User submits → optimistically append a `user` message → set status `streaming`.
2. `client.responses.stream({ model, input: text, previous_response_id, store: true })`.
3. Events drive the reducer: reasoning bubble streams first (collapsible, auto-collapses on done), then the Markdown answer streams; on `completed`, persist `last_response_id = response.id` for the next turn and refresh the sidebar (the backend created/touched the conversation).
4. **Stop** aborts the stream (the partial assistant message is kept). **Regenerate** re-sends the last user input with the `previous_response_id` from before the last assistant turn. **Copy** writes the message text to the clipboard.
5. **History:** selecting a conversation in the sidebar calls `GET /v1/conversations/{id}`, renders past turns, and sets `last_response_id` to that conversation's latest response so new turns continue it.

## Error handling

- Stream/network error → the assistant message shows an error state (from `response.failed` or a thrown SDK error); a sonner toast surfaces the message; the input is re-enabled.
- `GET /v1/conversations/{id}` 404 → toast + clear selection.
- Aborted stream (user Stop) is not an error — the partial message is finalized as-is.

## Testing

- **Backend:** `list_conversations` ordering + pagination, `get_conversation` item→message mapping, title derivation (first user message, trimmed), `delete_conversation` cascade, `store=false` creates no conversation, conversation `updated_at` bumped per turn — against both `InMemoryStore` and `SqlStore`; endpoint tests via FastAPI `TestClient`.
- **Frontend:** unit-test `stream/reducer.ts` against recorded `/v1/responses` event sequences (a plain text turn; a reasoning+text turn exercising the `reasoning.summary` family; a `response.failed` turn) — assert the resulting message state. Component tests: `CollapsibleReasoning` (streaming vs done collapse), `Markdown` rendering + code copy, `MessageControls` (copy/regenerate), `Composer` (send/stop toggle). A light smoke test of `useResponsesChat` against a mocked stream.

## Implementation sequencing

Two plans, built in order (the backend API is the sidebar's prerequisite):

1. **Backend Conversations API** — `Conversation.title` column, store methods (`list_conversations`/`get_conversation`/`delete_conversation`/`ensure`+`touch`), the persist-path change (create/touch conversation on stored turns), `routes/conversations.py`, wiring into `lean_main.py`, tests. The lean service stays green and import-lean.
2. **`newfrontend/` Vite SPA** — scaffold, openai SDK client + conversations API client, the pure stream reducer + `useResponsesChat`, the component tree, zustand stores, Vite proxy, tests. Buildable/runnable against the lean service.

## Risks / open questions

- **`openai` SDK Responses streaming shape fidelity:** the SDK's `responses.stream()` must parse exactly the events our serializer emits (reasoning-summary family, `output_item` indices, `completed`/`failed`). Lock with the reducer's recorded-sequence tests; if the SDK rejects an event, the serializer (not the test) is the source of truth to reconcile.
- **`dangerouslyAllowBrowser` + no auth:** acceptable while the lean service is unauthenticated and browser-only-to-our-service; revisit when an auth layer lands (move the key server-side behind a tiny proxy route).
- **Title quality:** first-user-message truncation is a deliberate v1 simplification; model-generated titles are deferred.
- **Conversation item → UI message mapping** must round-trip the same store-item shapes the serializer writes (`message`/`function_call`/`function_call_output`/`reasoning`); reuse the shared contract.
