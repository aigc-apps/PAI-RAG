# Agent API Protocol — Design

**Date:** 2026-06-26
**Status:** Approved (pending spec review)
**Scope:** A clean, self-contained **core agent** plus a dual public API: `/v1/responses` (flagship, stateful, typed events) and `/v1/chat/completions` (stateless compat shim). Decoupled from the existing RAG stack.

## Problem

The streaming contract between the LLM client, the agent loop, and the API serializer is an *implicit, field-presence* protocol over one overloaded `TextChunk`. A chunk's meaning is inferred from which optional field is set (`delta`? `usage`? `tool_calls`?). This has produced four bugs that are really one design flaw: dropped token usage, content-vs-usage coupling, an uncaught idle-timeout, and an invisible-timeout error message (message put in `.error_message` while the serializer renders `.delta`).

Separately, the product is an **agent served as an API** (the UI is a demo), so the public surface should be a real standard — OpenAI **Responses** (agent-native, typed events, optional state) — with **Chat Completions** kept for compatibility.

## Goals

- One internal **`AgentEvent`** discriminated union that the agent loop emits; usage, errors, and tool steps are first-class events, never inferred.
- Two thin serializers map that one stream to `/v1/responses` and `/v1/chat/completions`.
- An **agent-centric** package layout: the agent is the core; guardrail / context / tools / memory are its submodules; HTTP/serialization is a thin shell.
- A **clean core agent** with no dependency on llamaindex, `pairag.knowledgebases`, or the RAG tooling.

## Scope

**In scope (this version):**
- Core agent: loop, `AgentEvent`, context/`build_messages`, `Message`, a **clean `Tool`/`ToolBox`** abstraction (no `llama_index`), a guardrail interface, budgeting.
- Memory: a hybrid (Redis hot + SQL durable) **Response/Conversation store**.
- Protocol: `AgentEvent` → Responses serializer, `AgentEvent` → Chat-Completions serializer.
- Endpoints: `POST /v1/responses` (stream + sync), `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, and `/v1/chat/completions`.
- State linking via `previous_response_id` and `conversation`.
- Frontend `ChatModelAdapter` rewritten to consume `/v1/responses`.

**Out of scope (future adapters / follow-ups):**
- Knowledgebase / RAG / datasource tools, llamaindex, `pairag.knowledgebases` — the agent exposes a generic tool interface; concrete RAG tools are adapted onto it later.
- Conversations REST API (`POST /v1/conversations`, item CRUD), response `cancel`, `input_items` listing.
- Migrating the existing `tools/` and guardrail service code under `agent/` (incremental follow-up).

## Decisions (from brainstorming + review)

1. **Stateful** `/v1/responses`; **stateless** `/v1/chat/completions` shim over the same spine.
2. **Hybrid storage**: Redis hot cache + durable SQL.
3. **Core + retrieve** REST surface (POST/GET/DELETE + linking); defer cancel/input-items/Conversations API.
4. **Wire schema from `openai-python`** (`openai.types.responses.*`, verified present in openai 2.30.0); internal `AgentEvent` union stays wire-agnostic.
5. **Agent-centric** layout; build new code in the shape, migrate existing modules incrementally.
6. **Clean core agent**: own `Tool`/`ToolBox`, freshly-designed schema, no RAG/llamaindex deps.

## Architecture

`Agent.run(ctx)` is the single producer of an `AgentEvent` stream. Everything else is mapping or storage.

```text
POST /v1/responses ─┐
POST /v1/chat/completions ─┤
                          ▼
   resolve input (load prior items from store if previous_response_id/conversation)
                          ▼
   build AgentContext → Agent.run(ctx) → AsyncIterator[AgentEvent]    ← the spine
                          ▼
        ┌──────────── split by endpoint ────────────┐
   ResponsesSerializer                      ChatCompletionsSerializer
   AgentEvent → openai.types.responses      AgentEvent → chat.completion.chunk
   SSE + final Response object              text deltas + final usage chunk
        ▼                                          ▼
   ResponseStore.save() if store=true       (no store r/w; stateless)
```

### Package layout (agent-centric)

```text
backend/agent/
  core/
    agent.py        # Agent.run(ctx) -> AsyncIterator[AgentEvent]
    events.py       # AgentEvent discriminated union
    context.py      # AgentContext + build_messages
    message.py      # Message + from_thread
    builder.py      # assembles AgentContext + ToolBox for a request
  model/            # LLM adapter: provider chunks -> AgentEvent at the seam
  tools/
    base.py         # clean Tool + ToolBox + dispatch (NO llama_index)
  guardrail/        # input/output moderation interface
  memory/
    budgeting.py    # token budgeting
    store.py        # hybrid Redis+SQL Response/Conversation store
  prompts.py

backend/api/v1/
  responses.py                 # POST/GET/DELETE /v1/responses
  chat.py                      # /v1/chat/completions
  protocol/
    responses_serializer.py
    chat_serializer.py

backend/db/models/agent/       # clean schema: conversations, conversation_items, responses
```

Shared plumbing the agent merely calls (SQL engine, Redis, `openai` types) stays outside `agent/`.

## The `AgentEvent` union

Discriminated by `type`; each event carries exactly its own payload.

```text
run.started      { response_id, conversation_id? }
text.delta       { text }
reasoning.delta  { text }
tool.started     { call_id, name }
tool.completed   { call_id, name, arguments }      # full args resolved
tool.result      { call_id, name, ok, output?, error? }
run.completed    { usage{input,output,total}, finish_reason }   # terminal success
run.failed       { message, type }                 # terminal error (timeout/guardrail/llm)
```

`tool.args.delta { call_id, delta }` is reserved for future streamed-arg support; not in v1 (the model client hands us coalesced tool calls).

### Event → wire mapping

| `AgentEvent` | `/v1/responses` | `/v1/chat/completions` |
|---|---|---|
| `run.started` | `response.created` + `response.in_progress` | (none / first role chunk) |
| `text.delta` | `response.output_text.delta` | chunk `delta.content` |
| `reasoning.delta` | `response.reasoning_summary_text.delta` | chunk `delta.reasoning_content` |
| `tool.started` | `response.output_item.added` (`function_call`) | chunk `delta.tool_calls` + custom `pai.tool.progress` event |
| `tool.completed` | `response.function_call_arguments.done` + `output_item.done` | chunk `delta.tool_calls` |
| `tool.result` | `function_call_output` item (**extension**, see below) | custom `pai.observation` event (dropped for pure-OpenAI clients) |
| `run.completed` | `response.completed` (usage on `Response`) | final chunk `finish_reason=stop` + `usage` |
| `run.failed` | `response.failed` / `response.error` | error chunk (message in one field) |

Serializers `match event.type` — no field-presence inference. This makes the four historical bugs structurally impossible.

**Server-side-tool extension note:** standard Responses treats `function_call_output` as an *input* item supplied for the next model call. Our agent executes tools **server-side**, so we surface `function_call_output` inside `Response.output` to make the executed trace visible. This is a documented **PAI-RAG extension** to Responses semantics; clients should not assume vanilla-OpenAI behavior for these items.

## `/v1/responses` behavior

**Request:** OpenAI Responses core fields — `input` (string | input-item array), `model`, `instructions`, `stream`, `previous_response_id`, `conversation`, `store` (default `true`), `metadata`. Attachments arrive as input image/file content parts. (Agent tool configuration is generic in this version; concrete RAG knobs are out of scope.)

**Flow:** resolve input → build `AgentContext` → `Agent.run` → `ResponsesSerializer` → (stream: SSE `response.*`; sync: aggregate to one `Response`) → persist iff `store=true`.

**`store` semantics (explicit):**
- `store=true` (default): write-through to Redis (short TTL) **and** durable SQL. Retrievable via `GET`; usable as a future `previous_response_id`.
- `store=false`: **no durable SQL write.** At most an ephemeral Redis entry for in-request assembly, evicted on completion. `GET /v1/responses/{id}` → **404**. The response **cannot** be referenced by `previous_response_id` afterward.

**State linking & precedence:**
- Every **stored** (`store=true`) response belongs to a conversation: the explicit `conversation` if given, otherwise an **implicit conversation auto-created** for it — so its `conversation_items` always have a home and `previous_response_id` always resolves through a conversation. (Only `store=false` ephemeral responses have no conversation.)
- `conversation` → load that conversation's ordered `conversation_items` as history.
- `previous_response_id` (no conversation) → resolve via that response's (implicit) conversation, loading items up to and including it — a standalone chain.
- **Both provided** → the referenced previous response **must belong to that conversation**, else **400**. Never prepend both sources (no double-prepend).
- Long histories are trimmed by the agent's existing budgeting, so chains stay bounded.

**`GET /v1/responses/{id}`** → Redis→SQL lookup; 404 if absent or `store=false`. **`DELETE`** → evict from both.

`Response.output` is a typed item list: `message`(output_text) / `reasoning` / `function_call` / `function_call_output`.

## `/v1/chat/completions` shim

Same spine, **stateless with respect to the Responses store** (never reads/writes it). Builds `AgentContext` from the request `messages`, runs `Agent.run` → `ChatCompletionsSerializer` → `chat.completion.chunk`. Subsumes today's `convert_gen_to_stream_chat_completions`, but driven by typed `AgentEvent`s, so the usage-drop / content-swallow / invisible-timeout bugs are gone.

**Legacy Redis session-history** (the current `user_id`/`session_id` read/write) is a **separate compatibility flag**, default **preserved**, so migrating to the new serializer does not silently change multi-turn chat behavior. It is independent of the Responses store.

## Storage / DB schema (clean redesign)

Designed fresh — not mirroring existing `pai_*` tables.

```text
conversations
  id            text  pk        # conv_...
  tenant_id     text  index
  user_id       text  index
  created_at    timestamptz
  metadata      jsonb
  last_response_id text null     # convenience pointer

conversation_items                # canonical item log (ordered)
  id            text  pk          # item_... / msg_... / fc_...
  conversation_id text fk index
  seq           bigint            # ordering within conversation
  type          text              # message | reasoning | function_call | function_call_output
  role          text null         # user | assistant | system (for message)
  content       jsonb             # typed payload per `type`
  response_id   text null index   # which response produced/consumed it
  created_at    timestamptz

responses
  id            text  pk          # resp_...
  conversation_id text null fk index  # set for all stored responses (implicit conv if none given); null only for store=false
  previous_response_id text null index
  tenant_id     text  index
  user_id       text  index
  model         text
  status        text              # in_progress | completed | failed | incomplete
  usage         jsonb null
  error         jsonb null
  created_at    timestamptz
  metadata      jsonb
```

- `conversation_items` is the source of truth for history (supports future list/delete/tool-output/incomplete items cleanly). `responses` references items via `response_id`; it does **not** store input/output as opaque JSON blobs.
- **Redis hot cache:** `response:{id}` → serialized `Response`, short TTL; write-through on save, Redis→SQL fallback on read.

## Compatibility stance

Target is **schema-compatible / client-compatible**, **not** byte-identical. Chunk index, empty deltas, the final usage chunk, and custom events may differ in field order/defaults from the legacy output. **Golden tests** pin representative chunk sequences; **OpenAI-SDK integration tests** (below) verify real-client behavior.

## Testing

- **`AgentEvent` emission** (`FakeLLM` + dummy tools): correct event sequence for text-only, tool-call→answer, `run.failed` (timeout/guardrail), usage on `run.completed`.
- **`ResponsesSerializer` conformance:** emitted payloads validate against `openai.types.responses.*`; event ordering for stream.
- **`ChatCompletionsSerializer`:** golden chunk sequences; **regression locks** for the usage-only terminal chunk and the error-message field (invisible-timeout).
- **OpenAI-SDK integration:** drive a test server and parse responses with the **official `openai` client** — both the sync `Response` JSON and the streaming events (covers SSE ordering, `data: [DONE]`, client-side assembly). This is required in addition to Pydantic conformance.
- **Store:** save/get/delete; Redis→SQL fallback; `store=false` → no SQL row + `GET` 404; `previous_response_id`/`conversation` resolution incl. the cross-conversation **400** and no-double-prepend.
- **Endpoints:** POST stream + sync, GET, DELETE, linking.

## Migration sequencing (each step green)

1. `agent/core/events.py` (`AgentEvent`) + make `Agent.run` yield it, with a temporary `AgentEvent → legacy-chunk` shim so the current `chat.py` keeps working.
2. `ChatCompletionsSerializer` (typed-event driven) + repoint `chat.py`; delete the temp shim. (Usage/timeout regressions fixed here.)
3. Clean `Tool`/`ToolBox` (`agent/tools/base.py`) replacing the `llama_index` wrapper at the agent boundary.
4. `db/models/agent/*` + `agent/memory/store.py` (hybrid store).
5. `ResponsesSerializer` + `/v1/responses` endpoint (POST/GET/DELETE, store semantics, linking).
6. Frontend `ChatModelAdapter` → `/v1/responses`.
7. Follow-ups (separate specs): migrate `tools/`, guardrail under `agent/`; adapt RAG/knowledgebase tools onto the clean `Tool` interface; Conversations REST API.

## Risks / open questions

- **Clean `Tool` vs existing `FunctionTool`:** step 3 introduces a parallel tool abstraction; the existing RAG tools keep using `FunctionTool` until adapted (out of scope). The boundary must be explicit so both can coexist during migration.
- **`reasoning` mapping:** `response.reasoning_summary_text.delta` vs a full reasoning item — confirm which the target models populate; default to summary-text deltas.
- **Conversation item growth:** without the Conversations API, items still accumulate; rely on budgeting for context, and a retention/TTL policy for storage (define in the plan).
