# Standalone Lean Agent Service — Design

**Date:** 2026-06-26
**Status:** Approved (pending spec review)
**Branch:** `personal/yfei/agent-core` (off `personal/yfei/agent-loop`)
**Builds on:** `docs/superpowers/specs/2026-06-26-agent-api-protocol-design.md` (the `AgentEvent` + Responses/Chat protocol). Phases 1–2 (AgentEvent core, dual chat serializer, clean Tool) are done.

## Problem

The agent core is now clean (events, serializers, clean `Tool`), but it still boots inside the full PAI-RAG backend — DB-backed model registry, RAG/knowledgebase/datasource/chatdb/codesandbox factories, vector store, file extraction, and a complex DB-migration story. To move fast on the **agent + `/v1/responses`**, we want a **standalone lean service**: the agent core wired with the minimum it needs, a **fresh schema on an empty DB (no migrations)**, and the non-agent modules relocated to `backend/legacy/`.

## Goals

- A minimal `backend/app/` that runs `/v1/chat/completions` + `/v1/responses` over the existing `AgentEvent` stream, booting **without** RAG/knowledgebase/llamaindex/DB-registry.
- A **fresh, clean schema** created via `create_all()` — no Alembic, no migrations.
- **SQLite by default** (one portable file), Postgres-configurable; no Redis.
- A simple env/request **LLM config** (modern agent-SDK style), not a DB model registry.
- Relocate non-agent modules to `backend/legacy/` (after the lean path runs).

## Scope

**In scope:**
- `backend/app/`: `main.py`, `config.py`, `llm.py`, `builder.py`, `db.py`, `models.py`, `store/`, `routes/`.
- Clean schema: `conversations`, `conversation_items`, `responses` (SQLModel, `create_all`).
- `ResponseStore`: protocol + `InMemoryStore` + `SqlStore` (SQLite/Postgres).
- `/v1/responses` (strict OpenAI Responses, stateful) + `/v1/chat/completions` (reuse `api/protocol/chat_serializer`).
- Make `agent/budgeting` tokenizer optional (char/word fallback) so the service boots without tokenizer files.
- Legacy relocation to `backend/legacy/` (sequenced last).

**Out of scope (future):**
- RAG/knowledgebase/datasource tools (added later as plugins onto the clean `Tool` interface).
- Redis hot-cache layer; multi-tenant; the Conversations REST API; response `cancel`/`input_items`.
- Migrating legacy data (DB assumed empty).

## Decisions

1. **Standalone lean service** reusing `agent/` + `api/protocol/`; new wiring only.
2. **New-first, then relocate** legacy modules to `backend/legacy/`.
3. **Fresh schema, empty DB, `create_all`** — no migrations.
4. **SQLite default** (`aiosqlite`), Postgres-configurable (`asyncpg`); `InMemoryStore` for tests; **no Redis**.
5. **Simple LLM config** from env (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `DEFAULT_MODEL`) + per-request model override.
6. `/v1/responses` **strictly conforms** to OpenAI Responses; PAI-RAG-only signals stay on the opt-in `pai.*` extension channel (per the protocol spec).

## Architecture

### Module layout (`backend/app/`)

```text
backend/app/
  main.py        # minimal FastAPI app; startup → db.create_all()
  config.py      # Settings: OPENAI_BASE_URL, OPENAI_API_KEY, DEFAULT_MODEL, DB_URL, STORE_BACKEND
  llm.py         # build the LLM client from Settings (thin over the existing PaiLlm / openai AsyncClient)
  builder.py     # assemble AgentContext + ToolBox from a request (lean; tools from an in-process registry)
  db.py          # SQLModel async engine + create_all()
  models.py      # conversations / conversation_items / responses (SQLModel tables)
  store/
    base.py      # ResponseStore protocol + dataclasses
    memory.py    # InMemoryStore
    sql.py       # SqlStore (SQLModel)
  routes/
    chat.py      # /v1/chat/completions  → serialize_chat_stream_with_effects / sync
    responses.py # /v1/responses (POST stream+sync, GET, DELETE)
```

**Reused as-is:** `agent/` (loop, events, context, message, tools, budgeting), `api/protocol/chat_serializer.py`.
**New shared:** `api/protocol/responses_serializer.py` (`AgentEvent` → `openai.types.responses` events + `Response` object) — built here, lives with the other serializer.

### LLM config

`config.Settings` (pydantic-settings) reads env: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `DEFAULT_MODEL`, `DB_URL` (default `sqlite+aiosqlite:///./data/agent.db`), `STORE_BACKEND` (`sql`|`memory`). `llm.build_llm(model)` constructs the client from these (reusing the existing `PaiLlm` over an `openai.AsyncOpenAI` pointed at `OPENAI_BASE_URL`). No `llm_service`, no DB model rows, no tenants.

### Builder

`builder.build_context(request, store) -> AgentContext`: resolve input (load prior items via the store if `previous_response_id`/`conversation`), build system prompt + history + current turn + tools (`ToolBox` from a simple in-process tool registry — empty by default; built-in tools added later), `run_vars`. Returns the `AgentContext` the existing `Agent.run` consumes.

### Budgeting tokenizer (lean adjustment)

`agent/budgeting.py` currently calls `get_tokenizer()` (loads a local Qwen tokenizer). Make it optional: if the tokenizer can't load, fall back to a simple length estimate (`len(text)//4` heuristic) so the lean service boots with zero tokenizer files. Behavior under a real tokenizer is unchanged.

## Schema (empty DB, `create_all`, no migrations)

```text
conversations
  id           text  pk          # conv_<uuid>
  user_id      text  null index  # optional caller id (no tenant layer)
  created_at   timestamptz
  updated_at   timestamptz
  metadata     jsonb

conversation_items                # canonical ordered item log (history source of truth)
  id              text pk         # item_/msg_/fc_<uuid>
  conversation_id text fk index
  seq             bigint          # order within conversation
  type            text            # message | reasoning | function_call | function_call_output
  role            text null       # user | assistant | system (message items)
  content         jsonb           # typed payload per `type`
  response_id     text null index # which response produced/consumed it
  created_at      timestamptz

responses
  id                   text pk    # resp_<uuid>
  conversation_id      text null fk index   # set for stored responses (implicit conv if none given)
  previous_response_id text null index
  model                text
  status               text       # in_progress | completed | failed | incomplete
  usage                jsonb null
  error                jsonb null
  created_at           timestamptz
  metadata             jsonb
```

JSON columns: `JSON` type (portable across SQLite/Postgres via SQLModel/SQLAlchemy). `seq` ordering via a per-conversation counter (max(seq)+1 on append).

## Storage

`ResponseStore` protocol (async): `create_conversation`, `append_items`, `get_conversation_items`, `save_response`, `get_response`, `delete_response`, `resolve_history(previous_response_id | conversation)`.
- `SqlStore` — SQLModel async session over `DB_URL` (SQLite `aiosqlite` default, Postgres `asyncpg`); `create_all()` at startup.
- `InMemoryStore` — dict-backed, same protocol, for tests.
- `store=false`: skip all persistence; the response is assembled in-flight only; `GET` → 404; cannot be referenced later.
- Every stored response belongs to a conversation (explicit or auto-created implicit) so `previous_response_id` always resolves through `conversation_items`.

## Endpoints

- **`POST /v1/responses`** — resolve input (store) → `builder.build_context` → `Agent.run` → `ResponsesSerializer` → (stream: SSE `response.*`; sync: `Response` JSON) → persist iff `store=true`. `previous_response_id`/`conversation` linking + precedence (both → must match conversation else 400).
- **`GET /v1/responses/{id}`**, **`DELETE /v1/responses/{id}`** — store-backed; 404 if absent/`store=false`.
- **`POST /v1/chat/completions`** — stateless shim over `Agent.run` via the existing `chat_serializer` (no store).

## Legacy relocation (sequenced last)

After the lean path boots and is green, move non-agent modules into `backend/legacy/` (keep, don't delete): the RAG/knowledgebase/datasource/faq/chatdb/codesandbox tool factories under `tools/`, `rag/`, vector-store/embedding/rerank, file-extraction, the old `service/agent/agent_service.py` + `api/v1/chat.py` (superseded by `app/`), evaluation, and their tests. The partition rule: **anything the lean `backend/app/` + `agent/` + `api/protocol/` path does not import is legacy.** Fix imports as modules move; the lean app must keep booting + tests green after each move.

## Testing

- **Schema/store:** `InMemoryStore` and `SqlStore` (against an in-memory SQLite `sqlite+aiosqlite:///:memory:`) — CRUD, `seq` ordering, `previous_response_id`/`conversation` resolution incl. the cross-conversation 400 and no-double-prepend, `store=false` → no row + `GET` 404.
- **Responses serializer:** conformance vs `openai.types.responses`; **OpenAI-SDK integration test** (parse sync `Response` + streaming events; default stream is pure-OpenAI, no `pai.*`).
- **Endpoints:** POST stream + sync, GET, DELETE, linking; `/v1/chat/completions` still green via the existing serializer tests.
- **Lean boot:** the app imports + starts with `STORE_BACKEND=memory` and a fake/echo LLM, with **zero** RAG/llamaindex/tokenizer files present.

## Migration sequencing (each step green)

1. `app/config.py` + `app/db.py` + `app/models.py` (schema) + `create_all`.
2. `app/store/` (protocol + InMemory + Sql) with tests.
3. `app/llm.py` + `app/builder.py` (lean wiring) + budgeting tokenizer fallback.
4. `api/protocol/responses_serializer.py` (`AgentEvent` → Responses) with conformance + SDK tests.
5. `app/routes/` + `app/main.py` (both endpoints) — lean app boots and serves.
6. Legacy relocation to `backend/legacy/`, incrementally, keeping the lean app green.

## Risks / open questions

- **Reusing `PaiLlm`** may drag transitive deps (model classes, dashscope shims); if heavy, write a thin `openai.AsyncOpenAI` client adapter that emits the same chunk shape `agent/_stream_turn` consumes. Confirm during step 3.
- **`agent/` transitive deps:** `agent/agent.py` imports `extensions/trace/*`; verify the trace extension is lean enough to keep, else stub it behind a flag.
- **Legacy move is large** and risky; sequenced last and done incrementally so the lean app never breaks.
