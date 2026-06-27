# User Memory — Design (v1)

**Date:** 2026-06-27
**Status:** Approved by delegation (design agreed in conversation; user said "ok")
**Branch:** `personal/yfei/agent-core`
**Builds on:** the lean agent service — the store/Conversations layer (`user_id` already exists on `ResponsesRequest`/`Conversation`), SOUL (`render_system_prompt`), the detached `RunManager`, and model providers.

## Problem

The service has a notion of `user_id` (a tag for conversation isolation) but **no user record and no user memory**. We want a per-user long-term memory that persists across conversations: facts the agent learns about a user, injected into future turns — and the chat interface should align with OpenAI's end-user identifier convention, with stored messages associated to the user.

## Research synthesis (what to copy)

- **ChatGPT**: two layers — *saved memories* (explicit, auditable, editable, always-injected) + *referenced chat history* (implicit). A background "dreaming" process curates memories.
- **mem0**: a two-phase pipeline — *extraction* (salient facts from the recent exchange) → *consolidation* (an LLM decides **ADD / UPDATE / DELETE / NOOP** against existing memories, deduped).
- **Letta/MemGPT**: tiered memory — in-context editable *core blocks* (human/persona) + *archival* (retrieved). Agent can self-edit memory via tools.
- **OpenAI API**: a stable `user` / `safety_identifier` string per end-user (hash PII; ≤64 chars) for abuse monitoring.

**Adopted model:** per-user memory written by a **mem0-style background extraction+consolidation** pipeline, **always-injected** like ChatGPT saved memories (the per-user set is small), with the same `User` identity aligned to OpenAI's `user`/`safety_identifier`. Agent-driven memory tools and semantic/vector retrieval and a separate *system* memory layer are deferred to v2.

## Scope

**In scope (v1):**
- **Identity:** a `User` record; `ResponsesRequest` accepts `user` and `safety_identifier` as aliases for `user_id` (`resolved_user_id`); `ensure_user` on stored turns; `ConversationItem` gains `user_id` (messages associated to the user).
- **User memory store:** a `MemoryItem` model + store methods (`add_memory`, `list_memories`, `update_memory`, `delete_memory`, `delete_user_memories`) on both `InMemoryStore` and `SqlStore`.
- **Read / inject:** `build_context` loads the user's memories (capped, recency-ordered) and `render_system_prompt` injects them as a `# Memory` section. Composes with SOUL.
- **Write / extract (background):** a `MemoryExtractor` (LLM-backed, injected completion fn) extracts salient facts from a completed turn and consolidates (ADD/UPDATE/DELETE/NOOP) into the user's memory — run **async, fault-tolerant, gated**, after a stored turn (reusing the detached-run pattern), never blocking the response.
- **Management API:** `GET /v1/users/{id}/memories`, `DELETE /v1/users/{id}/memories` (+ delete one).

**Out of scope (v2, noted):**
- **System memory layer** (global operator-editable blocks) — SOUL already covers static system identity; dynamic system blocks come later.
- **Semantic / vector retrieval** — v1 injects the (small) capped per-user set wholesale (keyword/recency); vector top-K is a gated upgrade (like `web_search`).
- **Agent-driven memory tools** (`remember`/`forget`) — Letta-style; reuses the tool registry; later.
- **Upstream safety-identifier forwarding** to the provider call — noted; later.
- Memory UI in `newfrontend`.

## Architecture

### Identity (`schemas.py`, `store`, route)
- `ResponsesRequest` gains `user: Optional[str]` and `safety_identifier: Optional[str]`; a property `resolved_user_id = user_id or user or safety_identifier`.
- `User` dataclass + `users` table (`id`, `display_name?`, `created_at`, `meta`). Store: `ensure_user(user_id, display_name=None) -> User` (idempotent), `get_user(user_id)`.
- `ConversationItem`/`Item` gains `user_id`; the route stamps `resolved_user_id` on persisted items and passes it to `ensure_conversation` (replacing the raw `request.user_id`).

### Memory store (`store/base.py` + `memory.py`/`sql.py` + `models.py`)
- `MemoryItem` dataclass + `memory_items` table: `id`, `user_id`(idx), `text`, `kind="fact"`, `source_response_id?`, `status="active"`, `created_at`, `updated_at`.
- Methods (both stores): `add_memory(item)`, `list_memories(user_id, limit=50)` (active, newest `updated_at` first), `update_memory(id, text)`, `delete_memory(id)`, `delete_user_memories(user_id)`.

### Read / inject (`soul.py`, `builder.py`)
- `render_system_prompt(soul, *, tool_names, memories: Optional[List[str]] = None, extra="")` adds a `# Memory` section ("What you remember about this user:" + bullets) when `memories` is non-empty.
- `build_context(request, store, *, soul, registry, ...)`: when `resolved_user_id` is set, `memories = [m.text for m in await store.list_memories(uid, limit=MEMORY_INJECT_LIMIT)]` and pass to `render_system_prompt`. Capped (e.g. 30) so the prompt stays bounded.

### Write / extract (`app/memory.py`)
- `MemoryExtractor(complete: Callable[[str], Awaitable[str]])` — `async def extract(self, turn_text, existing: List[MemoryItem]) -> List[MemoryOp]`: prompts the LLM with the latest exchange + existing memories, parses a JSON array of ops. `MemoryOp = {op: "ADD"|"UPDATE"|"DELETE"|"NOOP", text?: str, target_id?: str}`.
- `apply_memory_ops(store, user_id, ops, source_response_id)` — deterministic: ADD→`add_memory`, UPDATE→`update_memory(target_id)`, DELETE→`delete_memory(target_id)`, NOOP→skip. Unknown/invalid ops skipped.
- **Hook:** after a stored turn persists (route sync path AND the detached-run finally), if memory is enabled and `resolved_user_id` is set, schedule `asyncio.create_task(update_user_memory(...))`: build `complete` from the request's model client (via the provider router / LeanLLM), extract, apply. Fully guarded (logs, never raises into the turn). Gated by `Settings.memory_enabled` (default off until a model is configured) + per-request `memory: bool = True`.
- `complete(prompt)` wraps a `LeanLLM.astream` single-shot (collect deltas) — the extractor is testable with a fake `complete`.

### Management API (`routes/users.py`)
- `GET /v1/users/{id}/memories` → `{ "data": [ { id, text, kind, created_at, updated_at } ] }`.
- `DELETE /v1/users/{id}/memories` → clear all for the user; `DELETE /v1/users/{id}/memories/{memory_id}` → delete one.

### Settings
- `memory_enabled: bool = False`, `memory_inject_limit: int = 30`, `memory_model: str = ""` (which catalog model runs extraction; defaults to the request's model when empty).

## Data flow (end-to-end)

**Read (per turn):** request → resolve `user_id` (`user_id`/`user`/`safety_identifier`) → `build_context` loads capped user memories → `render_system_prompt` injects `# Memory` → agent runs with the user's memory in context.

**Write (after a stored turn, async):** persist items (stamped with `user_id`) + response → schedule background `update_user_memory`: extract salient facts from (user msg + assistant reply) given existing memories → consolidate (ADD/UPDATE/DELETE/NOOP) into the user's memory store. Next turn's read picks it up.

**Multi-instance:** memory lives in the shared store (Sql) → consistent across instances; extraction runs on the owning instance (same as the detached run). No new cross-instance concern.

## Error handling / privacy
- Extraction failures (LLM/JSON) are logged and dropped — never affect the turn. Memory disabled (`memory_enabled=False` or no user_id or `memory:false`) → no extraction, no injection.
- Users can list and delete their memories (management API); `delete_user_memories` is the "forget me" control.
- PII: the safety identifier sent upstream (future) should be hashed; v1 stores memory text as the model extracts it (operator-controlled prompt can constrain sensitivity).

## Testing
- **Identity:** `resolved_user_id` precedence; `ensure_user` idempotent; items persisted with `user_id`; conversation `user_id` from the resolved id. Both stores.
- **Memory store:** add/list(order+cap+active)/update/delete/delete_user_memories. Both stores.
- **Inject:** `render_system_prompt(memories=...)` emits the `# Memory` section (and omits it when empty); `build_context` injects the user's memories and none when no user.
- **Extractor (pure):** parses ADD/UPDATE/DELETE/NOOP from a fake `complete`; `apply_memory_ops` performs the right store mutations; malformed JSON → no ops, no raise.
- **End-to-end (TestClient):** with memory enabled + a fake extractor, a stored turn populates the user's memory and a later turn injects it into the system prompt; memory disabled → nothing written; management API lists/deletes.
- Import-lean + boot/isolation gates green.

## Sequencing
One spec, one plan, subagent-driven. v2 (system memory, vector retrieval, memory tools, UI, upstream forwarding) is a separate later effort.
