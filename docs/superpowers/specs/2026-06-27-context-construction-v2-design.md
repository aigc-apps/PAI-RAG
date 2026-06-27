# Context Construction v2 — Rolling Summary + Cacheable Layered Prompt — Design

**Date:** 2026-06-27
**Status:** Approved by delegation (scope chosen in conversation: do ① and ③; ② deferred)
**Branch:** `personal/yfei/agent-core`
**Builds on:** `agent/agent.py` (`build_messages`, `AgentMessageManager`), `agent/soul.py` (`render_system_prompt`), `app/builder.py` (`build_context`), the store, and user-memory v1.

## Problem (from the OpenClaw/Hermes comparison)

Two weaknesses to fix this iteration:

1. **No rolling/persisted conversation summary (long-conversation memory is lossy).** Today `AgentMessageManager.fit_to_budget` summarizes/drops old history **in-place, per call, and throws the result away** — it's recomputed from the full item log every turn and the dropped content is bridged by nothing. A long conversation loses its middle. (OpenClaw flushes a summary to disk before discarding and rehydrates; mem0 keeps a rolling summary.)
2. **The system prompt is not stable/cacheable and trends toward a dumping ground.** `render_system_prompt` jams persona + ALL user memories + tool protocol + safety + per-request instructions into one system message that **changes every turn** (memory/instructions vary) → provider prefix caching can't hit, and the prompt bloats. (Hermes: keep a small **stable** layer; put volatile content elsewhere; load the right context at the right time.)

**Out of scope this iteration (chosen):** relevance/vector **retrieval** of user memory (②). User memory stays **injected within a sane cap** (recent, capped); query-relevant retrieval is a future TODO.

## Decisions

1. **Split the prompt into a stable (cacheable) layer and a volatile context block.**
   - **Stable system message (front, cacheable):** Identity, Personality, Operating principles, **Project context**, Tools (protocol + names), Safety. Depends only on the soul + tool set + static project context → identical across a conversation → a cacheable prefix.
   - **Volatile context block:** user memory, the rolling conversation summary, and per-request `instructions`. Rendered as a **trailing `system` message placed AFTER history, right before the current user turn** — so the cacheable prefix (`stable system` + `history`) is maximized and only the tail changes per turn.
   - `[System Time:]` stays in the current user turn (already the case).
2. **Project-context layer (static, no retrieval).** An operator-configured `project_context` string (from `Settings.project_context` / a `project.md` file) injected into the stable system layer. Addresses "no workspace/domain context" without retrieval (consistent with ② deferral).
3. **Rolling, persisted conversation summary.** `Conversation` gains `summary` + `summarized_seq`. After a stored turn, an **async, gated, fault-tolerant** summarizer folds the oldest *unsummarized* items (beyond a kept-recent window) into the rolling summary and persists it, advancing `summarized_seq`. `build_context` then serves `[rolling summary in the volatile block] + [items with seq > summarized_seq]` instead of the full log — so the model always sees a compact bridge over old turns plus recent raw turns.
4. **`AgentMessageManager.fit_to_budget` stays as the in-call safety net** (L1–L4) for turns that still overflow after summary-trimming; it is unchanged. The rolling summary is the *primary*, persisted compaction; fit is the *fallback*.
5. **User memory unchanged except the cap is explicit.** Keep injecting recent memories (cap `MEMORY_INJECT_LIMIT=30`); they move from the system prompt into the volatile block. Retrieval = TODO.

## Architecture

### Prompt layering (`agent/soul.py`, `agent/context.py`, `agent/agent.py`)
- `render_stable_system_prompt(soul, *, tool_names, project_context="") -> str` — Identity / Personality / Operating principles / **Project** (if set) / Tools / Safety. **No** memory, instructions, or summary.
- `render_context_block(*, memories: Optional[List[str]] = None, summary: str = "", instructions: str = "") -> str` — a volatile block with `# Memory`, `# Conversation summary`, `# Additional instructions`; returns `""` when all empty.
- `AgentContext` gains `context_block: str = ""`.
- `Agent.build_messages`:
  ```
  [system: ctx.system_prompt]                 # stable, cacheable
  *ctx.history                                 # cacheable (append-only across turns)
  [system: ctx.context_block]  (if non-empty)  # volatile, late
  [user: render_current_turn(...)]             # [System Time] + text + attachments + hints
  ```
- `build_context`: `system_prompt = render_stable_system_prompt(effective_soul, tool_names, project_context)`; `context_block = render_context_block(memories=<capped user memories>, summary=<conv.summary>, instructions=request.instructions)`.

### Rolling summary (`app/store`, `app/summarizer.py`, `build_context`, route hook)
- `Conversation` (+ row) gains `summary: Optional[str] = None`, `summarized_seq: int = -1`. Store: `update_conversation_summary(conversation_id, summary, summarized_seq) -> None`.
- `ConversationSummarizer(complete)` (mirrors `MemoryExtractor`): `async def summarize(self, prior_summary, items_to_fold) -> str` — folds the prior rolling summary + a batch of older items into a new concise summary (LLM via injected `complete`; guarded; returns `prior_summary` on failure so nothing is lost).
- `maybe_summarize_conversation(store, conversation_id, complete)` — load items; if the count of items with `seq > summarized_seq` exceeds `SUMMARY_KEEP_RECENT + SUMMARY_BATCH`, fold items in `(summarized_seq, new_seq]` (all but the last `SUMMARY_KEEP_RECENT`) into the summary and persist `(summary, new_seq)`. Async, gated, fully guarded.
- `build_context`: load `conv = get_conversation(conversation_id)`. If `conv.summary`: `history_items = [it for it in resolved if it.seq > conv.summarized_seq]` and pass `summary=conv.summary` to the context block. Else: full history as today.
- Route: after a stored turn (sync + background paths), `asyncio.create_task(maybe_summarize_conversation(...))` when `Settings.summary_enabled` — reuses the memory-update scheduling pattern; fire-and-forget; never blocks/breaks the turn.

### Settings
- `project_context: str = ""` (or a `project_context_path` read once at boot).
- `summary_enabled: bool = False`, `summary_keep_recent: int = 20`, `summary_batch: int = 20`.

## Data flow

**Read:** stable system (cacheable) → history *after* `summarized_seq` → volatile block (memory + rolling summary + instructions) → current user turn. Prefix `stable+history` is cache-friendly; only the volatile tail + new turn change.

**Write (async, post-turn):** persist the turn → schedule `maybe_summarize_conversation`: if unsummarized items exceed the window, fold the overflow into `conv.summary`, advance `summarized_seq`, persist. Next read serves the compact summary + recent raw turns.

**Fallback:** if a turn still overflows the model budget, `fit_to_budget` (L1–L4) compresses in-call as today.

## Error handling / compat
- Summary disabled (`summary_enabled=False`) or summarizer failure → no summary; `build_context` serves full history (today's behavior). Summarizer returns the prior summary on failure (never loses the bridge). Fire-and-forget; can't affect the turn.
- Existing single-system-prompt behavior is replaced by stable+volatile; tests asserting memory/instructions in `system_prompt` move to asserting them in `context_block`. Echo/route tests (assert on output) are unaffected. Multiple system messages are accepted by OpenAI-compatible endpoints; if a provider rejects a trailing system message, fold the volatile block into the user-turn prefix (noted risk).
- Multi-instance: summary lives in the shared store → consistent; summarization runs on the owning instance.

## Testing
- **Prompt split:** `render_stable_system_prompt` has persona/tools/safety/project but NOT memory/instructions/summary; `render_context_block` renders memory/summary/instructions and is empty when all blank; `build_messages` emits `[stable] + history + [volatile?] + user` in order, omitting the volatile message when empty; `build_context` routes memory/instructions to `context_block` and persona/project/tools to `system_prompt`. (Update the existing soul/builder/memory-inject tests accordingly.)
- **Project context:** appears in the stable system prompt when configured; absent otherwise.
- **Summary store:** `update_conversation_summary` sets summary + summarized_seq (both stores).
- **Summarizer (pure):** folds prior summary + items via a fake `complete`; returns prior summary on failure (guarded).
- **maybe_summarize:** below threshold → no-op; above → folds overflow, advances `summarized_seq`, keeps the recent window; persisted.
- **build_context with summary:** serves only items after `summarized_seq` and injects the summary into the context block.
- **End-to-end (TestClient):** with summary enabled + a fake summarizer LLM, a long conversation gets a persisted summary and later turns carry it while old raw turns drop out of the injected history; disabled → unchanged.
- Import-lean + boot/isolation gates green.

## Risks
- **Cache wins are structural, not explicit.** We order stable-first so automatic OpenAI prefix caching can hit; explicit Anthropic `cache_control` breakpoints are a later refinement (the OpenAI-compat surface may not expose them).
- **Trailing system message** placement assumes provider acceptance; fallback = user-turn prefix.
- **Summary quality / drift** — an LLM rolling summary can lose detail; mitigated by keeping a recent raw window and persisting (not re-deriving). Relevance retrieval (②) would complement it later.
- **Two async post-turn pipelines now** (memory + summary) — both fire-and-forget and guarded; share the same scheduling concern (untracked `create_task`) tracked for cleanup.

## Sequencing
One spec, one plan, subagent-driven. ② (relevance retrieval) and explicit cache_control remain future TODOs.
