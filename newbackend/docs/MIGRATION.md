# Migration: `backend/` → `newbackend/`

`newbackend/` is the **base** going forward. The legacy `../backend/` keeps running (the heavy
RAG/KB app at `app.main`, the worker, llama-index/chroma stacks); we migrate capabilities into
`newbackend/` incrementally, one bounded slice at a time, keeping `newbackend/` green and
heavy-dep-free at every step.

## What's already here

`newbackend/` is the exact runtime import closure of the lean service (`app.lean_main`) extracted
from `backend/`, plus the tools extension point (`agent/tools/mcp.py`) and the lean test suite —
56 module files, 35 test files, all passing under `uv`. It does **not** include the legacy heavy
app (`backend/app/main.py`), the worker, RAG/vector/embedding code, i18n/middleware, or anything
that pulls `llama_index`/`torch`/`transformers`/`chromadb`.

## Principles for migrating new slices

1. **Keep the import-isolation gate green.** `tests/test_lean_import_isolation.py` blocks the heavy
   stack at import. Any module migrated in must import cleanly without it (lazy/guard, or drop the
   dependency).
2. **Bring the runtime closure, not whole directories.** Copy only what a feature's closure needs;
   leave legacy siblings (`main.py`, `worker.py`, middlewares) behind.
3. **Port tests with the slice.** Each migrated capability arrives with its tests; the flat
   `tests/` layout + `pythonpath=["."]` (in `pyproject.toml`) keeps imports working.
4. **Manage deps via uv.** `uv add` only the genuinely-needed packages; never re-introduce the
   heavy ML stack into the lean base — if a feature needs it (e.g. real RAG retrieval), it belongs
   behind an **optional dependency group** + a guarded import, not the core deps.
5. **Spec → plan → subagent-driven** (the `docs/superpowers/` flow) for each slice, as with the
   features already built here.

## Candidate slices to migrate next (not yet here)

- **Real RAG / knowledge-base retrieval** — the original product's strength; bring it back as an
  *optional* retrieval layer that feeds the context's project/system layer (the v2 context design
  left a slot for this). Needs a vector backend behind an optional dep group.
- **Embeddings / search providers** — wire a real `web_search` `SearchProvider` and/or an
  embedding-based memory retrieval (currently user memory is injected capped, no relevance search).
- **Auth / tenancy** — the lean service has no auth; `user_id` is an unauthenticated tag. A real
  auth layer (server-side keys, authenticated `user_id`, per-user memory ownership checks) is the
  prerequisite for multi-tenant use.
- **Live MCP transport** — `agent/tools/mcp.py` has the schema→Tool adapter; a real stdio/SSE MCP
  client + lifecycle is the integration piece.
- **Multi-instance runtime state** — `RunManager` is single-process/in-memory; resume/cancel across
  instances needs a shared bus (e.g. Redis Streams; the `sequence_number` cursor was built for it).
- **Cleanup of guarded legacy hooks** — remove the vestigial `extensions.trace` tracing wrapper and
  the llama-index `tools/adapter.py` coupling once their replacements (or removal) are decided.

## Cutover

When `newbackend/` covers everything in use, switch deployment to it and retire `backend/`. Until
then both coexist; `newbackend/` is authoritative for the agent core.
