# PAI-RAG Knowledge Base Skill — Design

**Date:** 2026-06-21
**Status:** Approved (pending implementation plan)

## Purpose

Give any CLI-capable agent (Claude Code and other mainstream agents) direct,
read-only access to a running PAI-RAG knowledge base service for search and file
retrieval. The skill packages a single zero-dependency CLI plus agent-facing
instructions, so an agent can discover knowledge bases, retrieve relevant
passages (semantic search), browse the document catalog by metadata, grep
document bodies, and fetch a file's full text from a search result — then cite
results back to the user.

The exposed surface is intentionally small — five commands — to keep the skill
focused and avoid over-exposing the service's full API.

Scope is **read-only retrieval**. No knowledge-base creation, file ingestion, or
chunk mutation. This keeps the surface safe for autonomous agents and matches how
mainstream RAG/retrieval skills are scoped.

## Goals

- Portable to "most mainstream agents": one self-contained Python 3 script using
  only the standard library — no `pip install`, no venv.
- Ergonomic for agents: discover → search → cite → inspect, with human-readable
  output by default and `--json` for parsing.
- Two targeting scenarios supported: a discovery command for multi-KB use, and an
  optional pinned default KB for single-KB use.
- Elegant and extensible: one CLI file with shared connection/auth/output
  plumbing; git-style subcommands.

## Non-Goals

- Write operations (create/delete KB, upload/parse files, edit/delete chunks).
- Replacing the existing MCP retrieval integration — this is a complementary CLI.
- File listing and chunk listing — deliberately omitted to keep the surface
  small; `catalog` covers document discovery and `read` covers content.
- Multi-server orchestration or session/state management beyond per-process
  caching of the KB list for name→id resolution.

## Architecture

Approach **A**: a single `pairag` CLI with git-style subcommands, one
self-contained script with shared config/HTTP/error-handling/output code.

### Layout

The skill ships inside the repo (versioned with the service; symlinkable into
`~/.claude/skills/` for personal use):

```
skills/pairag-knowledge/
├── SKILL.md            # agent-facing instructions + command vocabulary
└── pairag.py           # single zero-dep Python 3 CLI (stdlib only)
```

With the trimmed surface, the five commands and config fit comfortably in a lean
`SKILL.md` — no separate reference file is needed. If retrieval tuning or catalog
facets are added later, the deeper detail can move into a `reference/` doc then.

### Configuration & connection

Resolution order: **flags → environment → defaults**.

| Setting    | Env                 | Default                       | Notes |
|------------|---------------------|-------------------------------|-------|
| Base URL   | `PAIRAG_BASE_URL`   | `http://localhost:8682`       | service backend port |
| Tenant     | `PAIRAG_TENANT_ID`  | unset                         | sent as `X-TENANT-ID` header; omitted when unset (server uses default tenant unless `ENABLE_TENANT_ID`) |
| Default KB | `PAIRAG_KB`         | unset                         | name or id; used when `--kb` omitted (pinned-KB scenario) |
| Auth token | `PAIRAG_TOKEN`      | unset                         | optional `Authorization: Bearer <token>` (future-proof; current endpoints unauthenticated) |

Configuration is env-vars + flags only — no config file. Flags override env for
per-call overrides; an empty env value is treated as unset.

### Command surface

Five top-level verbs mapped to confirmed service endpoints (all under the running
server's base URL):

| Command | Purpose | Endpoint |
|---|---|---|
| `pairag kbs [query]` | Discover KBs — id, name, description | `GET /v1/config/knowledgebases` |
| `pairag search <query> [--kb]` | Semantic / hybrid retrieval | `POST /v1/retrieval` |
| `pairag catalog [--kb] [--query] [--limit]` | List KB files — name, title, source (no body reads) | `GET /v1/config/knowledgebases/{kb}/files` |
| `pairag grep <pattern> [--kb] [--context] [--limit]` | Literal keyword grep over the whole KB (line numbers + context) | `GET /v1/config/knowledgebases/{kb}/keyword` |
| `pairag read <id> [--kb] [--max-chars] [--offset]` | Fetch a file's full text (from a search/catalog/grep result) | `GET /v1/config/knowledgebases/{kb}/file-content` |

`read` accepts the `file_id` or `doc_id` carried by any `search`, `catalog`, or
`grep` result — that is the "fetch file from result" path. `catalog` always
yields a `file_id`; `search`/`grep` yield `doc_id` (falling back to `file_id`).

`chunks` (chunk listing) is intentionally **not** exposed: `catalog` covers file
discovery and `read` covers content.

**Coverage.** The commands cover manually-uploaded files, not just data-source
documents — but `catalog` and the search commands differ on *indexed* vs *all*:
- `catalog` lists `KbFileEntity` rows via the files endpoint — **every file in
  the KB regardless of parse status** — returning `file_name`,
  `file_metadata.title`, and `file_source`.
- `grep` hits the keyword endpoint, whose unfiltered case now searches the whole
  KB. `keyword_search`'s default branch no longer restricts to data-source files.
  Its candidate set comes from `KbChunkEntity`, so **only indexed (chunked) files
  are searched** — files still parsing, failed, or chunk-less are not. The
  chunk-text prefilter uses `contains(pattern, autoescape=True)` so literal
  `%` / `_` are not treated as SQL wildcards. (The `/keyword` HTTP endpoint still
  accepts optional `doc_id` / `path_prefix` / `datasource` filters; the agent
  `grep` tool no longer exposes them — see the agent-tools note below.)
- `search` (`/v1/retrieval`) queries the vector store, so it likewise covers
  **indexed content only**.

Because `search`/`grep` see only indexed content, an empty result does not prove
a file is absent; `catalog` is the source of truth for "does this file exist."
SKILL.md states this so an agent does not misread an empty search as "absent."

**Related backend change — agent KB file tools.** The same KB-wide capabilities
are mirrored to the in-process agent. The old data-source-gated `datasource_tool`
module (search/catalog/keyword/fetch, exposed only for KBs with data sources) was
replaced by `tools/knowledgebase/knowledgebase_file_tools.py` providing three
whole-KB tools — `catalog` (list files via `list_files`), `grep` (KB-wide
`keyword_search`), and `fetch` (read a file by id) — registered for **every** KB
in `agent_service.py`. The redundant semantic `search` tool was dropped; the
existing `aget_knowledgebase_tool` remains the canonical semantic retriever. This
keeps the agent surface coherent with the CLI's `catalog`/`grep`/`read`.

Behavior details:

- `--kb` accepts a **name or id**. The CLI resolves name→id by fetching the `kbs`
  listing once per process and caching it. If `--kb` is omitted, it falls back to
  `PAIRAG_KB`; if neither is set, the command errors with the list of available
  KBs.
- `search` posts `{query, knowledge_id}` to `/v1/retrieval` (the records-format
  endpoint), not the MCP tool endpoint. The records endpoint preserves each
  hit's full `metadata` dict — including `doc_id`, `file_path`, `file_name` —
  which is what `read` needs to fetch the source file. The MCP tool endpoint
  (`/v1/tools/retrieval`) drops those ids, so it cannot feed `read`. `search`
  sends no `retrieval_setting`, so the service applies the KB's own configured
  retrieval defaults (mode, rerank, top_k, threshold). Tuning flags and metadata
  filters are deferred — see Future extensions.
- `catalog` lists KB files via the files endpoint with an optional free-text
  `--query` (matches file name + title) and `--limit` (→ `size`). It renders
  title, file name, `file_id`, and source per file.

### Confirmed endpoint contracts

**Response envelopes.** `/v1/retrieval` returns a **flat** body `{records: [...]}`.
All `/v1/config/...` endpoints wrap their payload as `{code, message, data}` —
the CLI reads `data`. The HTTP helper handles both.

- Retrieval (records format): `POST /v1/retrieval` with body
  `{query, knowledge_id, retrieval_setting?, metadata_condition?}` → flat
  `{records: [{content, score, title, url, metadata}]}`. Each record's
  `metadata` carries `doc_id`, `file_path`, `file_name`, `file_source`.
- KB list: `GET /v1/config/knowledgebases?page&size&query` →
  `data: {items: [{id, name, description, ...}], total, pages, page, size}`.
- File list (catalog): `GET /v1/config/knowledgebases/{kb_id}/files?query&size` →
  `data: {items: [{id, file_name, file_source, file_metadata: {title, source_url, ...}, ...}], total, pages, page, size}`.
- Keyword grep: `GET /v1/config/knowledgebases/{kb_id}/keyword?pattern&context&limit` →
  `data: {results: [{doc_id, file_id, line, match, context, source_url, title}], scanned_files, scan_capped, limit_reached}`.
  The unfiltered call covers the whole KB (indexed files).
- File content: `GET /v1/config/knowledgebases/{kb_id}/file-content?file_id&doc_id&max_chars&offset`
  (accepts `file_id` **or** `doc_id`) → `data: {file_id, file_name, title,
  source_url, doc_id, content, content_length, offset, returned_chars,
  truncated, next_offset, degraded, metadata}`.
- Tenant header: `X-TENANT-ID` (optional; omitted when not configured).

### Output format

Default = **compact markdown**, token-efficient and citable. Example for `search`:

```
3 results for "vector index config" in kb=docs

1. [0.87] Configuring the vector store · file_id=a1b2c3 · setup/vectordb.md
   …set `vector_store.type` to `elasticsearch` and provide the endpoint…

2. [0.81] …
```

- Each result carries a citable id (`file_id` / `doc_id`) so the agent can chain
  into `read`.
- `--json` emits the raw service payload for programmatic parsing.
- Empty results are explicit (`No results for …`), never silent.

### Error handling

- Friendly, actionable failures:
  - connection refused → "Is the PAI-RAG server running on `{base_url}`?"
  - 404 / unknown KB → print the list of available KBs.
  - HTTP error status → surface status code and server message.
- Non-zero process exit codes on failure so agents can detect errors
  programmatically.

### SKILL.md structure

- Frontmatter: `name` + trigger-rich `description` (knowledge base, retrieval,
  search docs, RAG, PAI-RAG, "look it up in the knowledge base").
- Body (one screen): decision guide (`search` = semantic, `catalog` = browse by
  metadata, `grep` = exact strings, `kbs` = discover, `read` = fetch full text),
  the five commands with one example each, the config note, and the "cite the
  `file_id`" convention.

## Testing

- A `--self-test` path (or pytest module) that mocks the HTTP layer (stdlib
  `http.server` or monkeypatched `urllib` opener) — no live server required.
- Coverage: command/argument parsing, KB name→id resolution, request building
  per command, output rendering (markdown + `--json`), and error paths
  (connection refused, 404 KB, HTTP error).

**Deferred — backend `keyword_search` service test.** Making `grep` KB-wide
changed `keyword_search`'s default branch (drop the data-source filter; autoescape
the literal pattern). This is not covered by a backend test because the repo has
no real-DB test harness: `tests/service` mocks the `AsyncSession` (so the SQL
`WHERE` is never executed), `tests/db` mocks the engine, and `tests/integration`
is end-to-end but skipped without `DASHSCOPE_API_KEY` + Redis + Chroma. A
meaningful test needs net-new infrastructure (an in-memory sqlite session, seeded
`KbChunkEntity` / `KbFileEntity` / `DataSourceDocumentEntity` rows, and a mocked
`file_store`) to assert whole-KB coverage, tenant isolation, the `MAX_SCAN_FILES`
cap, and `%` / `_` literal patterns. Tracked as follow-up; the change is
behavior-additive for the CLI and covered at the CLI layer (request building).

## Future extensions

Deferred for now; addable without breaking the interface:

- `search` retrieval tuning flags: `--mode` (vector/fulltext/hybrid), `--top-k`,
  `--rerank`, `--threshold`, and a `--filter` syntax compiled into the service's
  `metadata_condition`.
- `catalog` facet flags: `--product`, `--section`, `--lang`.

## Open questions

None blocking. Auth remains a no-op pass-through until the service adds
authentication; the `PAIRAG_TOKEN` plumbing is in place for that day.
