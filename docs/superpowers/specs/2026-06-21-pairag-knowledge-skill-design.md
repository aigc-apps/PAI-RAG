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
├── pairag.py           # single zero-dep Python 3 CLI (stdlib only)
└── reference/
    └── retrieval.md    # deep reference: modes, metadata filters, examples
```

`SKILL.md` stays lean (always loaded). `reference/retrieval.md` holds heavier
detail the agent reads only when needed (retrieval modes, full metadata-filter
operator set, worked examples).

### Configuration & connection

Resolution order: **flags → environment → config file → defaults**.

| Setting    | Env                 | Default                       | Notes |
|------------|---------------------|-------------------------------|-------|
| Base URL   | `PAIRAG_BASE_URL`   | `http://localhost:8682`       | service backend port |
| Tenant     | `PAIRAG_TENANT_ID`  | unset                         | sent as `X-TENANT-ID` header; omitted when unset (server uses default tenant unless `ENABLE_TENANT_ID`) |
| Default KB | `PAIRAG_KB`         | unset                         | name or id; used when `--kb` omitted (pinned-KB scenario) |
| Auth token | `PAIRAG_TOKEN`      | unset                         | optional `Authorization: Bearer <token>` (future-proof; current endpoints unauthenticated) |

Optional config file at `~/.config/pairag/config.json` accepts the same keys.

### Command surface

Five top-level verbs mapped to confirmed service endpoints (all under the running
server's base URL):

| Command | Purpose | Endpoint |
|---|---|---|
| `pairag kbs [query]` | Discover KBs — id, name, description | `GET /v1/config/knowledgebases` |
| `pairag search <query> [--kb] [--mode] [--top-k] [--rerank] [--threshold] [--filter]` | Semantic / hybrid retrieval | `POST /v1/tools/retrieval/{kb}` |
| `pairag catalog [--kb] [--query] [--product] [--section] [--lang] [--limit]` | Browse documents by metadata (no body reads) | `GET /v1/config/knowledgebases/{kb}/catalog` |
| `pairag grep <pattern> [--kb] [--context] [--path-prefix] [--limit]` | Literal keyword grep (line numbers + context) | `GET /v1/config/knowledgebases/{kb}/keyword` |
| `pairag read <id> [--kb] [--max-chars] [--offset]` | Fetch a file's full text (from a search/catalog/grep result) | `GET /v1/config/knowledgebases/{kb}/file-content` |

`read` accepts the `file_id` or `doc_id` carried by any `search`, `catalog`, or
`grep` result — that is the "fetch file from search result" path.

`files` (file listing) and `chunks` (chunk listing) are intentionally **not**
exposed: `catalog` covers document discovery and `read` covers content.

Behavior details:

- `--kb` accepts a **name or id**. The CLI resolves name→id by fetching the `kbs`
  listing once per process and caching it. If `--kb` is omitted, it falls back to
  `PAIRAG_KB`; if neither is set, the command errors with the list of available
  KBs.
- `--mode` ∈ `vector | fulltext | hybrid`, mapped into the request's
  `retrieval_setting.retrieval_mode`. `--rerank` toggles
  `retrieval_setting.enable_rerank`. `--top-k` → `top_k`. `--threshold` →
  `similarity_threshold`.
- `--filter` takes a compact `key=value` (equals) / `key~value` (contains)
  syntax that the CLI compiles into the service's `metadata_condition` JSON. The
  full operator reference lives in `reference/retrieval.md`.

### Confirmed endpoint contracts

- Retrieval (agent-tool format): `POST /v1/tools/retrieval/{knowledgebase_id}`
  with body `{query, image_list?, user_id?, retrieval_setting?, metadata_condition?}`
  → `{status, status_code, data: {total, nodes: [...]}, request_id}`.
- KB list: `GET /v1/config/knowledgebases?page&size&query&ids` (paginated KB
  entities with `kb_id`, `name`, `description`).
- Catalog search: `GET /v1/config/knowledgebases/{kb_id}/catalog?query&product&section&lang&limit`
  → `{results, total}` (metadata-level document entries; no body reads).
- File content: `GET /v1/config/knowledgebases/{kb_id}/file-content?file_id&doc_id&max_chars&offset`
  (accepts `file_id` or `doc_id`).
- Keyword grep: `GET /v1/config/knowledgebases/{kb_id}/keyword?pattern&doc_id&path_prefix&datasource&context&limit`
  → `{results, scanned_files, scan_capped, limit_reached}`.
- Tenant header: `X-TENANT-ID` (optional; omitted when not configured).

### Output format

Default = **compact markdown**, token-efficient and citable. Example for `search`:

```
3 results for "vector index config" in kb=docs (hybrid, reranked)

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
- Heavy detail deferred to `reference/retrieval.md`.

## Testing

- A `--self-test` path (or pytest module) that mocks the HTTP layer (stdlib
  `http.server` or monkeypatched `urllib` opener) — no live server required.
- Coverage: command/argument parsing, KB name→id resolution, `--filter`
  compilation into `metadata_condition`, output rendering (markdown + `--json`),
  and error paths (connection refused, 404 KB, HTTP error).

## Open questions

None blocking. Auth remains a no-op pass-through until the service adds
authentication; the `PAIRAG_TOKEN` plumbing is in place for that day.
