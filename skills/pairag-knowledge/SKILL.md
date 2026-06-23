---
name: pairag-knowledge
description: Search a PAI-RAG knowledge base and fetch source files from the command line. Use when the user asks to look something up in the knowledge base, search docs/RAG, find where something is documented, retrieve a passage with a citation, or read a knowledge-base file. Triggers — knowledge base, retrieval, RAG, "search the docs", "look it up in the KB", PAI-RAG.
---

# PAI-RAG Knowledge Base

Read-only command-line access to a running PAI-RAG service. Use it to find and
cite knowledge-base content. All commands print compact markdown; add `--json`
for raw output.

Run via: `python <skill-dir>/pairag.py <command> [...]` (Python 3, no dependencies).

**Shorter commands.** Each command also has a wrapper next to `pairag.py`, so once
the skill dir is on your `PATH` you can drop the `python pairag.py` prefix:

| Wrapper | Equivalent to |
|---|---|
| `kb_list` | `pairag.py kbs` |
| `kb_search` | `pairag.py search` |
| `kb_catalog` | `pairag.py catalog` |
| `kb_read` | `pairag.py read` |
| `kb_grep` | `pairag.py grep` |

```bash
export PATH="<skill-dir>:$PATH"     # once per shell (or add to your shell rc)
kb_search "how to configure the vector store" --kb docs
```

## Which command

- **`search <query>`** — semantic / hybrid retrieval. Use for meaning-based
  questions ("how do I configure the vector store?"). Returns ranked passages,
  each with a `doc_id`.
- **`catalog`** — list files in the knowledge base: file name, title, and source
  (no body reads). Use to see what's in the KB. `--query <text>` filters by name
  and title; `--limit N` caps the count.
- **`grep <pattern>`** — literal keyword search over file bodies, with line
  numbers and context. Use for exact strings (an error message, a config key).
- **`read <id>`** — fetch a file's full text. Pass the `file_id` (or `doc_id`)
  from any `search` / `catalog` / `grep` result. Supports `--max-chars` and
  `--offset` for paging large files.
- **`kbs [query]`** — list available knowledge bases (id, name, description).

Coverage: `catalog` lists **every file** in the KB (any status). `search` and
`grep` cover all **indexed** content — a file that is still parsing, failed to
parse, or otherwise has no chunks yet will not appear in their results. So an
empty `search`/`grep` result does not by itself mean a file is absent; check
`catalog` to confirm whether the file exists.

## Targeting a knowledge base

Every command except `kbs` needs a KB. Pass `--kb <name-or-id>`, or set a default
once with `PAIRAG_KB`. Names are resolved to ids automatically; a 32-char hex
value is treated as an id directly. If you don't know the KB, run `kbs` first.

## Configuration

Resolution order: flags → environment → defaults. Set the env vars once (shell rc,
or the container's `environment:`) and you can drop the flags entirely.

| Setting    | Flag         | Env                | Default                 |
|------------|--------------|--------------------|-------------------------|
| Base URL   | `--base-url` | `PAIRAG_BASE_URL`  | `http://localhost:8682` |
| Tenant     | `--tenant`   | `PAIRAG_TENANT_ID` | (unset)                 |
| Default KB | `--kb`       | `PAIRAG_KB`        | (unset)                 |
| Auth token | `--token`    | `PAIRAG_TOKEN`     | (unset)                 |

## Examples

```bash
# Discover knowledge bases
kb_list        

# Semantic search in the "docs" KB
kb_search "how to configure the vector store" --kb docs

# Browse the catalog, then read a document by its id
kb_catalog --query install --kb docs

# Read doc from "docs" KB 
kb_read d1f2... --kb docs

# Exact-string search with more context, as JSON
kb_grep "timeout = 600" --kb docs --context 3 --json
```

## Citing results

Each result carries a `doc_id` (and `read` also shows `file_id`). When you relay
an answer to the user, cite the source document/file the passage came from, and
use `read` to pull the full text when you need more than the snippet.

## Notes

- The PAI-RAG server must be running. If a command reports it can't reach the
  server, confirm the service is up and `PAIRAG_BASE_URL` points at it.
- This skill is read-only: it never creates, edits, uploads, or deletes anything.
