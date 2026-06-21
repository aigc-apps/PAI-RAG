---
name: pairag-knowledge
description: Search a PAI-RAG knowledge base and fetch source files from the command line. Use when the user asks to look something up in the knowledge base, search docs/RAG, find where something is documented, retrieve a passage with a citation, or read a knowledge-base file. Triggers — knowledge base, retrieval, RAG, "search the docs", "look it up in the KB", PAI-RAG.
---

# PAI-RAG Knowledge Base

Read-only command-line access to a running PAI-RAG service. Use it to find and
cite knowledge-base content. All commands print compact markdown; add `--json`
for raw output.

Run via: `python <skill-dir>/pairag.py <command> [...]` (Python 3, no dependencies).

## Which command

- **`search <query>`** — semantic / hybrid retrieval. Use for meaning-based
  questions ("how do I configure the vector store?"). Returns ranked passages,
  each with a `doc_id`. **Covers the whole KB**, including manually-uploaded files.
- **`catalog`** — browse the document catalog by metadata (no body reads). Use to
  see what documents exist: `--query <text>` to filter, `--limit N`.
  **Data-source documents only** (see Scope below).
- **`grep <pattern>`** — literal keyword search over document bodies, with line
  numbers and context. Use for exact strings (an error message, a config key).
  **Data-source documents only** (see Scope below).
- **`read <id>`** — fetch a file's full text. Pass the `doc_id` (or `file_id`)
  from any `search` / `catalog` / `grep` result. Supports `--max-chars` and
  `--offset` for paging large files.
- **`kbs [query]`** — list available knowledge bases (id, name, description).

## Scope: catalog and grep see data-source documents only

`catalog` and `grep` operate over documents ingested through a **data source**
(llms.txt, Sphinx, GitHub, and similar). Files added by **manual upload** are not
listed by `catalog` and not scanned by `grep`. So an empty `catalog`/`grep`
result does **not** mean a document is absent from the knowledge base — it may
have been uploaded directly. When you need full-KB coverage (including manual
uploads), use `search`, which retrieves over all indexed content.

## Targeting a knowledge base

Every command except `kbs` needs a KB. Pass `--kb <name-or-id>`, or set a default
once with `PAIRAG_KB`. Names are resolved to ids automatically; a 32-char hex
value is treated as an id directly. If you don't know the KB, run `kbs` first.

## Configuration

Resolution order: flags → environment → `~/.config/pairag/config.json` → defaults.

| Setting    | Flag         | Env                | Default                 |
|------------|--------------|--------------------|-------------------------|
| Base URL   | `--base-url` | `PAIRAG_BASE_URL`  | `http://localhost:8682` |
| Tenant     | `--tenant`   | `PAIRAG_TENANT_ID` | (unset)                 |
| Default KB | `--kb`       | `PAIRAG_KB`        | (unset)                 |
| Auth token | `--token`    | `PAIRAG_TOKEN`     | (unset)                 |

## Examples

```bash
# Discover knowledge bases
python pairag.py kbs

# Semantic search in the "docs" KB
python pairag.py search "how to configure the vector store" --kb docs

# Browse the catalog, then read a document by its doc_id
python pairag.py catalog --query install --kb docs
python pairag.py read d1f2... --kb docs

# Exact-string search with more context, as JSON
python pairag.py grep "timeout = 600" --kb docs --context 3 --json
```

## Citing results

Each result carries a `doc_id` (and `read` also shows `file_id`). When you relay
an answer to the user, cite the source document/file the passage came from, and
use `read` to pull the full text when you need more than the snippet.

## Notes

- The PAI-RAG server must be running. If a command reports it can't reach the
  server, confirm the service is up and `PAIRAG_BASE_URL` points at it.
- This skill is read-only: it never creates, edits, uploads, or deletes anything.
