# Data Source Unified Document ID Design

## Goal

Use one locally generated identifier for both a data-source document and its
knowledge-base file while keeping source identity and content versioning
independent from that local identifier.

For every newly synchronized document:

```text
doc_id == file_id == "doc_" + 8 random hexadecimal characters
```

This feature has not been released, so existing data and manifests do not need
migration or compatibility handling.

## Identity Model

The synchronization model has three distinct identifiers:

| Field | Meaning | Generation |
|---|---|---|
| `source_id` | Stable identity in the upstream data source | Yuque UUID; normalized source path for adapters without a native ID |
| `content_hash` | Version of the normalized document content | SHA-256 of normalized content |
| `doc_id` / `file_id` | PAI-RAG-local document and file identity | `doc_` plus 8 random hexadecimal characters, generated once on first ingest |

`source_id` determines whether two discoveries represent the same upstream
document. `content_hash` determines whether that document changed. The local ID
is unrelated to either value.

## Persistence Model

`pai_datasource_document` gains a non-null `source_id` column and a unique
constraint on `(datasource_id, source_id)`. Its existing `doc_id` and `file_id`
columns contain the same local ID.

`pai_knowledgebase_file.id` contains that same local ID. Existing foreign-key
consumers continue treating the value as an opaque string, so file tasks,
chunks, metadata relations, and vector metadata require no separate ID mapping.

The existing `(datasource_id, doc_id)` uniqueness remains valid. The
`pai_knowledgebase_file.id` primary key provides global collision detection.
This change does not add random-ID collision retries; a collision is reported as
an ingest failure and retry support may be added separately.

## Adapter Contract

`DiscoveredDoc` carries `source_id`, because synchronization must identify a
document before fetching its body.

- Yuque uses the UUID returned by the TOC or document-list API.
- Adapters without a native stable ID use their normalized source-relative path.

`SourceDocument` carries the same `source_id` and the normalized content hash.
Adapters do not generate local `doc_id` values.

## Synchronization Flow

1. Discover documents and index them by `source_id`.
2. Load the manifest and index it by `source_id`.
3. Compute additions, possible updates, and deletions from the two source-ID
   sets.
4. Fetch new and possibly changed documents.
5. For an existing manifest row, reuse its `doc_id` and require its `file_id` to
   be the same value.
6. For a new document, generate `doc_<8 random hex>`, create
   `KbFileEntity.id` with it, and store it in both manifest ID columns.
7. Compare `content_hash` to skip unchanged content. A changed hash updates the
   existing file under the same ID.
8. Delete missing source IDs through the existing file-deletion pipeline.

Changing a Yuque slug or directory does not create a new local document when
the Yuque UUID remains unchanged; the stored path and URLs are updated instead.

## Read and Retrieval Behavior

New records expose a single identifier. Both `file_id` and `doc_id` parameters
may remain in public APIs for interface compatibility, but they carry the same
`doc_xxxxxxxx` value. Existing file lookup, chunk lookup, and attachment code
continue to treat IDs as opaque strings; no UUID-specific fallback is added.

## Error Handling

- Missing or empty `source_id` is a discovery error for that document.
- Duplicate `(datasource_id, source_id)` is rejected by the database.
- Random local-ID collision is rejected by the file primary key and recorded as
  a document ingest failure.
- A failed flush must be rolled back before recording the document failure, so
  the SQLAlchemy session is not reused in a failed transaction state.

## Testing

Targeted tests must verify:

1. Yuque discovery uses its UUID as `source_id`.
2. Generic adapters fall back to a normalized path source ID.
3. A new document receives a `doc_` ID with exactly 8 hexadecimal characters.
4. The resulting manifest has `doc_id == file_id == KbFileEntity.id`.
5. A later sync with the same source ID reuses the local ID.
6. A changed content hash updates the existing file.
7. An unchanged content hash skips ingestion.
8. A changed slug with the same Yuque UUID updates rather than duplicates the
   document.
9. Source IDs missing from discovery are deleted.
10. An ingest `IntegrityError` does not leave failure recording on a poisoned
    SQLAlchemy session.

## Out of Scope

- Migrating or backfilling existing manifests or file IDs.
- Compatibility matching by old `doc_id`, UUID file ID, or path.
- Retrying random local-ID collisions.
- Increasing the random suffix beyond 8 hexadecimal characters.
