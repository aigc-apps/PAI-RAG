# Production Offline Sync Pipeline Design

Date: 2026-07-13

## Objective

Redesign knowledge-data-source synchronization as a production-oriented,
bounded asynchronous pipeline for PostgreSQL, DashScope embeddings, and
Elasticsearch. A 2,144-document PAI documentation sync must complete within ten
minutes under healthy external services while exposing durable, useful progress.

SQLite, local hash embeddings, and local retrieval must remain functionally
correct, but they are not throughput targets.

## Current Problems

The current implementation discovers every document, starts one task per fetch,
waits for every fetch result, and only then imports documents sequentially. Each
document independently embeds its chunks, commits SQL, refreshes aggregate KB
counts, deletes its old Elasticsearch records, performs a bulk request, and
refreshes Elasticsearch.

This causes five concrete problems:

1. No persisted progress is visible until the complete run finishes.
2. All fetched document bodies remain in memory at once.
3. Fetching, embedding, SQL writes, and Elasticsearch indexing cannot overlap.
4. DashScope requests are batched only within one document and are sent
   sequentially with a new HTTP client per document.
5. Elasticsearch performs delete, bulk, and refresh work once per document.

The existing queue is not the throughput bottleneck. A production run is
claimed promptly by a worker; the bottleneck is the work performed inside the
sync handler.

## Architecture

Use a bounded producer-consumer pipeline:

```text
discover
   -> fetch workers
   -> parse/chunk workers
   -> embedding batch scheduler
   -> PostgreSQL batch writer
   -> Elasticsearch batch indexer
   -> durable progress/finalization
```

Each boundary uses a capacity-limited `asyncio.Queue`. Backpressure prevents
unbounded body, chunk, or vector accumulation. The stages overlap so the next
documents can be fetched while previous documents are embedded and persisted.

The production defaults are:

- Fetch concurrency: 16
- Maximum fetched-body queue: 100 documents
- DashScope request size: at most 10 texts, matching the API limit
- DashScope request concurrency: 6 across the process
- PostgreSQL commit size: 25 documents, additionally capped by 2,000 chunks
- Elasticsearch bulk target: 5 MiB, with a hard cap of 10 MiB
- Elasticsearch refresh: disabled for intermediate bulks; one refresh at run end
- Progress checkpoint interval: every committed batch or every five seconds,
  whichever occurs first

Limits are settings with safe minimum and maximum values. DashScope and
Elasticsearch concurrency limiters are shared across sync jobs in a process so
multiple data sources cannot each consume the full configured limit.

## Component Boundaries

### Sync coordinator

The coordinator owns one run, starts and closes stages, propagates cancellation,
and determines the final result. It does not perform network or persistence work
itself.

### Fetch stage

The fetch stage consumes discovered descriptors and emits fetched source
documents. Individual fetches retry transient failures with exponential backoff
and jitter. Permanent per-document failures are recorded and do not stop the
pipeline. Authentication failures and other run-wide configuration errors open
a circuit breaker and fail the run immediately.

### Preparation and embedding stage

Preparation normalizes and chunks documents without database I/O. The embedding
scheduler aggregates chunk inputs across document boundaries. It uses a shared,
long-lived `httpx.AsyncClient`, sends at most ten texts per DashScope request,
and permits up to the configured number of requests concurrently.

Results retain document and chunk positions so vectors are deterministically
reassociated with their source chunks. A failed embedding request retries as a
unit. If a batch repeatedly fails for a content-dependent reason, it is split to
isolate the failing input without discarding successful documents.

### PostgreSQL batch writer

One transaction is the minimum durable content boundary. It:

1. Upserts documents and exact document content.
2. Replaces the documents' SQL chunks.
3. Creates completed ingestion records for successful documents.
4. Marks each document's search-index state as `pending`.
5. Updates durable run progress.

If the transaction fails, the complete batch rolls back and progress does not
advance. KB document and chunk totals are refreshed once per SQL batch, not once
per document.

### Elasticsearch batch indexer

For one batch, the indexer removes old records for all affected document IDs in
one operation and sends chunk index operations in size-bounded bulk requests.
Intermediate operations use `refresh=false`. After successful bulk indexing, a
small PostgreSQL transaction marks the documents `indexed` and advances indexed
progress.

The run performs one final index refresh before it can be reported as succeeded.
SQL remains the source of truth, but a document is not counted as searchable
until Elasticsearch acknowledgement has been persisted.

## Durable State

### Background job execution

Extend `background_jobs` with:

- `progress` JSON, default `{}`
- `heartbeat_at` timezone-aware timestamp
- `lease_expires_at` timezone-aware timestamp
- `cancel_requested_at` timezone-aware timestamp, nullable

Execution states are:

- `queued`: available to claim
- `running`: owned by a worker with a valid lease
- `retry_wait`: run-level transient failure awaiting `run_after`
- `succeeded`: all required stages and final refresh completed
- `partial`: at least one document is searchable and at least one permanently failed
- `failed`: run-wide failure or no document became searchable
- `cancelled`: explicit cancellation completed

`waiting` remains reserved for human-in-the-loop jobs outside this pipeline.

### Data-source aggregate state

Extend `knowledge_data_sources` with `active_job_id`. The field is set when a
sync is enqueued and cleared only by terminal finalization. Existing aggregate
fields retain the latest terminal result and report. The active job supplies
live phase and counters.

A PostgreSQL compare-and-set operation acquires the data source:

```sql
UPDATE knowledge_data_sources
SET active_job_id = :job_id
WHERE id = :datasource_id AND active_job_id IS NULL
```

Failure to update produces HTTP 409. This replaces the current read-then-enqueue
race.

### Document search-index state

Extend `knowledge_documents` with:

- `search_index_status`: `pending`, `indexed`, `failed`, or `delete_pending`
- `search_index_error`: last bounded error text, nullable
- `search_index_attempts`: integer

Content status and search-index status remain separate. SQL content may be
durable while Elasticsearch work is pending.

Chunk IDs become deterministic from the active index version, document ID,
chunk index, and text hash. Repeated indexing overwrites the same Elasticsearch
IDs rather than creating duplicates.

## Progress Contract

The job's `progress` JSON uses this stable shape:

```json
{
  "phase": "embedding",
  "total": 2144,
  "discovered": 2144,
  "fetched": 680,
  "embedded": 610,
  "persisted": 590,
  "indexed": 560,
  "unchanged": 12,
  "deleted": 0,
  "failed": 3,
  "bytes_fetched": 12345678,
  "docs_per_second": 5.8,
  "estimated_seconds_remaining": 267
}
```

Counters are monotonic for one job attempt and are reconstructed from durable
database state after recovery. Progress writes are throttled to batch completion
or five-second intervals so polling does not become a write bottleneck.

## Error Semantics

### Document-local errors

Fetch, parsing, content-specific embedding, or indexing errors are retried within
their stage. After the retry limit, the document is recorded as failed and the
pipeline continues. The final run is `partial` if at least one other document is
searchable.

### Run-wide errors

Invalid credentials, invalid source configuration, an untrustworthy discovery
result, or a sustained required-service outage stops new work. The coordinator
drains or cancels in-flight work safely, checkpoints committed state, and applies
job-level retry policy where the error is transient.

HTTP 401/403 responses are non-retryable until configuration changes. HTTP 429,
timeouts, connection resets, PostgreSQL serialization/deadlock errors, and
Elasticsearch 429/5xx responses are retryable with bounded exponential backoff
and jitter.

### PostgreSQL failures

A SQL batch is atomic. Failed transactions roll back in full and are retried as
the same logical batch. No persisted or indexed counter is incremented until its
corresponding transaction commits.

### Elasticsearch failures

After SQL commit, Elasticsearch failures leave documents `pending`. Retries read
chunks back from PostgreSQL; fetch and embedding are not repeated. Exhausted
items become `failed` with a bounded error message, and the run becomes partial
or failed according to whether any documents are searchable.

If Elasticsearch succeeds but the process crashes before SQL acknowledgement,
recovery repeats idempotent delete-and-bulk operations using deterministic IDs.

### Deletion safety

Deletion reconciliation occurs only after discovery produced a complete,
authoritative manifest. A partial or failed discovery never deletes existing
documents. Missing documents are first soft-deleted in SQL and marked
`delete_pending`; batched Elasticsearch deletion then completes the operation.

Fetch failure for a discovered document does not classify it as missing because
the descriptor remains in the discovered URI set.

## Lease, Restart, and Recovery

Workers update `heartbeat_at` and extend a 60-second lease every ten seconds. A
worker may claim a queued job or atomically reclaim a running job whose lease has
expired. Startup must not blindly requeue every running job because another
process may still own a valid lease.

On graceful shutdown, workers stop accepting new batches, allow the current SQL
transaction a bounded completion period, checkpoint progress, and release or
let the lease expire. A forced shutdown relies on transaction rollback and lease
expiry.

Recovery performs these steps:

1. Atomically acquire the expired lease.
2. Repeat discovery to establish a current authoritative source snapshot.
3. Treat matching SQL content with `indexed` search state as unchanged.
4. Rebuild `pending` Elasticsearch documents directly from SQL chunks.
5. Continue `delete_pending` Elasticsearch deletions.
6. Refetch only source documents whose content was never durably committed.
7. Recalculate counters from database state and resume normal pipeline flow.

Fetched but uncommitted bodies are intentionally not persisted. A crash may
repeat network fetching, but it does not repeat durable embedding or indexing
work. This avoids a temporary-body store and its cleanup lifecycle.

## Cancellation

Cancellation sets `cancel_requested_at`; workers observe it between bounded work
units. No new fetch or embedding work starts after cancellation. An active SQL
transaction is allowed to commit or roll back atomically.

SQL-committed `pending` documents are finished in Elasticsearch before the job
becomes `cancelled`, so cancellation cannot knowingly leave durable but
unsearchable documents. Already completed documents remain available. The final
report records how much work completed before cancellation.

## Compatibility

The same coordinator and state model serve SQLite and local providers. SQLite
uses a single writer and smaller batches. Local embedding and local search stages
remain no-ops or in-process operations. Production concurrency defaults apply
only to PostgreSQL, remote embedding, and Elasticsearch.

Existing single-document imports continue through the public
`import_text_document` API. Shared preparation and persistence primitives are
extracted so imports and the sync pipeline use identical chunking, metadata, and
ID rules without routing the production sync through the per-document API.

## Observability

Each run logs structured events containing job ID, data-source ID, phase, batch
size, duration, retries, and cumulative counters. Credentials, source bodies,
and embedding vectors are never logged.

Expose job progress through the existing data-source polling response by joining
the active job. A run with an expired lease is displayed as recovering or
stalled based on whether another worker has reclaimed it.

Track at least these timings:

- Discovery duration
- Fetch documents/second and bytes/second
- Embedding texts/second, latency, and 429 count
- PostgreSQL batch latency and rows/chunks per transaction
- Elasticsearch bulk latency, bytes, item errors, and refresh duration
- End-to-end indexed documents/second

## Verification Strategy

### Unit tests

- Queue backpressure never exceeds configured capacity.
- DashScope scheduling respects ten texts per request and global concurrency.
- Vectors are associated with the correct document chunks across batches.
- Deterministic IDs remain stable and change when content or index version changes.
- Progress counters remain monotonic and terminal status rules are correct.
- Retry classification distinguishes authentication, throttling, and content errors.

### Transaction and integration tests

- PostgreSQL batch failure rolls back documents, chunks, ingestion rows, and progress.
- Crash after SQL commit leaves recoverable `pending` documents.
- Crash after Elasticsearch success but before SQL acknowledgement is idempotent.
- Expired leases can be reclaimed; valid leases cannot be stolen.
- Two concurrent sync requests for one data source yield one accepted job and one 409.
- Partial discovery never triggers deletion.
- Cancellation never interrupts a SQL transaction halfway.

Use real PostgreSQL and Elasticsearch containers for integration behavior that
SQLite or fakes cannot represent, especially `FOR UPDATE SKIP LOCKED`, leases,
bulk item failures, and transaction rollback.

### Fault injection

Inject failures at every stage boundary:

- Fetch timeout, 401, 429, malformed body, and oversized body
- DashScope timeout, 401, 429, partial/missing embeddings, and invalid input
- PostgreSQL deadlock, disconnect before commit, and disconnect after commit
- Elasticsearch bulk transport error, per-item error, timeout, and successful
  bulk followed by process termination
- Worker cancellation during fetch, embedding, SQL, ES, and finalization

### Performance acceptance

With the production providers healthy and a warm service process, syncing the
current 2,144-document PAI manifest must:

- Complete within ten minutes
- Surface non-zero fetched progress within 15 seconds
- Surface non-zero indexed progress within 30 seconds
- Keep resident fetched-body buffering at or below the configured 100-document cap
- Perform no per-document Elasticsearch refresh
- Leave no `pending`, `delete_pending`, or expired-running state after successful completion

Record stage metrics during the benchmark so a failure identifies the limiting
service rather than only reporting an end-to-end timeout.

## Delivery Boundaries

The implementation changes the backend pipeline, migrations, queue lease logic,
progress API payloads, and focused frontend progress rendering. It does not add a
general scheduler, distributed queue product, persisted raw-body cache, or a
versioned shadow-index activation system.
