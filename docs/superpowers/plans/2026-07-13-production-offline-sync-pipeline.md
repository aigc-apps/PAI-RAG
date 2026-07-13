# Production Offline Sync Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the all-fetch-then-serial-import data-source sync with a durable, observable production pipeline that indexes the current 2,144-document PAI manifest in less than ten minutes.

**Architecture:** A bounded coordinator overlaps fetching, preparation, cross-document DashScope embedding, PostgreSQL batch persistence, and Elasticsearch bulk indexing. PostgreSQL is the source of truth; document-level search-index state and job leases make Elasticsearch work and interrupted jobs recoverable without repeating durable embedding work.

**Tech Stack:** Python 3.12, asyncio, FastAPI, SQLModel/SQLAlchemy async, Alembic, httpx, DashScope native embeddings, Elasticsearch async client, PostgreSQL, React 19, TypeScript, Vitest.

## Global Constraints

- Production target: PostgreSQL + DashScope + Elasticsearch.
- SQLite, local hash embeddings, and local retrieval remain functionally correct but are not throughput targets.
- A healthy 2,144-document PAI sync completes within ten minutes.
- Fetched progress becomes non-zero within 15 seconds and indexed progress within 30 seconds.
- Fetch concurrency defaults to 16 and fetched-body buffering is capped at 100 documents.
- DashScope requests contain at most 10 texts and default to 6 concurrent requests process-wide.
- PostgreSQL commits contain at most 25 documents or 2,000 chunks.
- Elasticsearch bulk payloads target 5 MiB and never exceed 10 MiB.
- Intermediate Elasticsearch operations use `refresh=false`; one refresh occurs at successful run completion.
- Progress is checkpointed after each committed batch or five seconds, whichever occurs first.
- No credentials, source bodies, or embedding vectors may be logged.

---

## File Structure

- Create `backend/app/sync_pipeline.py`: pipeline data types, retry classification, bounded stage coordinator, and progress accumulator.
- Create `backend/alembic/versions/20260713_1430_f2a9c6d8e1b4_offline_sync_pipeline.py`: durable job lease, progress, data-source active job, and document index-state columns.
- Create `backend/tests/test_sync_pipeline.py`: deterministic stage, backpressure, progress, retry, cancellation, and recovery tests.
- Create `backend/tests/test_offline_sync_postgres.py`: opt-in PostgreSQL transaction, lease, and concurrency integration tests.
- Create `backend/scripts/benchmark_offline_sync.py`: repeatable production-path timing harness using an already configured data source.
- Modify `backend/app/models.py`: new durable fields and index-state constants.
- Modify `backend/app/config.py`: bounded pipeline settings.
- Modify `backend/app/jobs.py`: leases, heartbeats, progress, cancellation, terminal outcomes, and expired-job claiming.
- Modify `backend/app/retrieval_models.py`: shared HTTP client and concurrent cross-document DashScope request scheduling.
- Modify `backend/app/search_engine.py`: multi-document delete/bulk/index-state contract and final refresh.
- Modify `backend/app/knowledge.py`: reusable preparation and SQL batch persistence primitives plus pipeline integration.
- Modify `backend/app/routes/knowledge.py`: atomic single-flight enqueue, live progress projection, and cancellation endpoint.
- Modify `backend/app/lean_main.py`: shared production limiters and removal of blind orphan recovery.
- Modify `backend/tests/test_jobs.py`, `backend/tests/test_retrieval_models.py`, `backend/tests/test_search_engine.py`, `backend/tests/test_routes_knowledge_datasource.py`, `backend/tests/test_migrations.py`, and `backend/tests/test_models.py`: focused regression coverage.
- Modify `frontend/src/api/knowledge.ts`: live progress and expanded sync-state types.
- Modify `frontend/src/components/KnowledgeView.tsx`: progress bar, phase counters, throughput, ETA, and cancel action.
- Modify `frontend/src/components/__tests__/KnowledgeView.test.tsx` or create it if absent: progress rendering and polling tests.
- Modify `frontend/src/i18n/en.ts` and `frontend/src/i18n/zh.ts`: progress and terminal-state copy.

---

### Task 1: Add Durable Pipeline Schema and Bounded Settings

**Files:**
- Create: `backend/alembic/versions/20260713_1430_f2a9c6d8e1b4_offline_sync_pipeline.py`
- Modify: `backend/app/models.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_models.py`
- Test: `backend/tests/test_migrations.py`
- Test: `backend/tests/test_config_database.py`

**Interfaces:**
- Produces: `BackgroundJobRow.progress`, `heartbeat_at`, `lease_expires_at`, and `cancel_requested_at`.
- Produces: `KnowledgeDataSourceRow.active_job_id`.
- Produces: `KnowledgeDocumentRow.search_index_status`, `search_index_error`, and `search_index_attempts`.
- Produces: validated `Settings.sync_*` limits consumed by later tasks.

- [ ] **Step 1: Write failing model and settings tests**

Add assertions equivalent to:

```python
def test_pipeline_model_defaults():
    job = BackgroundJobRow(id="job_1", kind="kb_sync")
    doc = KnowledgeDocumentRow(
        id="doc_1", kb_id="kb_1", title="x", created_by="u_1"
    )
    ds = KnowledgeDataSourceRow(
        id="ds_1", kb_id="kb_1", name="x", created_by="u_1"
    )
    assert job.progress == {}
    assert job.heartbeat_at is None
    assert job.lease_expires_at is None
    assert job.cancel_requested_at is None
    assert ds.active_job_id is None
    assert doc.search_index_status == "indexed"
    assert doc.search_index_error is None
    assert doc.search_index_attempts == 0


def test_pipeline_settings_are_bounded():
    settings = Settings(
        sync_fetch_concurrency=0,
        sync_fetch_queue_size=10000,
        sync_embedding_concurrency=100,
        sync_sql_batch_documents=0,
        sync_sql_batch_chunks=100000,
        sync_es_bulk_target_bytes=1,
        sync_es_bulk_max_bytes=1,
    )
    assert settings.sync_fetch_concurrency == 1
    assert settings.sync_fetch_queue_size == 1000
    assert settings.sync_embedding_concurrency == 16
    assert settings.sync_sql_batch_documents == 1
    assert settings.sync_sql_batch_chunks == 10000
    assert settings.sync_es_bulk_target_bytes == 1024 * 1024
    assert settings.sync_es_bulk_max_bytes == 1024 * 1024
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
cd backend
uv run pytest tests/test_models.py tests/test_config_database.py -q
```

Expected: failures reporting missing model fields and settings.

- [ ] **Step 3: Add model fields and validated settings**

Use timezone-aware fields via the existing `_utc_field`. Add these settings and clamp them in the existing model validator:

```python
sync_fetch_concurrency: int = 16
sync_fetch_queue_size: int = 100
sync_embedding_concurrency: int = 6
sync_sql_batch_documents: int = 25
sync_sql_batch_chunks: int = 2_000
sync_es_bulk_target_bytes: int = 5 * 1024 * 1024
sync_es_bulk_max_bytes: int = 10 * 1024 * 1024
sync_progress_interval_seconds: float = 5.0
job_heartbeat_seconds: float = 10.0
job_lease_seconds: float = 60.0

self.sync_fetch_concurrency = min(64, max(1, self.sync_fetch_concurrency))
self.sync_fetch_queue_size = min(1000, max(1, self.sync_fetch_queue_size))
self.sync_embedding_concurrency = min(
    16, max(1, self.sync_embedding_concurrency)
)
self.sync_sql_batch_documents = min(200, max(1, self.sync_sql_batch_documents))
self.sync_sql_batch_chunks = min(10_000, max(1, self.sync_sql_batch_chunks))
self.sync_es_bulk_target_bytes = min(
    10 * 1024 * 1024, max(1024 * 1024, self.sync_es_bulk_target_bytes)
)
self.sync_es_bulk_max_bytes = min(
    20 * 1024 * 1024,
    max(self.sync_es_bulk_target_bytes, self.sync_es_bulk_max_bytes),
)
self.sync_progress_interval_seconds = max(
    1.0, self.sync_progress_interval_seconds
)
self.job_heartbeat_seconds = max(1.0, self.job_heartbeat_seconds)
self.job_lease_seconds = max(
    self.job_heartbeat_seconds * 3, self.job_lease_seconds
)
```

Set new documents to `search_index_status="indexed"` for backward-compatible local and single-document behavior; the production batch writer explicitly sets `pending` before ES work.

- [ ] **Step 4: Add the Alembic migration**

Use revision `f2a9c6d8e1b4` with down revision `9a4e7b2c1d30`. Add columns with server-safe defaults, then remove defaults where the ORM owns them:

```python
def upgrade() -> None:
    op.add_column(
        "background_jobs",
        sa.Column(
            "progress",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column(
        "background_jobs",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "background_jobs",
        sa.Column(
            "lease_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "background_jobs",
        sa.Column(
            "cancel_requested_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "knowledge_data_sources",
        sa.Column("active_job_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_knowledge_data_sources_active_job_id",
        "knowledge_data_sources",
        ["active_job_id"],
    )
    op.add_column(
        "knowledge_documents",
        sa.Column(
            "search_index_status",
            sa.String(length=32),
            nullable=False,
            server_default="indexed",
        ),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column("search_index_error", sa.Text(), nullable=True),
    )
    op.add_column(
        "knowledge_documents",
        sa.Column(
            "search_index_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.create_index(
        "ix_knowledge_documents_search_index_status",
        "knowledge_documents",
        ["search_index_status"],
    )
```

The downgrade drops indexes before columns in the reverse order.

- [ ] **Step 5: Verify migration and model tests**

Run:

```bash
cd backend
uv run pytest tests/test_models.py tests/test_migrations.py tests/test_config_database.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the schema slice**

```bash
git add backend/app/models.py backend/app/config.py backend/alembic/versions/20260713_1430_f2a9c6d8e1b4_offline_sync_pipeline.py backend/tests/test_models.py backend/tests/test_migrations.py backend/tests/test_config_database.py
git commit -m "feat(sync): add durable pipeline state"
```

---

### Task 2: Replace Blind Recovery with Job Leases, Heartbeats, and Outcomes

**Files:**
- Modify: `backend/app/jobs.py`
- Modify: `backend/app/lean_main.py`
- Test: `backend/tests/test_jobs.py`

**Interfaces:**
- Produces: `JobOutcome(status: str, result: dict | None)`.
- Produces: `JobContext(job_id: str, queue: JobQueue)` with `checkpoint`, `cancel_requested`, and `raise_if_cancelled`.
- Produces: handlers with signature `async handler(payload: dict, ctx: JobContext) -> JobOutcome | dict | None`.
- Produces: `JobQueue.build_job`, `notify`, `request_cancel`, and expired-lease-aware `claim_one`.

- [ ] **Step 1: Write failing lease and cancellation tests**

Add tests for these exact behaviors:

```python
def test_valid_running_lease_cannot_be_stolen():
    async def scenario():
        q = await _fresh_queue(lease_seconds=60)
        job_id = await q.enqueue(kind="x")
        first = await q.claim_one("w1")
        assert first is not None and first.id == job_id
        assert await q.claim_one("w2") is None

    asyncio.run(scenario())


def test_expired_running_lease_is_reclaimed():
    async def scenario():
        q = await _fresh_queue(lease_seconds=60)
        job_id = await q.enqueue(kind="x")
        await q.claim_one("dead")
        async with AsyncSession(q._engine) as session:
            row = await session.get(BackgroundJobRow, job_id)
            row.lease_expires_at = _now() - timedelta(seconds=1)
            session.add(row)
            await session.commit()
        reclaimed = await q.claim_one("live")
        assert reclaimed is not None
        assert reclaimed.worker_id == "live"

    asyncio.run(scenario())


def test_checkpoint_and_cancel_are_durable():
    async def scenario():
        q = await _fresh_queue()
        job_id = await q.enqueue(kind="x")
        await q.checkpoint(job_id, {"phase": "fetching", "fetched": 8})
        await q.request_cancel(job_id)
        row = await _get(q, job_id)
        assert row.progress == {"phase": "fetching", "fetched": 8}
        assert row.cancel_requested_at is not None

    asyncio.run(scenario())
```

- [ ] **Step 2: Run job tests and verify failure**

Run `cd backend && uv run pytest tests/test_jobs.py -q`.

Expected: new tests fail because leases, progress, and cancellation APIs do not exist.

- [ ] **Step 3: Introduce handler context and terminal outcomes**

Add:

```python
@dataclass(frozen=True)
class JobOutcome:
    status: str = "succeeded"
    result: Optional[dict] = None


class JobCancelled(Exception):
    pass


@dataclass(frozen=True)
class JobContext:
    job_id: str
    queue: "JobQueue"

    async def checkpoint(self, progress: dict) -> None:
        await self.queue.checkpoint(self.job_id, progress)

    async def cancel_requested(self) -> bool:
        return await self.queue.is_cancel_requested(self.job_id)

    async def raise_if_cancelled(self) -> None:
        if await self.cancel_requested():
            raise JobCancelled(self.job_id)
```

Adapt registered knowledge handlers and test handlers through a compatibility wrapper so existing single-argument handlers still work during the transition.

- [ ] **Step 4: Implement atomic lease claiming and heartbeat**

The claim predicate is:

```python
or_(
    and_(BackgroundJobRow.status.in_(["queued", "retry_wait"]), due),
    and_(
        BackgroundJobRow.status == "running",
        BackgroundJobRow.lease_expires_at.is_not(None),
        BackgroundJobRow.lease_expires_at < now,
    ),
)
```

PostgreSQL continues to use `FOR UPDATE SKIP LOCKED`; SQLite keeps the in-process claim lock. On claim, set `heartbeat_at=now` and `lease_expires_at=now + lease_seconds`.

While a handler runs, start one heartbeat coroutine that updates only when both job ID and worker ID still match. Stop it in `_process` cleanup. `retry_wait` replaces queued-with-future-`run_after` for handler failures.

- [ ] **Step 5: Implement progress, cancellation, and outcome persistence**

`checkpoint` replaces the JSON value and updates `updated_at`; it does not increment attempts. `request_cancel` only affects non-terminal jobs. `JobCancelled` becomes a `cancelled` outcome. `JobOutcome(status="partial")` persists `partial` without passing through retry logic.

Remove `recover_orphans()` from application startup. Keep it as a deprecated test helper only if another caller requires it; otherwise delete it and update tests to exercise lease expiry.

- [ ] **Step 6: Run job and boot tests**

Run:

```bash
cd backend
uv run pytest tests/test_jobs.py tests/test_lean_main_boot.py tests/test_lean_import_isolation.py -q
```

Expected: all selected tests pass, including valid-lease protection and expired-lease reclaim.

- [ ] **Step 7: Commit the queue slice**

```bash
git add backend/app/jobs.py backend/app/lean_main.py backend/tests/test_jobs.py
git commit -m "feat(jobs): add leases progress and cancellation"
```

---

### Task 3: Make DashScope Embedding Cross-Document and Concurrent

**Files:**
- Modify: `backend/app/retrieval_models.py`
- Modify: `backend/app/providers.py`
- Test: `backend/tests/test_retrieval_models.py`
- Test: `backend/tests/test_providers.py`

**Interfaces:**
- Produces: `DashScopeEmbedder.embed` preserving input order while scheduling batches concurrently.
- Produces: `DashScopeEmbedder.aclose()` for application shutdown.
- Consumes: process-wide `asyncio.Semaphore` injected by `ProviderRouter`.

- [ ] **Step 1: Write failing concurrency, batching, and lifecycle tests**

Use a fake async client that records active requests and returns vectors tagged by input text. Assert:

```python
async def exercise():
    gate = asyncio.Semaphore(3)
    client = RecordingAsyncClient(delay=0.01)
    embedder = DashScopeEmbedder(
        base_url="https://ds/emb",
        api_key="k",
        model="text-embedding-v4",
        dimension=8,
        concurrency_gate=gate,
        client=client,
    )
    texts = [f"text-{i}" for i in range(25)]
    vectors = await embedder.embed(texts)
    assert [v[0] for v in vectors] == list(range(25))
    assert [len(call["input"]["texts"]) for call in client.payloads] == [
        10,
        10,
        5,
    ]
    assert 1 < client.max_active <= 3
    await embedder.aclose()
    assert client.closed is True
```

- [ ] **Step 2: Run retrieval tests and verify failure**

Run `cd backend && uv run pytest tests/test_retrieval_models.py -q`.

Expected: constructor rejects injected client/gate and calls are sequential.

- [ ] **Step 3: Reuse one HTTP client and schedule bounded requests**

Store an injected client or lazily create one `httpx.AsyncClient`. Convert each API-sized slice into a coroutine and run them under the shared semaphore:

```python
async def run_batch(start: int, batch: list[str]) -> tuple[int, list[dict]]:
    async with self._gate:
        response = await self._client_instance().post(
            self.base_url,
            headers=self._headers(),
            json=self._payload(batch, text_type),
        )
    response.raise_for_status()
    entries = (response.json().get("output") or {}).get("embeddings") or []
    return start, entries


results = await asyncio.gather(
    *(
        run_batch(start, items[start : start + self._batch])
        for start in range(0, len(items), self._batch)
    )
)
```

Place each returned vector at `start + text_index`, validate missing indices, and preserve the existing public protocol. Add `aclose` without closing an injected client unless ownership was explicitly transferred.

- [ ] **Step 4: Inject the shared semaphore from provider construction**

`ProviderRouter` accepts `embedding_concurrency: int = 6`, creates one semaphore, and passes it to every DashScope embedder it caches. Application shutdown closes cached retrieval clients.

- [ ] **Step 5: Run provider and retrieval tests**

Run:

```bash
cd backend
uv run pytest tests/test_retrieval_models.py tests/test_providers.py tests/test_knowledge_retrieval_models.py -q
```

Expected: all selected tests pass and maximum fake-client concurrency respects the gate.

- [ ] **Step 6: Commit the embedding slice**

```bash
git add backend/app/retrieval_models.py backend/app/providers.py backend/tests/test_retrieval_models.py backend/tests/test_providers.py
git commit -m "perf(sync): batch concurrent DashScope embeddings"
```

---

### Task 4: Add Multi-Document Elasticsearch Bulk Operations

**Files:**
- Modify: `backend/app/search_engine.py`
- Test: `backend/tests/test_search_engine.py`

**Interfaces:**
- Produces: `index_document_batch(kb, documents, *, refresh=False) -> BatchIndexResult`.
- Produces: `delete_document_batch(kb_id, document_ids, *, refresh=False) -> None`.
- Produces: `refresh_kb(kb_id) -> None`.
- Keeps: `index_chunks` as a compatibility wrapper around the batch API.

- [ ] **Step 1: Write failing multi-document bulk tests**

Test a two-document batch and assert one terms-based delete, size-bounded bulk calls, no intermediate refresh, and one explicit final refresh:

```python
async def exercise():
    client = FakeESClient()
    engine = ElasticsearchEngine(
        "http://es:9200",
        client_factory=lambda: client,
        bulk_target_bytes=1024,
        bulk_max_bytes=2048,
    )
    result = await engine.index_document_batch(
        FakeKB(), [(FakeDoc("d1"), chunks(3)), (FakeDoc("d2"), chunks(4))]
    )
    assert client.delete_queries == [{"terms": {"document_id": ["d1", "d2"]}}]
    assert all(call["refresh"] is False for call in client.bulk_calls)
    assert result.failed_document_ids == set()
    await engine.refresh_kb("kb_1")
    assert client.refreshed == ["kb-kb_1"]
```

- [ ] **Step 2: Run search-engine tests and verify failure**

Run `cd backend && uv run pytest tests/test_search_engine.py -q`.

Expected: batch methods and bulk-size constructor arguments are missing.

- [ ] **Step 3: Implement deterministic operation packing**

Add:

```python
@dataclass(frozen=True)
class BatchIndexResult:
    indexed_document_ids: set[str]
    failed_document_ids: set[str]
    errors: dict[str, str]
```

Serialize each action/source pair with compact JSON to estimate bytes. Flush before adding an operation that would cross the target. Reject a single operation exceeding the hard maximum with its document ID in `errors`; never send a request over the hard cap.

Run one `delete_by_query` using a `terms` query over all document IDs in the SQL batch, then bulk index chunks with `refresh=False`. Parse per-item bulk errors back to document IDs using an in-memory chunk-ID-to-document-ID map.

- [ ] **Step 4: Preserve compatibility and add final refresh**

Implement `index_chunks` by calling `index_document_batch(kb, [(doc, chunks)])`. Implement `delete_document` with `delete_document_batch`. Neither wrapper refreshes by default; explicit single-document API callers pass `refresh=True` where immediate visibility is part of their contract.

- [ ] **Step 5: Run search tests**

Run `cd backend && uv run pytest tests/test_search_engine.py tests/test_routes_knowledge.py -q`.

Expected: selected tests pass, including old single-document behavior and new multi-document batching.

- [ ] **Step 6: Commit the Elasticsearch slice**

```bash
git add backend/app/search_engine.py backend/tests/test_search_engine.py backend/tests/test_routes_knowledge.py
git commit -m "perf(sync): add Elasticsearch document batching"
```

---

### Task 5: Build the Bounded Pipeline Coordinator

**Files:**
- Create: `backend/app/sync_pipeline.py`
- Create: `backend/tests/test_sync_pipeline.py`

**Interfaces:**
- Produces: immutable `FetchedDocument`, `PreparedDocument`, `EmbeddedDocument`, and `FailedDocument` records.
- Produces: `SyncProgress` with monotonic counters and `as_dict()`.
- Produces: `PipelineLimits` and `SyncPipeline.run(discovered) -> SyncRunResult`.
- Consumes callbacks `fetch`, `prepare`, `embed`, `persist_batch`, `index_batch`, `checkpoint`, and `cancel_requested`.

- [ ] **Step 1: Write failing backpressure and overlap tests**

Construct callbacks controlled by events and assert fetching continues while persistence is blocked, but never exceeds queue capacity:

```python
async def test_pipeline_overlaps_stages_and_applies_backpressure():
    probes = PipelineProbes()
    pipeline = SyncPipeline(
        limits=PipelineLimits(
            fetch_concurrency=4,
            fetched_queue_size=3,
            sql_batch_documents=2,
            sql_batch_chunks=20,
        ),
        fetch=probes.fetch,
        prepare=probes.prepare,
        embed=probes.embed,
        persist_batch=probes.persist_batch,
        index_batch=probes.index_batch,
        checkpoint=probes.checkpoint,
        cancel_requested=probes.cancel_requested,
    )
    result = await pipeline.run(make_discovered(12))
    assert result.status == "succeeded"
    assert probes.fetch_started_before_first_persist_finished is True
    assert probes.max_fetched_buffer <= 3
    assert probes.persist_batch_sizes == [2, 2, 2, 2, 2, 2]
```

Also test cancellation, per-document fetch failure, run-wide authentication failure, monotonic progress, sentinel propagation, and absence of leaked tasks after completion.

- [ ] **Step 2: Run pipeline tests and verify failure**

Run `cd backend && uv run pytest tests/test_sync_pipeline.py -q`.

Expected: import failure because `app.sync_pipeline` does not exist.

- [ ] **Step 3: Implement records, progress, and retry classification**

Define explicit dataclasses. `SyncProgress.advance(field, amount)` rejects counter regression. `as_dict` includes phase, counters, elapsed throughput, and ETA when total and throughput are known.

Classify errors into:

```python
class FailureScope(StrEnum):
    DOCUMENT = "document"
    TRANSIENT_RUN = "transient_run"
    PERMANENT_RUN = "permanent_run"


@dataclass(frozen=True)
class ClassifiedFailure:
    scope: FailureScope
    retryable: bool
    message: str
```

HTTP 401/403 and invalid configuration are permanent run failures. HTTP 429, timeout, connection reset, PostgreSQL serialization/deadlock, and ES 429/5xx are transient. Content parse and oversized-body errors are document-local.

- [ ] **Step 4: Implement bounded stages with structured task cleanup**

Use one queue per boundary and `asyncio.TaskGroup`. Producers always place one sentinel per downstream consumer in `finally`. The coordinator owns cancellation; a run-wide failure sets a shared stop event and lets each stage finish its current bounded unit.

The SQL batch collector flushes when either document or chunk limit is reached. It sends the committed batch to the index queue only after `persist_batch` succeeds. Checkpoint after batch commits and after index acknowledgements, throttled by elapsed monotonic time.

- [ ] **Step 5: Implement terminal result rules**

Return `succeeded` when all discovered documents are indexed or unchanged and no permanent document failures remain. Return `partial` when at least one document is indexed/unchanged and at least one permanently failed. Return `failed` when a run-wide failure occurs or no document succeeds. Return `cancelled` when cancellation is requested after draining committed index-pending work.

- [ ] **Step 6: Run pipeline tests with asyncio debug**

Run:

```bash
cd backend
PYTHONASYNCIODEBUG=1 uv run pytest tests/test_sync_pipeline.py -q
```

Expected: all tests pass without pending-task or un-awaited-coroutine warnings.

- [ ] **Step 7: Commit the coordinator slice**

```bash
git add backend/app/sync_pipeline.py backend/tests/test_sync_pipeline.py
git commit -m "feat(sync): add bounded offline pipeline"
```

---

### Task 6: Add Shared Preparation and PostgreSQL Batch Persistence

**Files:**
- Modify: `backend/app/knowledge.py`
- Modify: `backend/app/sync_pipeline.py`
- Test: `backend/tests/test_sync_pipeline.py`
- Test: `backend/tests/test_routes_knowledge_datasource.py`

**Interfaces:**
- Produces: `KnowledgeService.prepare_source_document(kb, source_doc) -> PreparedDocument`.
- Produces: `KnowledgeService.embed_prepared_documents(kb, docs) -> list[EmbeddedDocument]`.
- Produces: `KnowledgeService.persist_document_batch(...) -> PersistedBatch`.
- Produces: deterministic `_chunk_id(index_version_id, document_id, chunk_index, text_hash) -> str`.
- Keeps: `import_text_document` using the same preparation/persistence primitives.

- [ ] **Step 1: Write failing deterministic-ID and transaction tests**

Add tests asserting stable IDs and complete rollback:

```python
def test_chunk_id_is_stable_and_content_sensitive():
    first = _chunk_id("iv_1", "doc_1", 0, "hash-a")
    assert first == _chunk_id("iv_1", "doc_1", 0, "hash-a")
    assert first != _chunk_id("iv_1", "doc_1", 1, "hash-a")
    assert first != _chunk_id("iv_1", "doc_1", 0, "hash-b")
    assert first != _chunk_id("iv_2", "doc_1", 0, "hash-a")


async def test_persist_batch_rolls_back_content_chunks_and_progress(
    monkeypatch,
):
    service, kb, run = await make_service_and_run()
    batch = await make_embedded_batch(2)
    monkeypatch.setattr(service, "_before_batch_commit", raising_hook)
    with pytest.raises(RuntimeError, match="injected commit failure"):
        await service.persist_document_batch(kb, run, batch)
    assert await count_documents(service, kb.id) == 0
    assert await count_chunks(service, kb.id) == 0
    assert (await load_job(service, run.job_id)).progress.get(
        "persisted", 0
    ) == 0
```

- [ ] **Step 2: Run focused tests and verify failure**

Run `cd backend && uv run pytest tests/test_sync_pipeline.py tests/test_routes_knowledge_datasource.py -q`.

Expected: missing preparation, batch persistence, and deterministic-ID APIs.

- [ ] **Step 3: Extract pure preparation from single-document import**

Move split, stored-content capping, metadata shaping, and embedding-input construction into reusable helpers. Preparation performs no database writes. `import_text_document` calls the same helpers so chunking and metadata remain identical across manual imports and data-source syncs.

- [ ] **Step 4: Implement cross-document embedding association**

Flatten prepared chunks into `(document_position, chunk_position, text)` entries, call the KB embedder once for the flattened text sequence, and rebuild embedded documents by the recorded positions. Reject a vector count mismatch before any SQL write.

- [ ] **Step 5: Implement one SQL transaction per batch**

Within one `AsyncSession.begin()`:

1. Load the KB and current documents for all batch URIs in one query.
2. Assign existing or new document IDs.
3. Delete old content/chunks with set-based predicates.
4. Upsert document and content rows.
5. Insert deterministic chunk rows and completed ingestion rows.
6. Set `search_index_status="pending"`, clear index error, and reset attempts.
7. Refresh KB counts once.
8. Update the background job's persisted progress in the same transaction.

Return detached document metadata and ES chunk payloads as `PersistedBatch`; do not call Elasticsearch inside this method.

- [ ] **Step 6: Preserve SQLite and manual-import behavior**

Use one writer batch at a time for SQLite. Manual imports call the batch primitive with one document, then use the compatibility ES batch wrapper with immediate refresh. Verify current upload, pagination, content, and retrieval tests.

- [ ] **Step 7: Run knowledge regression tests**

Run:

```bash
cd backend
uv run pytest tests/test_routes_knowledge.py tests/test_routes_knowledge_upload.py tests/test_routes_knowledge_datasource.py tests/test_pagination.py tests/test_search_engine.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit the persistence slice**

```bash
git add backend/app/knowledge.py backend/app/sync_pipeline.py backend/tests/test_sync_pipeline.py backend/tests/test_routes_knowledge_datasource.py
git commit -m "perf(sync): persist documents in PostgreSQL batches"
```

---

### Task 7: Integrate Recovery, Single-Flight Enqueue, and Cancellation

**Files:**
- Modify: `backend/app/knowledge.py`
- Modify: `backend/app/jobs.py`
- Modify: `backend/app/routes/knowledge.py`
- Modify: `backend/app/lean_main.py`
- Test: `backend/tests/test_routes_knowledge_datasource.py`
- Test: `backend/tests/test_jobs.py`

**Interfaces:**
- Produces: `KnowledgeService.enqueue_data_source_sync(queue, kb_id, ds_id, user) -> tuple[KnowledgeDataSourceRow, str]`.
- Produces: `KnowledgeService.run_data_source_pipeline(..., ctx: JobContext) -> JobOutcome`.
- Produces: `POST /v1/knowledge-bases/{kb_id}/datasources/{ds_id}/sync/cancel`.
- Produces: data-source JSON containing `active_job_id` and `sync_progress`.

- [ ] **Step 1: Write failing atomic enqueue and recovery tests**

Add route/service tests:

```python
def test_concurrent_sync_requests_have_one_winner():
    c = _client()
    kb, ds = create_kb_and_source(c)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda _: c.post(
                    f"/v1/knowledge-bases/{kb['id']}/datasources/{ds['id']}/sync"
                ),
                range(2),
            )
        )
    assert sorted(r.status_code for r in responses) == [202, 409]


async def test_recovery_indexes_pending_from_sql_without_refetch():
    service, queue, kb, ds = await persisted_pending_fixture()
    fetch = AsyncMock(
        side_effect=AssertionError("pending docs must not refetch")
    )
    outcome = await service.run_data_source_pipeline(
        kb.id,
        ds.id,
        user=ADMIN,
        ctx=claimed_context(queue),
        fetch_override=fetch,
    )
    assert outcome.status == "succeeded"
    assert await pending_count(service, ds.id) == 0
```

Also test cancellation after SQL commit finishes pending ES work, active job clearing on every terminal outcome, and partial discovery suppressing deletion.

- [ ] **Step 2: Run route and queue tests and verify failure**

Run `cd backend && uv run pytest tests/test_routes_knowledge_datasource.py tests/test_jobs.py -q`.

Expected: two requests can enqueue, recovery API is absent, and cancel route returns 404.

- [ ] **Step 3: Implement atomic single-flight enqueue**

Generate the job row before the transaction. In one PostgreSQL/SQLite transaction, validate permissions and enabled state, compare-and-set `active_job_id` only when null, add the background job row, and set aggregate status to `syncing`. Commit, then call `queue.notify()`.

The route maps a failed compare-and-set to HTTP 409 and returns `job_id` on success. This replaces the read-status-then-enqueue sequence.

- [ ] **Step 4: Replace `sync_data_source` internals with the pipeline**

The job handler calls `run_data_source_pipeline`. Before normal fetching it:

1. Loads `pending` and `delete_pending` documents for this data source.
2. Reindexes pending documents directly from SQL chunks.
3. Completes pending deletions.
4. Runs authoritative discovery.
5. Streams remaining discovered documents through the bounded pipeline.

Matching `indexed` content hashes become unchanged. Only a complete discovery snapshot schedules stale documents for deletion.

- [ ] **Step 5: Implement finalization and cancellation**

Finalization uses one transaction to set the data-source terminal status/report, `last_sync_finished_at`, and `doc_count`; clear `active_job_id` only when it still equals the finishing job ID; and persist the terminal job outcome.

The cancel endpoint validates manage permission and calls `queue.request_cancel(active_job_id)`. Pipeline producers stop creating work, SQL transactions remain atomic, and already persisted `pending` documents drain through ES before returning `cancelled`.

- [ ] **Step 6: Project live progress in list/get responses**

When a data source has `active_job_id`, load its job and include:

```json
{
  "active_job_id": "job_123",
  "sync_progress": {
    "phase": "indexing",
    "total": 2144,
    "fetched": 200,
    "embedded": 175,
    "persisted": 150,
    "indexed": 125,
    "failed": 1,
    "docs_per_second": 4.8,
    "estimated_seconds_remaining": 415
  }
}
```

Do not expose job payloads, credentials, raw errors over 500 characters, or worker IDs.

- [ ] **Step 7: Run service, route, and boot regressions**

Before running the tests, add one structured log event at discovery completion,
each durable batch boundary, lease reclaim, cancellation, and finalization. Bind
`job_id`, `datasource_id`, `phase`, batch document/chunk counts, duration,
retries, and cumulative counters through Loguru. Pass only numeric metrics and
stable IDs; do not bind source bodies, source configuration, credentials, or
vectors. Emit stage durations and throughput in the progress payload so the
benchmark can identify fetch, embedding, SQL, or Elasticsearch as the limiting
stage without parsing free-form messages.

Run:

```bash
cd backend
uv run pytest tests/test_routes_knowledge_datasource.py tests/test_jobs.py tests/test_lean_main_boot.py tests/test_pagination.py -q
```

Expected: all selected tests pass, including 202/409 concurrency and pending-index recovery.

- [ ] **Step 8: Commit the integration slice**

```bash
git add backend/app/knowledge.py backend/app/jobs.py backend/app/routes/knowledge.py backend/app/lean_main.py backend/tests/test_routes_knowledge_datasource.py backend/tests/test_jobs.py
git commit -m "feat(sync): add recovery and atomic lifecycle"
```

---

### Task 8: Render Live Sync Progress in the Frontend

**Files:**
- Modify: `frontend/src/api/knowledge.ts`
- Modify: `frontend/src/components/KnowledgeView.tsx`
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/i18n/zh.ts`
- Create or modify: `frontend/src/components/__tests__/KnowledgeView.test.tsx`

**Interfaces:**
- Consumes: backend `active_job_id` and `sync_progress` fields.
- Produces: typed `KnowledgeSyncProgress` and cancel API method.
- Produces: accessible progressbar and cancel button.

- [ ] **Step 1: Write failing progress rendering tests**

Mock one syncing data source and assert:

```tsx
expect(await screen.findByRole("progressbar", { name: "同步进度" })).toHaveAttribute(
  "aria-valuenow", "125"
)
expect(screen.getByText("125 / 2144")).toBeInTheDocument()
expect(screen.getByText(/4.8.*篇\/秒/)).toBeInTheDocument()
expect(screen.getByText(/预计剩余/)).toBeInTheDocument()
expect(screen.getByRole("button", { name: "取消同步" })).toBeEnabled()
```

Also assert polling continues for active jobs even when aggregate status is stale, and stops after a terminal response.

- [ ] **Step 2: Run the frontend test and verify failure**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/KnowledgeView.test.tsx
```

Expected: progressbar, counters, and cancel control are absent.

- [ ] **Step 3: Add API types and cancellation client**

Add:

```typescript
export interface KnowledgeSyncProgress {
  phase: "discovering" | "fetching" | "embedding" | "persisting" | "indexing" | "finalizing" | "recovering";
  total: number;
  discovered: number;
  fetched: number;
  embedded: number;
  persisted: number;
  indexed: number;
  unchanged: number;
  deleted: number;
  failed: number;
  bytes_fetched: number;
  docs_per_second: number;
  estimated_seconds_remaining: number | null;
}
```

Add nullable `active_job_id` and `sync_progress` to `KnowledgeDataSource`, add `cancelled` to terminal status, and implement `cancelDataSourceSync(kbId, dsId)` using the new POST endpoint.

- [ ] **Step 4: Render progress and phase details**

Compute percentage from `indexed + unchanged + failed` over `total`, clamp to 0–100, and render an accessible progressbar. Show indexed/total, localized phase, failed count when nonzero, documents/second, and humanized ETA. Keep the existing terminal report after completion.

Poll while `active_job_id` is non-null, not only while `status === "syncing"`. The cancel button calls the endpoint once, disables during the request, and leaves polling active until terminal acknowledgement.

- [ ] **Step 5: Add English and Chinese copy**

Add explicit keys for all phases, indexed/total, throughput, ETA, cancelling, cancelled, cancel action, and failure count. Chinese uses “篇/秒”; English uses “docs/s”.

- [ ] **Step 6: Run frontend tests and type checking**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/KnowledgeView.test.tsx
npm run build
```

Expected: test passes and TypeScript/Vite production build succeeds.

- [ ] **Step 7: Commit the frontend slice**

```bash
git add frontend/src/api/knowledge.ts frontend/src/components/KnowledgeView.tsx frontend/src/components/__tests__/KnowledgeView.test.tsx frontend/src/i18n/en.ts frontend/src/i18n/zh.ts
git commit -m "feat(sync): show live offline progress"
```

---

### Task 9: Add PostgreSQL Fault Tests and the Ten-Minute Benchmark

**Files:**
- Create: `backend/tests/test_offline_sync_postgres.py`
- Create: `backend/scripts/benchmark_offline_sync.py`
- Modify: `backend/pyproject.toml` only if an existing pytest marker must be registered
- Modify: `docs/superpowers/specs/2026-07-13-offline-sync-pipeline-design.md` only if measured provider constraints require a documented correction

**Interfaces:**
- Consumes: completed pipeline, job lease, recovery, and progress APIs.
- Produces: opt-in `PAIRAG_TEST_POSTGRES_URL` integration suite.
- Produces: benchmark command taking KB ID, data-source ID, and a 600-second limit.

- [ ] **Step 1: Write PostgreSQL-only transaction and lease tests**

Skip cleanly unless `PAIRAG_TEST_POSTGRES_URL` is set. Cover:

```python
@pytest.mark.postgres
async def test_skip_locked_claims_have_one_owner(pg_engine):
    queue = JobQueue(pg_engine, concurrency=2)
    job_id = await queue.enqueue(kind="noop")
    first, second = await asyncio.gather(
        queue.claim_one("worker-a"), queue.claim_one("worker-b")
    )
    assert [job.id for job in (first, second) if job is not None] == [job_id]


@pytest.mark.postgres
async def test_sql_commit_then_es_crash_recovers_without_embedding(pg_fixture):
    run = await pg_fixture.persist_batch_then_fail_es()
    assert await pg_fixture.pending_documents(run) > 0
    pg_fixture.embedder.fail_if_called = True
    await pg_fixture.reclaim_and_finish(run)
    assert await pg_fixture.pending_documents(run) == 0
    assert pg_fixture.embedder.calls == 0
```

Add cases for transaction rollback, valid lease protection, expired lease reclaim, ES success followed by process-level interruption before SQL acknowledgement, partial discovery deletion suppression, and cancellation during each stage boundary.

- [ ] **Step 2: Run unit suite and opt-in PostgreSQL suite**

Run:

```bash
cd backend
uv run pytest -q
PAIRAG_TEST_POSTGRES_URL="$PAIRAG_TEST_POSTGRES_URL" uv run pytest tests/test_offline_sync_postgres.py -q
```

Expected: complete unit suite passes; PostgreSQL suite passes when configured and otherwise reports clean skips.

- [ ] **Step 3: Implement the benchmark harness**

The script loads normal settings, validates that the configured backend is PostgreSQL and vector engine is Elasticsearch, enqueues the requested data source, polls the background job once per second, prints one JSON line per progress change, and exits nonzero when duration exceeds 600 seconds or status is not `succeeded`.

Command:

```bash
cd backend
uv run python scripts/benchmark_offline_sync.py \
  --kb-id kb_bNcZCeaYJT9 \
  --datasource-id ds_AVdE1d99u31 \
  --timeout-seconds 600
```

The final JSON contains duration, discovered, fetched, embedded, persisted, indexed, unchanged, failed, average indexed documents/second, and maximum observed fetched-minus-persisted buffer.

- [ ] **Step 4: Run the production-path benchmark**

Run the command against the configured PostgreSQL, DashScope, and Elasticsearch services.

Expected acceptance:

```text
status=succeeded
duration_seconds<=600
first_fetched_seconds<=15
first_indexed_seconds<=30
max_fetched_buffer<=100
pending_documents=0
delete_pending_documents=0
expired_running_jobs=0
```

If provider throttling prevents the target, adjust only configured concurrency and batch-byte limits, rerun, and record the measured limiting service. Do not weaken correctness or progress acceptance criteria.

- [ ] **Step 5: Run final backend and frontend verification**

Run:

```bash
cd backend
uv run pytest -q
cd ../frontend
npm test -- --run
npm run build
```

Expected: all backend tests, all frontend tests, and production build pass.

- [ ] **Step 6: Commit verification assets**

```bash
git add backend/tests/test_offline_sync_postgres.py backend/scripts/benchmark_offline_sync.py backend/pyproject.toml docs/superpowers/specs/2026-07-13-offline-sync-pipeline-design.md
git commit -m "test(sync): verify recovery and production throughput"
```

---

## Final Review Gate

Before declaring completion:

1. Confirm every database change has an upgrade and downgrade migration path.
2. Confirm no valid running lease is reclaimed during multi-process tests.
3. Confirm no terminal successful run leaves `pending` or `delete_pending` documents.
4. Confirm progress counters never decrease during one job attempt.
5. Confirm the frontend polls by `active_job_id` and handles partial, failed, cancelled, and recovered runs.
6. Confirm intermediate Elasticsearch requests never use refresh.
7. Confirm logs contain identifiers and metrics but no source bodies, credentials, or vectors.
8. Confirm the 2,144-document benchmark meets the ten-minute, 15-second progress, and 30-second indexed-progress targets.
