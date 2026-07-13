import asyncio
from dataclasses import dataclass

from app.sync_pipeline import (
    EmbeddedDocument,
    FetchedDocument,
    PersistedBatch,
    PipelineLimits,
    PreparedDocument,
    SyncPipeline,
)
from app.knowledge import _chunk_id


@dataclass(frozen=True)
class Descriptor:
    path: str


def test_chunk_id_is_stable_and_content_sensitive():
    first = _chunk_id("iv_1", "doc_1", 0, "hash-a")

    assert first == _chunk_id("iv_1", "doc_1", 0, "hash-a")
    assert first != _chunk_id("iv_1", "doc_1", 1, "hash-a")
    assert first != _chunk_id("iv_1", "doc_1", 0, "hash-b")
    assert first != _chunk_id("iv_2", "doc_1", 0, "hash-a")


def test_pipeline_overlaps_stages_and_respects_batch_and_queue_limits():
    async def scenario():
        active_fetches = 0
        max_active_fetches = 0
        first_persist_started = asyncio.Event()
        later_fetch_started = asyncio.Event()
        persist_batches: list[int] = []
        checkpoints: list[dict] = []

        async def fetch(descriptor):
            nonlocal active_fetches, max_active_fetches
            active_fetches += 1
            max_active_fetches = max(max_active_fetches, active_fetches)
            if descriptor.path == "doc-4":
                later_fetch_started.set()
            await asyncio.sleep(0.005)
            active_fetches -= 1
            return FetchedDocument(descriptor, f"body:{descriptor.path}")

        async def prepare(fetched):
            return PreparedDocument(
                fetched.source,
                fetched.body,
                [{"text": fetched.body, "heading_path": []}],
            )

        async def embed(documents):
            return [
                EmbeddedDocument(doc, [[float(index)] for index, _ in enumerate(doc.chunks)])
                for doc in documents
            ]

        async def persist(documents):
            persist_batches.append(len(documents))
            first_persist_started.set()
            await later_fetch_started.wait()
            return PersistedBatch(list(documents))

        async def index(batch):
            return len(batch.documents)

        async def checkpoint(progress):
            checkpoints.append(progress)

        async def not_cancelled():
            return False

        pipeline = SyncPipeline(
            limits=PipelineLimits(
                fetch_concurrency=3,
                fetched_queue_size=2,
                sql_batch_documents=2,
                sql_batch_chunks=10,
            ),
            fetch=fetch,
            prepare=prepare,
            embed=embed,
            persist_batch=persist,
            index_batch=index,
            checkpoint=checkpoint,
            cancel_requested=not_cancelled,
        )

        result = await pipeline.run([Descriptor(f"doc-{i}") for i in range(7)])

        assert result.status == "succeeded"
        assert result.progress["fetched"] == 7
        assert result.progress["embedded"] == 7
        assert result.progress["persisted"] == 7
        assert result.progress["indexed"] == 7
        assert persist_batches == [2, 2, 2, 1]
        assert max_active_fetches == 3
        assert pipeline.max_fetched_buffer <= 2
        assert first_persist_started.is_set()
        assert checkpoints[-1]["phase"] == "completed"

    asyncio.run(scenario())


def test_pipeline_continues_after_document_fetch_failure():
    async def scenario():
        async def fetch(descriptor):
            if descriptor.path == "bad":
                raise ValueError("invalid document")
            return FetchedDocument(descriptor, descriptor.path)

        async def prepare(fetched):
            return PreparedDocument(
                fetched.source,
                fetched.body,
                [{"text": fetched.body, "heading_path": []}],
            )

        async def embed(documents):
            return [EmbeddedDocument(doc, [[1.0]]) for doc in documents]

        async def persist(documents):
            return PersistedBatch(list(documents))

        async def index(batch):
            return len(batch.documents)

        async def checkpoint(progress):
            return None

        async def not_cancelled():
            return False

        pipeline = SyncPipeline(
            limits=PipelineLimits(fetch_concurrency=2, fetched_queue_size=2),
            fetch=fetch,
            prepare=prepare,
            embed=embed,
            persist_batch=persist,
            index_batch=index,
            checkpoint=checkpoint,
            cancel_requested=not_cancelled,
        )

        result = await pipeline.run(
            [Descriptor("good-1"), Descriptor("bad"), Descriptor("good-2")]
        )

        assert result.status == "partial"
        assert result.progress["indexed"] == 2
        assert result.progress["failed"] == 1
        assert result.errors == [
            {"path": "bad", "stage": "fetch", "error": "invalid document"}
        ]

    asyncio.run(scenario())


def test_pipeline_cancellation_stops_starting_new_fetches_and_drains_batches():
    async def scenario():
        fetch_calls = 0

        async def fetch(descriptor):
            nonlocal fetch_calls
            fetch_calls += 1
            await asyncio.sleep(0)
            return FetchedDocument(descriptor, descriptor.path)

        async def prepare(fetched):
            return PreparedDocument(fetched.source, fetched.body, [{"text": "x"}])

        async def embed(documents):
            return [EmbeddedDocument(doc, [[1.0]]) for doc in documents]

        async def persist(documents):
            return PersistedBatch(list(documents))

        async def index(batch):
            return len(batch.documents)

        async def checkpoint(progress):
            return None

        async def cancel_requested():
            return fetch_calls >= 2

        pipeline = SyncPipeline(
            limits=PipelineLimits(fetch_concurrency=1, fetched_queue_size=1),
            fetch=fetch,
            prepare=prepare,
            embed=embed,
            persist_batch=persist,
            index_batch=index,
            checkpoint=checkpoint,
            cancel_requested=cancel_requested,
        )

        result = await pipeline.run([Descriptor(f"doc-{i}") for i in range(10)])

        assert result.status == "cancelled"
        assert fetch_calls == 2
        assert result.progress["persisted"] == 2
        assert result.progress["indexed"] == 2

    asyncio.run(scenario())
