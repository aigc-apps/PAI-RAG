"""Bounded asynchronous data-source synchronization pipeline.

The coordinator is deliberately storage/provider agnostic. KnowledgeService
supplies callbacks for source fetch, document preparation, embedding, durable
SQL persistence, and search indexing; this module owns overlap, backpressure,
batch boundaries, progress, and cooperative cancellation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Awaitable, Callable, Sequence


@dataclass(frozen=True)
class PipelineLimits:
    fetch_concurrency: int = 16
    fetched_queue_size: int = 100
    sql_batch_documents: int = 25
    sql_batch_chunks: int = 2_000
    progress_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "fetch_concurrency", max(1, self.fetch_concurrency))
        object.__setattr__(self, "fetched_queue_size", max(1, self.fetched_queue_size))
        object.__setattr__(
            self, "sql_batch_documents", max(1, self.sql_batch_documents)
        )
        object.__setattr__(self, "sql_batch_chunks", max(1, self.sql_batch_chunks))
        object.__setattr__(
            self,
            "progress_interval_seconds",
            max(0.1, self.progress_interval_seconds),
        )


@dataclass(frozen=True)
class FetchedDocument:
    source: Any
    body: str

    @property
    def byte_count(self) -> int:
        return len(self.body.encode("utf-8"))


@dataclass(frozen=True)
class PreparedDocument:
    source: Any
    content: str
    chunks: list[dict]
    metadata: dict = field(default_factory=dict)
    unchanged: bool = False


@dataclass(frozen=True)
class EmbeddedDocument:
    prepared: PreparedDocument
    vectors: list[list[float]]

    @property
    def chunks(self) -> list[dict]:
        return self.prepared.chunks


@dataclass(frozen=True)
class PersistedBatch:
    documents: list[Any]
    payload: Any = None


@dataclass(frozen=True)
class SyncRunResult:
    status: str
    progress: dict
    errors: list[dict]


class SyncProgress:
    COUNTERS = (
        "total",
        "discovered",
        "fetched",
        "embedded",
        "persisted",
        "indexed",
        "unchanged",
        "deleted",
        "failed",
        "bytes_fetched",
    )

    def __init__(self, total: int) -> None:
        self.phase = "discovering"
        self._started = time.monotonic()
        self._values = {key: 0 for key in self.COUNTERS}
        self._values["total"] = total
        self._values["discovered"] = total

    def advance(self, field_name: str, amount: int = 1) -> None:
        if field_name not in self._values or field_name == "total":
            raise ValueError(f"unsupported progress counter {field_name!r}")
        if amount < 0:
            raise ValueError("progress counters cannot decrease")
        self._values[field_name] += amount

    def snapshot(self) -> dict:
        elapsed = max(time.monotonic() - self._started, 0.001)
        completed = self._values["indexed"] + self._values["unchanged"]
        rate = completed / elapsed
        remaining = max(self._values["total"] - completed - self._values["failed"], 0)
        eta = round(remaining / rate) if rate > 0 else None
        return {
            "phase": self.phase,
            **self._values,
            "docs_per_second": round(rate, 3),
            "estimated_seconds_remaining": eta,
        }


Fetch = Callable[[Any], Awaitable[FetchedDocument]]
Prepare = Callable[[FetchedDocument], Awaitable[PreparedDocument]]
Embed = Callable[[list[PreparedDocument]], Awaitable[list[EmbeddedDocument]]]
Persist = Callable[[list[EmbeddedDocument]], Awaitable[PersistedBatch]]
Index = Callable[[PersistedBatch], Awaitable[int]]
Checkpoint = Callable[[dict], Awaitable[None]]
CancelRequested = Callable[[], Awaitable[bool]]

_END = object()


class SyncPipeline:
    def __init__(
        self,
        *,
        limits: PipelineLimits,
        fetch: Fetch,
        prepare: Prepare,
        embed: Embed,
        persist_batch: Persist,
        index_batch: Index,
        checkpoint: Checkpoint,
        cancel_requested: CancelRequested,
    ) -> None:
        self._limits = limits
        self._fetch = fetch
        self._prepare = prepare
        self._embed = embed
        self._persist_batch = persist_batch
        self._index_batch = index_batch
        self._checkpoint = checkpoint
        self._cancel_requested = cancel_requested
        self.max_fetched_buffer = 0
        self._progress_lock = asyncio.Lock()
        self._last_checkpoint = 0.0

    async def _save_progress(self, progress: SyncProgress) -> None:
        async with self._progress_lock:
            snapshot = progress.snapshot()
            snapshot["max_fetched_buffer"] = self.max_fetched_buffer
            await self._checkpoint(snapshot)
            self._last_checkpoint = time.monotonic()

    async def _save_progress_if_due(self, progress: SyncProgress) -> None:
        async with self._progress_lock:
            now = time.monotonic()
            if now - self._last_checkpoint < self._limits.progress_interval_seconds:
                return
            snapshot = progress.snapshot()
            snapshot["max_fetched_buffer"] = self.max_fetched_buffer
            await self._checkpoint(snapshot)
            self._last_checkpoint = now

    async def run(self, discovered: Sequence[Any]) -> SyncRunResult:
        progress = SyncProgress(len(discovered))
        await self._save_progress(progress)
        errors: list[dict] = []
        cancelled = False
        descriptor_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self._limits.fetch_concurrency * 2
        )
        fetched_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self._limits.fetched_queue_size
        )
        prepared_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self._limits.fetched_queue_size
        )
        embedded_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self._limits.sql_batch_documents * 2
        )
        index_queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        def path_of(source: Any) -> str:
            return str(getattr(source, "path", getattr(source, "uri", "unknown")))

        def fail(source: Any, stage: str, exc: Exception) -> None:
            progress.advance("failed")
            if len(errors) < 20:
                errors.append(
                    {"path": path_of(source), "stage": stage, "error": str(exc)[:500]}
                )

        async def producer() -> None:
            nonlocal cancelled
            for descriptor in discovered:
                if await self._cancel_requested():
                    cancelled = True
                    break
                await descriptor_queue.put(descriptor)
            for _ in range(self._limits.fetch_concurrency):
                await descriptor_queue.put(_END)

        async def fetch_worker() -> None:
            nonlocal cancelled
            while True:
                descriptor = await descriptor_queue.get()
                if descriptor is _END:
                    return
                if await self._cancel_requested():
                    cancelled = True
                    continue
                try:
                    fetched = await self._fetch(descriptor)
                except Exception as exc:  # document-local boundary
                    fail(descriptor, "fetch", exc)
                    continue
                progress.phase = "fetching"
                progress.advance("fetched")
                progress.advance("bytes_fetched", fetched.byte_count)
                await fetched_queue.put(fetched)
                self.max_fetched_buffer = max(
                    self.max_fetched_buffer, fetched_queue.qsize()
                )
                await self._save_progress_if_due(progress)

        async def fetch_stage() -> None:
            async with asyncio.TaskGroup() as group:
                for _ in range(self._limits.fetch_concurrency):
                    group.create_task(fetch_worker())
            await fetched_queue.put(_END)

        async def prepare_stage() -> None:
            while True:
                fetched = await fetched_queue.get()
                if fetched is _END:
                    await prepared_queue.put(_END)
                    return
                try:
                    prepared = await self._prepare(fetched)
                except Exception as exc:
                    fail(fetched.source, "prepare", exc)
                    continue
                await prepared_queue.put(prepared)

        async def embed_stage() -> None:
            batch: list[PreparedDocument] = []
            chunk_count = 0

            async def flush() -> None:
                nonlocal batch, chunk_count
                if not batch:
                    return
                ready = batch
                batch = []
                chunk_count = 0
                unchanged = [doc for doc in ready if doc.unchanged]
                if unchanged:
                    progress.advance("unchanged", len(unchanged))
                work = [doc for doc in ready if not doc.unchanged]
                if not work:
                    return
                try:
                    embedded = await self._embed(work)
                except Exception as exc:
                    for doc in work:
                        fail(doc.source, "embedding", exc)
                    return
                progress.phase = "embedding"
                progress.advance("embedded", len(embedded))
                for document in embedded:
                    await embedded_queue.put(document)

            while True:
                prepared = await prepared_queue.get()
                if prepared is _END:
                    await flush()
                    await embedded_queue.put(_END)
                    return
                next_chunks = len(prepared.chunks)
                if batch and (
                    len(batch) >= self._limits.sql_batch_documents
                    or chunk_count + next_chunks > self._limits.sql_batch_chunks
                ):
                    await flush()
                batch.append(prepared)
                chunk_count += next_chunks

        async def persist_stage() -> None:
            batch: list[EmbeddedDocument] = []
            chunk_count = 0

            async def flush() -> None:
                nonlocal batch, chunk_count
                if not batch:
                    return
                ready = batch
                batch = []
                chunk_count = 0
                try:
                    persisted = await self._persist_batch(ready)
                except Exception as exc:
                    for doc in ready:
                        fail(doc.prepared.source, "persist", exc)
                    return
                progress.phase = "persisting"
                progress.advance("persisted", len(ready))
                await self._save_progress(progress)
                await index_queue.put(persisted)

            while True:
                embedded = await embedded_queue.get()
                if embedded is _END:
                    await flush()
                    await index_queue.put(_END)
                    return
                next_chunks = len(embedded.chunks)
                if batch and (
                    len(batch) >= self._limits.sql_batch_documents
                    or chunk_count + next_chunks > self._limits.sql_batch_chunks
                ):
                    await flush()
                batch.append(embedded)
                chunk_count += next_chunks

        async def index_stage() -> None:
            while True:
                batch = await index_queue.get()
                if batch is _END:
                    return
                try:
                    indexed = await self._index_batch(batch)
                except Exception as exc:
                    for doc in batch.documents:
                        source = getattr(doc, "prepared", doc)
                        fail(getattr(source, "source", source), "index", exc)
                    continue
                progress.phase = "indexing"
                progress.advance("indexed", indexed)
                await self._save_progress(progress)

        async with asyncio.TaskGroup() as group:
            group.create_task(producer())
            group.create_task(fetch_stage())
            group.create_task(prepare_stage())
            group.create_task(embed_stage())
            group.create_task(persist_stage())
            group.create_task(index_stage())

        successes = progress.snapshot()["indexed"] + progress.snapshot()["unchanged"]
        if cancelled:
            status = "cancelled"
        elif progress.snapshot()["failed"]:
            status = "partial" if successes else "failed"
        else:
            status = "succeeded"
        progress.phase = "completed" if status == "succeeded" else status
        final = progress.snapshot()
        final["max_fetched_buffer"] = self.max_fetched_buffer
        await self._checkpoint(final)
        return SyncRunResult(status=status, progress=final, errors=errors)
