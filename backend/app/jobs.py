"""Durable in-process background job queue + bounded worker pool.

The execution substrate for deferred work (KB ingest/sync today; cron-fired
runs later). Jobs live in the ``background_jobs`` table so they survive a
restart — a worker claims a ``queued`` row, dispatches on ``kind`` to a
registered handler, and records the outcome, retrying with backoff on failure.

Lean base: no Redis/Celery. One process, a small `asyncio` worker pool. Claiming
is dialect-aware — ``FOR UPDATE SKIP LOCKED`` on Postgres, an in-process lock on
SQLite (single writer). A non-terminal ``waiting`` status is reserved for a
future HITL pause: the worker never claims it; an external event returns it to
``queued``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

from loguru import logger
from sqlalchemy import or_, update
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import BackgroundJobRow
from app.store.base import User, _uuid

# A handler receives the job payload and returns an optional JSON-able result.
Handler = Callable[[dict], Awaitable[Optional[dict]]]

TERMINAL = {"succeeded", "failed"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobQueue:
    """Owns the queue table access + the worker pool for one engine."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        concurrency: int = 4,
        default_max_attempts: int = 3,
        poll_interval: float = 3.0,
        retry_backoff: float = 1.0,
    ) -> None:
        self._engine = engine
        self._concurrency = max(1, concurrency)
        self._default_max_attempts = max(1, default_max_attempts)
        self._poll_interval = poll_interval
        self._retry_backoff = retry_backoff
        self._dialect = engine.dialect.name
        self._handlers: dict[str, Handler] = {}
        self._wake = asyncio.Event()
        self._claim_lock = asyncio.Lock()   # serializes SQLite claims (single writer)
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    # -- registration ------------------------------------------------------
    def register(self, kind: str, handler: Handler) -> None:
        self._handlers[kind] = handler

    # -- producer ----------------------------------------------------------
    async def enqueue(
        self,
        *,
        kind: str,
        payload: Optional[dict] = None,
        kb_id: Optional[str] = None,
        created_by: Optional[str] = None,
        run_after: Optional[datetime] = None,
        max_attempts: Optional[int] = None,
    ) -> str:
        job = BackgroundJobRow(
            id=_uuid("job"),
            kind=kind,
            status="queued",
            payload=payload or {},
            kb_id=kb_id,
            created_by=created_by,
            run_after=run_after,
            max_attempts=max_attempts or self._default_max_attempts,
        )
        async with AsyncSession(self._engine, expire_on_commit=False) as s:
            s.add(job)
            await s.commit()
        self._wake.set()
        return job.id

    # -- claim (dialect-aware) --------------------------------------------
    async def claim_one(self, worker_id: str) -> Optional[BackgroundJobRow]:
        now = _now()
        stmt = (
            select(BackgroundJobRow)
            .where(
                BackgroundJobRow.status == "queued",
                or_(
                    BackgroundJobRow.run_after.is_(None),
                    BackgroundJobRow.run_after <= now,
                ),
            )
            .order_by(BackgroundJobRow.priority, BackgroundJobRow.created_at)
            .limit(1)
        )

        async def _do(session: AsyncSession) -> Optional[BackgroundJobRow]:
            if self._dialect == "postgresql":
                job = (await session.exec(stmt.with_for_update(skip_locked=True))).first()
            else:
                job = (await session.exec(stmt)).first()
            if job is None:
                return None
            job.status = "running"
            job.worker_id = worker_id
            job.claimed_at = now
            job.started_at = now
            job.updated_at = now
            session.add(job)
            await session.commit()
            return job

        async with AsyncSession(self._engine, expire_on_commit=False) as s:
            if self._dialect == "postgresql":
                # skip-locked lets concurrent claimers pass each other; no app lock.
                return await _do(s)
            # SQLite: one writer — serialize the select+update so two workers can't
            # both read the same queued row before either flips it to running.
            async with self._claim_lock:
                return await _do(s)

    # -- dispatch + outcome ------------------------------------------------
    async def _process(self, job: BackgroundJobRow) -> None:
        handler = self._handlers.get(job.kind)
        if handler is None:
            await self._fail(job.id, job.attempts, f"no handler for kind {job.kind!r}")
            return
        try:
            result = await handler(dict(job.payload or {}))
        except Exception as exc:  # noqa: BLE001 — record and (maybe) retry
            logger.warning(f"[jobs] {job.kind} job {job.id} failed: {exc!r}")
            await self._on_error(job, exc)
        else:
            await self._succeed(job.id, result)

    async def _succeed(self, job_id: str, result: Optional[dict]) -> None:
        async with AsyncSession(self._engine) as s:
            row = await s.get(BackgroundJobRow, job_id)
            if row is None:
                return
            row.status = "succeeded"
            row.result = result
            row.error = None
            row.finished_at = _now()
            row.updated_at = _now()
            s.add(row)
            await s.commit()

    async def _on_error(self, job: BackgroundJobRow, exc: Exception) -> None:
        attempts = job.attempts + 1
        if attempts < job.max_attempts:
            backoff = self._retry_backoff * (2 ** (attempts - 1))
            run_after = _now() + timedelta(seconds=backoff)
            async with AsyncSession(self._engine) as s:
                row = await s.get(BackgroundJobRow, job.id)
                if row is None:
                    return
                row.status = "queued"
                row.attempts = attempts
                row.run_after = run_after
                row.worker_id = None
                row.error = str(exc)
                row.updated_at = _now()
                s.add(row)
                await s.commit()
            self._wake.set()
        else:
            await self._fail(job.id, attempts, str(exc))

    async def _fail(self, job_id: str, attempts: int, error: str) -> None:
        async with AsyncSession(self._engine) as s:
            row = await s.get(BackgroundJobRow, job_id)
            if row is None:
                return
            row.status = "failed"
            row.attempts = attempts
            row.error = error
            row.finished_at = _now()
            row.updated_at = _now()
            s.add(row)
            await s.commit()

    # -- crash recovery ----------------------------------------------------
    async def recover_orphans(self) -> int:
        """Re-queue jobs left ``running`` by a crashed/restarted process.

        Handlers are idempotent (ingestion re-deletes+re-inserts chunks), so a
        re-run is safe. ``waiting`` is left alone — it is a deliberate pause.
        """
        async with AsyncSession(self._engine) as s:
            result = await s.exec(
                update(BackgroundJobRow)
                .where(BackgroundJobRow.status == "running")
                .values(status="queued", worker_id=None, updated_at=_now())
            )
            await s.commit()
        count = result.rowcount or 0
        if count:
            logger.info(f"[jobs] recovered {count} orphaned running job(s)")
            self._wake.set()
        return count

    # -- worker pool -------------------------------------------------------
    async def _worker(self, idx: int) -> None:
        worker_id = f"w{idx}"
        while not self._stopping:
            try:
                job = await self.claim_one(worker_id)
            except Exception as exc:  # noqa: BLE001 — never let the loop die
                logger.warning(f"[jobs] claim failed on {worker_id}: {exc!r}")
                job = None
            if job is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._process(job)

    def start(self) -> None:
        if self._tasks:
            return
        self._stopping = False
        self._tasks = [
            asyncio.create_task(self._worker(i), name=f"jobworker-{i}")
            for i in range(self._concurrency)
        ]
        logger.info(f"[jobs] started {self._concurrency} background worker(s)")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks = []

    # -- test/sync helper --------------------------------------------------
    async def run_until_empty(self, worker_id: str = "drain") -> int:
        """Claim + process jobs until none are currently available. Returns the
        number processed. Used by tests to drain the queue deterministically."""
        processed = 0
        while True:
            job = await self.claim_one(worker_id)
            if job is None:
                return processed
            await self._process(job)
            processed += 1


# --------------------------------------------------------------------------- #
# Handlers — bind a duck-typed KnowledgeService into the generic queue.
# Payload carries a serialized user ({user_id,user_email,user_role}) so the
# service's permission checks run as the enqueuing user.
# --------------------------------------------------------------------------- #
def _user_from_payload(payload: dict) -> User:
    return User(
        id=payload["user_id"],
        email=payload.get("user_email"),
        role=payload.get("user_role", "user"),
    )


def make_kb_ingest_handler(knowledge) -> Handler:
    async def handler(payload: dict) -> Optional[dict]:
        user = _user_from_payload(payload)
        doc, _job = await knowledge.import_text_document(
            payload["kb_id"],
            user=user,
            title=payload["title"],
            content=payload["content"],
            uri=payload.get("uri"),
            source_type=payload.get("source_type", "file"),
            source_id=payload.get("source_id"),
            mime_type=payload.get("mime_type", "text/markdown"),
            description=payload.get("description", ""),
            tags=payload.get("tags"),
            category=payload.get("category"),
            custom_metadata=payload.get("custom_metadata"),
            trigger_type=payload.get("trigger_type", "manual"),
        )
        return {"document_id": doc.id, "chunk_count": doc.chunk_count}

    return handler


def make_kb_sync_handler(knowledge) -> Handler:
    async def handler(payload: dict) -> Optional[dict]:
        user = _user_from_payload(payload)
        ds = await knowledge.sync_data_source(
            payload["kb_id"], payload["ds_id"], user=user
        )
        return {"status": ds.status, "doc_count": ds.doc_count}

    return handler


def register_knowledge_handlers(queue: JobQueue, knowledge) -> None:
    queue.register("kb_ingest", make_kb_ingest_handler(knowledge))
    queue.register("kb_sync", make_kb_sync_handler(knowledge))
