"""Durable background job queue — claim/dispatch/retry/recover mechanics.

Offline, no heavy deps: handlers are plain in-memory callables. Each test runs
its own scenario coroutine under asyncio.run against an in-memory SQLite engine
(StaticPool → one shared connection across create_all and the queue sessions).
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import timedelta

from app.db import create_all, make_engine
from app.jobs import JobQueue, _now
from app.models import BackgroundJobRow
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession


async def _fresh_queue(**kw) -> JobQueue:
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    return JobQueue(engine, concurrency=1, **kw)


async def _get(q: JobQueue, job_id: str) -> BackgroundJobRow:
    async with AsyncSession(q._engine) as s:
        return await s.get(BackgroundJobRow, job_id)


def test_enqueue_claim_process_succeeds():
    async def scenario():
        q = await _fresh_queue()
        seen = {}

        async def handler(payload):
            seen["payload"] = payload
            return {"echo": payload["n"] * 2}

        q.register("double", handler)
        job_id = await q.enqueue(kind="double", payload={"n": 21})

        processed = await q.run_until_empty()
        assert processed == 1
        assert seen["payload"] == {"n": 21}

        row = await _get(q, job_id)
        assert row.status == "succeeded"
        assert row.result == {"echo": 42}
        assert row.error is None
        assert row.attempts == 0
        assert row.finished_at is not None

    asyncio.run(scenario())


def test_handler_failure_retries_then_fails():
    async def scenario():
        # retry_backoff=0 → requeued jobs are immediately claimable, so a single
        # run_until_empty drives all attempts in one drain loop.
        q = await _fresh_queue(retry_backoff=0.0)
        calls = {"n": 0}

        async def boom(payload):
            calls["n"] += 1
            raise RuntimeError("nope")

        q.register("boom", boom)
        job_id = await q.enqueue(kind="boom", payload={}, max_attempts=3)

        processed = await q.run_until_empty()
        assert processed == 3          # attempt 1, 2, 3 all claimed+run
        assert calls["n"] == 3

        row = await _get(q, job_id)
        assert row.status == "failed"
        assert row.attempts == 3
        assert "nope" in row.error
        assert row.finished_at is not None

    asyncio.run(scenario())


def test_retry_backoff_defers_next_attempt():
    async def scenario():
        # Non-zero backoff → after the first failure the job's run_after is in the
        # future, so it is NOT immediately re-claimable within the same drain.
        q = await _fresh_queue(retry_backoff=60.0)

        async def boom(payload):
            raise RuntimeError("later")

        q.register("boom", boom)
        job_id = await q.enqueue(kind="boom", payload={}, max_attempts=3)

        processed = await q.run_until_empty()
        assert processed == 1          # only the first attempt runs now

        row = await _get(q, job_id)
        assert row.status == "queued"
        assert row.attempts == 1
        # deferred to the future (SQLite round-trips datetimes tz-naive, so we
        # don't compare against tz-aware _now(); the deferral itself is already
        # proven by processed == 1 — the requeued job was not re-claimed).
        assert row.run_after is not None

    asyncio.run(scenario())


def test_no_handler_fails_immediately():
    async def scenario():
        q = await _fresh_queue()
        job_id = await q.enqueue(kind="unregistered", payload={})
        assert await q.run_until_empty() == 1
        row = await _get(q, job_id)
        assert row.status == "failed"
        assert "no handler" in row.error

    asyncio.run(scenario())


def test_run_after_in_future_is_not_claimed():
    async def scenario():
        q = await _fresh_queue()
        q.register("noop", lambda payload: _async_none())
        await q.enqueue(kind="noop", run_after=_now() + timedelta(hours=1))
        # nothing due yet
        assert await q.run_until_empty() == 0

    asyncio.run(scenario())


def test_waiting_status_is_never_claimed():
    async def scenario():
        q = await _fresh_queue()
        ran = {"n": 0}

        async def handler(payload):
            ran["n"] += 1

        q.register("hitl", handler)
        job_id = await q.enqueue(kind="hitl", payload={})
        # Simulate a HITL pause: flip the row to the reserved non-terminal state.
        async with AsyncSession(q._engine) as s:
            row = await s.get(BackgroundJobRow, job_id)
            row.status = "waiting"
            s.add(row)
            await s.commit()

        assert await q.run_until_empty() == 0
        assert ran["n"] == 0
        row = await _get(q, job_id)
        assert row.status == "waiting"   # untouched

    asyncio.run(scenario())


def test_recover_orphans_requeues_running():
    async def scenario():
        q = await _fresh_queue()
        ran = {"n": 0}

        async def handler(payload):
            ran["n"] += 1
            return {"ok": True}

        q.register("work", handler)
        job_id = await q.enqueue(kind="work", payload={})
        # Simulate a crash mid-run: a row left 'running' with a stale worker.
        async with AsyncSession(q._engine) as s:
            row = await s.get(BackgroundJobRow, job_id)
            row.status = "running"
            row.worker_id = "dead-worker"
            s.add(row)
            await s.commit()

        # a running row is not claimable until recovered
        assert await q.run_until_empty() == 0

        recovered = await q.recover_orphans()
        assert recovered == 1
        mid = await _get(q, job_id)
        assert mid.status == "queued"
        assert mid.worker_id is None

        # now it runs to completion
        assert await q.run_until_empty() == 1
        assert ran["n"] == 1
        assert (await _get(q, job_id)).status == "succeeded"

    asyncio.run(scenario())


def test_concurrent_claim_single_winner():
    async def scenario():
        q = await _fresh_queue()
        await q.enqueue(kind="x", payload={})
        # Two claimers race for the one queued row; the SQLite claim lock must
        # serialize them so exactly one wins and the other sees nothing.
        a, b = await asyncio.gather(q.claim_one("wa"), q.claim_one("wb"))
        winners = [j for j in (a, b) if j is not None]
        assert len(winners) == 1
        assert winners[0].status == "running"

    asyncio.run(scenario())


def test_priority_and_fifo_ordering():
    async def scenario():
        q = await _fresh_queue()
        order = []

        async def handler(payload):
            order.append(payload["tag"])

        q.register("ord", handler)
        # lower priority value first; ties broken by created_at (insertion order)
        await q.enqueue(kind="ord", payload={"tag": "a"})
        await q.enqueue(kind="ord", payload={"tag": "b"})
        await q.enqueue(kind="ord", payload={"tag": "hi"})
        # bump the last one's priority to run first
        async with AsyncSession(q._engine) as s:
            rows = list((await s.exec(select(BackgroundJobRow))).all())
            hi = next(r for r in rows if r.payload["tag"] == "hi")
            hi.priority = -10
            s.add(hi)
            await s.commit()

        await q.run_until_empty()
        assert order[0] == "hi"
        assert order[1:] == ["a", "b"]

    asyncio.run(scenario())


async def _async_none():
    return None
