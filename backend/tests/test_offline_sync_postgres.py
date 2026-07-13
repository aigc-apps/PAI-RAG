"""Opt-in PostgreSQL queue ownership and lease tests."""

import asyncio
from datetime import timedelta
import os

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import create_all, make_engine
from app.jobs import JobQueue, _now
from app.models import BackgroundJobRow


pytestmark = pytest.mark.postgres
PG_URL = os.getenv("PAIRAG_TEST_POSTGRES_URL")


@pytest.mark.skipif(not PG_URL, reason="PAIRAG_TEST_POSTGRES_URL is not configured")
def test_skip_locked_claims_have_one_owner():
    async def scenario():
        engine = make_engine(PG_URL)
        await create_all(engine)
        queue = JobQueue(engine, concurrency=2)
        job_id = await queue.enqueue(kind="postgres-claim-test")
        try:
            first, second = await asyncio.gather(
                queue.claim_one("worker-a"), queue.claim_one("worker-b")
            )
            claimed = [job.id for job in (first, second) if job is not None]
            assert claimed == [job_id]
        finally:
            async with AsyncSession(engine) as session:
                row = await session.get(BackgroundJobRow, job_id)
                if row is not None:
                    await session.delete(row)
                    await session.commit()
            await engine.dispose()

    asyncio.run(scenario())


@pytest.mark.skipif(not PG_URL, reason="PAIRAG_TEST_POSTGRES_URL is not configured")
def test_only_expired_running_lease_is_reclaimed():
    async def scenario():
        engine = make_engine(PG_URL)
        await create_all(engine)
        queue = JobQueue(engine, lease_seconds=60)
        job_id = await queue.enqueue(kind="postgres-lease-test")
        try:
            assert (await queue.claim_one("owner")).id == job_id
            assert await queue.claim_one("other") is None
            async with AsyncSession(engine) as session:
                row = await session.get(BackgroundJobRow, job_id)
                row.lease_expires_at = _now() - timedelta(seconds=1)
                session.add(row)
                await session.commit()
            assert (await queue.claim_one("other")).id == job_id
        finally:
            async with AsyncSession(engine) as session:
                row = await session.get(BackgroundJobRow, job_id)
                if row is not None:
                    await session.delete(row)
                    await session.commit()
            await engine.dispose()

    asyncio.run(scenario())
