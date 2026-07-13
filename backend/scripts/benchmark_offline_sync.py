# ruff: noqa: E402
"""Enqueue and measure one production PostgreSQL/Elasticsearch sync run."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agent_config import load_agent_config  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.jobs import JobQueue  # noqa: E402
from app.models import (  # noqa: E402
    BackgroundJobRow,
    KnowledgeDataSourceRow,
    KnowledgeDocumentRow,
)


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


async def run(args) -> int:
    settings = get_settings()
    if not settings.db_url.startswith("postgresql"):
        raise RuntimeError("benchmark requires a PostgreSQL PAIRAG_DB_URL")
    config = load_agent_config(settings.config_path)
    if config.knowledgebase.vectordb.engine != "elasticsearch":
        raise RuntimeError("benchmark requires knowledgebase.vectordb.engine=elasticsearch")

    engine = make_engine(settings.db_url)
    queue = JobQueue(engine, default_max_attempts=settings.job_max_attempts)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        stmt = (
            select(KnowledgeDataSourceRow)
            .where(
                KnowledgeDataSourceRow.id == args.datasource_id,
                KnowledgeDataSourceRow.kb_id == args.kb_id,
                KnowledgeDataSourceRow.deleted_at.is_(None),
            )
            .with_for_update()
        )
        ds = (await session.exec(stmt)).first()
        if ds is None:
            raise RuntimeError("data source not found")
        if ds.active_job_id:
            raise RuntimeError(f"data source already has active job {ds.active_job_id}")
        job = queue.build_job(
            kind="kb_sync",
            kb_id=args.kb_id,
            created_by=ds.created_by,
            payload={
                "kb_id": args.kb_id,
                "ds_id": args.datasource_id,
                "user_id": ds.created_by,
                "user_email": None,
                "user_role": "user",
            },
        )
        ds.active_job_id = job.id
        ds.status = "syncing"
        session.add(ds)
        session.add(job)
        await session.commit()

    started = asyncio.get_running_loop().time()
    first_fetched = None
    first_indexed = None
    previous = None
    terminal = {"succeeded", "partial", "failed", "cancelled"}
    row = None
    while asyncio.get_running_loop().time() - started <= args.timeout_seconds:
        async with AsyncSession(engine) as session:
            row = await session.get(BackgroundJobRow, job.id)
        progress = dict(row.progress or {})
        elapsed = asyncio.get_running_loop().time() - started
        if progress.get("fetched", 0) and first_fetched is None:
            first_fetched = elapsed
        if progress.get("indexed", 0) and first_indexed is None:
            first_indexed = elapsed
        snapshot = {"job_id": job.id, "status": row.status, "elapsed_seconds": round(elapsed, 2), **progress}
        if snapshot != previous:
            emit(snapshot)
            previous = snapshot
        if row.status in terminal:
            break
        await asyncio.sleep(1)

    duration = asyncio.get_running_loop().time() - started
    async with AsyncSession(engine) as session:
        pending = (
            await session.exec(
                select(func.count()).select_from(KnowledgeDocumentRow).where(
                    KnowledgeDocumentRow.kb_id == args.kb_id,
                    KnowledgeDocumentRow.search_index_status.in_(["pending", "delete_pending"]),
                )
            )
        ).one()
        expired = (
            await session.exec(
                select(func.count()).select_from(BackgroundJobRow).where(
                    BackgroundJobRow.status == "running",
                    BackgroundJobRow.lease_expires_at < datetime.now(timezone.utc),
                )
            )
        ).one()
    progress = dict((row.progress if row else {}) or {})
    final = {
        "status": row.status if row else "timeout",
        "duration_seconds": round(duration, 2),
        "first_fetched_seconds": None if first_fetched is None else round(first_fetched, 2),
        "first_indexed_seconds": None if first_indexed is None else round(first_indexed, 2),
        "pending_documents": int(pending),
        "expired_running_jobs": int(expired),
        **progress,
    }
    emit(final)
    await engine.dispose()
    accepted = (
        row is not None
        and row.status == "succeeded"
        and duration <= args.timeout_seconds
        and (first_fetched or float("inf")) <= 15
        and (first_indexed or float("inf")) <= 30
        and progress.get("max_fetched_buffer", 0) <= settings.sync_fetch_queue_size
        and int(pending) == 0
        and int(expired) == 0
    )
    return 0 if accepted else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--datasource-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit({"status": "error", "error": str(exc)})
        raise SystemExit(2) from exc
