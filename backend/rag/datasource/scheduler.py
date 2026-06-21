"""DB-driven scheduler for data source syncs.

Celery beat is static, so a single periodic task (``datasource_sync_dispatch``)
sweeps the data sources and enqueues a sync for every one that is due, then
advances its ``next_sync_at``.

``sync_schedule`` is an interval in seconds (e.g. "3600"). If ``croniter`` is
installed, a cron expression (e.g. "0 3 * * *") is also accepted.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List

from loguru import logger
from sqlmodel import select
from sqlalchemy import or_

from db.db_context import create_db_session
from db.models.knowledgebase.datasource import DataSourceEntity
from common.knowledgebase.types import DataSourceStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def compute_next_sync_at(schedule: Optional[str], base: datetime) -> Optional[datetime]:
    """Compute the next run time from a schedule string. None if unparsable."""
    if not schedule:
        return None
    schedule = schedule.strip()
    # interval in seconds
    try:
        seconds = int(schedule)
        if seconds <= 0:
            return None
        return base + timedelta(seconds=seconds)
    except ValueError:
        pass
    # cron expression (optional dependency)
    try:
        from croniter import croniter
        if croniter.is_valid(schedule):
            return croniter(schedule, base).get_next(datetime)
    except ImportError:
        logger.warning(
            f"Cron schedule '{schedule}' requires croniter (not installed); skipping."
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Invalid sync_schedule '{schedule}': {e}")
    return None


def _default_enqueue(datasource_id: str, tenant_id: str) -> None:
    from app.worker import sync_datasource
    sync_datasource.delay(
        datasource_id=datasource_id, tenant_id=tenant_id, trigger="scheduled", triggered_by=None,
    )


async def dispatch_due_datasources(enqueue_fn=None) -> List[str]:
    """Enqueue a sync for every enabled, scheduled, due data source (cross-tenant).

    Returns the list of dispatched data source ids.
    """
    enqueue_fn = enqueue_fn or _default_enqueue
    now = _utcnow()
    dispatched: List[str] = []

    async with create_db_session() as session:
        stmt = select(DataSourceEntity).where(
            DataSourceEntity.enabled == True,  # noqa: E712
            DataSourceEntity.sync_schedule.is_not(None),
            DataSourceEntity.sync_schedule != "",
            DataSourceEntity.status != DataSourceStatus.syncing,  # avoid overlap
            or_(
                DataSourceEntity.next_sync_at.is_(None),
                DataSourceEntity.next_sync_at <= now,
            ),
        )
        due = (await session.exec(stmt)).all()

        for ds in due:
            next_at = compute_next_sync_at(ds.sync_schedule, now)
            if next_at is None:
                # unparsable schedule: park it a day out so we don't hot-loop on logs
                ds.next_sync_at = now + timedelta(days=1)
                ds.updated_at = now
                session.add(ds)
                continue
            try:
                enqueue_fn(ds.id, ds.tenant_id)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[datasource-dispatch] enqueue failed for {ds.id}: {e}")
                continue
            ds.next_sync_at = next_at
            ds.updated_at = now
            session.add(ds)
            dispatched.append(ds.id)

        await session.commit()

    if dispatched:
        logger.info(f"[datasource-dispatch] dispatched {len(dispatched)} due data source(s).")
    return dispatched
