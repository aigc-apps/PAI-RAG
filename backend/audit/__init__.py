"""Append-only audit log for SDK-runner runs.

Three pieces:

- :class:`AuditStore` — synchronous SQLite writer (table
  ``audit_events`` defined in :mod:`backend.session_store_sqlite`).
- :class:`RedisAuditPublisher` — used by the runner hot path to push events
  into ``audit:{audit_log_id}`` Redis streams; falls back to direct
  ``AuditStore.append`` when Redis is unreachable.
- :class:`AuditConsumer` — daemon thread that drains the streams into the
  store in batches.
"""
from backend.audit.consumer import AuditConsumer
from backend.audit.publisher import RedisAuditPublisher
from backend.audit.store import AuditStore

__all__ = ['AuditStore', 'RedisAuditPublisher', 'AuditConsumer']
