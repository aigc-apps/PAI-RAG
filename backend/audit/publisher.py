"""Redis-stream publisher used by the SDK runner.

The runner pushes ``AuditEvent``s into ``audit:{audit_log_id}`` so the
streaming hot path is not blocked on SQLite writes. A daemon
:class:`backend.audit.consumer.AuditConsumer` drains the stream into the
``audit_events`` table.

Falls back to direct ``AuditStore.append`` when Redis is unreachable so a
broken cache does not silently lose audit rows.
"""
from __future__ import annotations

import json
import time

from backend.audit.store import AuditEvent, AuditStore, _sanitize, ALL_CATEGORIES

STREAM_PREFIX = 'audit:'
DEFAULT_STREAM_TTL_SECONDS = 24 * 60 * 60


class RedisAuditPublisher:
    """``append``-compatible publisher that writes to a Redis stream first
    and to ``AuditStore`` only if Redis is unavailable.
    """

    def __init__(
        self,
        *,
        redis_client,
        fallback_store: AuditStore,
        stream_ttl_seconds: int = DEFAULT_STREAM_TTL_SECONDS,
    ):
        self._redis = redis_client
        self._fallback = fallback_store
        self._ttl = int(stream_ttl_seconds or 0)

    def append(self, event: AuditEvent) -> None:
        if event.category not in ALL_CATEGORIES:
            raise ValueError(f'unknown audit category: {event.category!r}')
        ts_ms = event.ts_ms if event.ts_ms is not None else int(time.time() * 1000)
        payload_json = json.dumps(_sanitize(event.payload), ensure_ascii=False, default=str)
        fields = {
            'audit_log_id': event.audit_log_id,
            'run_id': event.run_id,
            'session_id': event.session_id,
            'response_id': event.response_id or '',
            'ts_ms': str(ts_ms),
            'category': event.category,
            'payload': payload_json,
        }
        key = f'{STREAM_PREFIX}{event.audit_log_id}'
        try:
            self._redis.xadd(key, fields)
            if self._ttl > 0:
                self._redis.expire(key, self._ttl)
            return
        except Exception:
            # Redis hiccup — write straight to SQLite so the audit row is
            # never lost. The consumer will pick up future events as soon as
            # Redis is healthy again.
            self._fallback.append(event)

    def append_many(self, events) -> int:
        count = 0
        for ev in events:
            self.append(ev)
            count += 1
        return count
