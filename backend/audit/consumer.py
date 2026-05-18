"""Redis-stream → SQLite audit drainer.

Hot path (the SDK runner streaming loop) pushes audit events into a Redis
stream ``audit:{audit_log_id}``; this consumer batches them into the
``audit_events`` table so the streaming RTT is unaffected by SQLite writes.

Phase 1 scaffold: the daemon thread starts cleanly but is not yet hooked
into the FastAPI startup. Phase 2 hooks it in :mod:`backend.server`.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Iterable

from backend.audit.store import AuditEvent, AuditStore

DEFAULT_BATCH_SIZE = 100
DEFAULT_POLL_SECONDS = 0.25
STREAM_PREFIX = 'audit:'


class AuditConsumer:
    """Daemon thread that drains audit events from a Redis stream.

    Tests can inject a fake ``redis_client`` (anything with ``xreadgroup`` /
    ``xack`` semantics matching ``redis.Redis``) and call ``run_once`` to
    avoid spinning up a real thread.
    """

    def __init__(
        self,
        *,
        redis_client,
        store: AuditStore,
        consumer_group: str = 'audit-drain',
        consumer_name: str = 'audit-drain-1',
        batch_size: int = DEFAULT_BATCH_SIZE,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ):
        self._redis = redis_client
        self._store = store
        self._group = consumer_group
        self._consumer = consumer_name
        self._batch_size = batch_size
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name='audit-consumer', daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                drained = self.run_once()
            except Exception as e:  # pragma: no cover — daemon must not die
                print(f'[AuditConsumer] error: {e}', flush=True)
                drained = 0
            if drained == 0:
                self._stop.wait(self._poll_seconds)

    def run_once(self) -> int:
        """Drain one batch from any active audit stream. Returns count.

        Streams are named ``audit:{audit_log_id}``. We discover them with
        ``SCAN`` because runs come and go; for steady-state workloads a
        single shared stream might be cheaper — leave that for Phase 5.
        """
        total = 0
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match=f'{STREAM_PREFIX}*', count=200)
            for raw in keys or []:
                key = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
                total += self._drain_stream(key)
            if cursor == 0:
                break
        return total

    def _drain_stream(self, stream_key: str) -> int:
        events: list[AuditEvent] = []
        try:
            entries = self._redis.xrange(stream_key, min='-', max='+', count=self._batch_size)
        except Exception:
            return 0
        if not entries:
            return 0
        last_id = None
        for entry_id, fields in entries:
            last_id = entry_id
            try:
                payload = {k.decode() if isinstance(k, (bytes, bytearray)) else k:
                           (v.decode() if isinstance(v, (bytes, bytearray)) else v)
                           for k, v in fields.items()}
                events.append(AuditEvent(
                    audit_log_id=payload.get('audit_log_id', ''),
                    run_id=payload.get('run_id', ''),
                    session_id=payload.get('session_id', ''),
                    response_id=payload.get('response_id') or None,
                    category=payload.get('category', ''),
                    payload=json.loads(payload.get('payload') or '{}'),
                    ts_ms=int(payload.get('ts_ms')) if payload.get('ts_ms') else None,
                ))
            except Exception:
                continue
        written = self._store.append_many(events)
        if last_id is not None:
            try:
                self._redis.xdel(stream_key, last_id)
                # Trim everything we processed; xdel single id keeps siblings.
                processed_ids = [e[0] for e in entries]
                self._redis.xdel(stream_key, *processed_ids)
            except Exception:
                pass
        return written
