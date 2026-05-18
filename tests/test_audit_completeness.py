"""End-to-end audit pipeline test.

Pushes ``AuditEvent``s through :class:`RedisAuditPublisher`, drains them via
:class:`AuditConsumer.run_once`, and verifies:

1. Every category produced by the SDK runner (``llm_chunk``, ``tool_call``,
   ``tool_result``, ``hitl_pause``, ``hitl_resume``, ``hitl_auto_continue``,
   ``run_complete``, ``run_failed``) has a matching row in ``audit_events``.
2. Sensitive payload text (matching ``SECRET_PATTERNS`` from
   ``agent_loop.redact_sensitive_text``) is redacted before persistence.
3. When Redis is unavailable, the publisher falls back to the synchronous
   :class:`AuditStore` so audit rows are never lost.
"""
import os
import tempfile
import unittest

from session_store import SessionStore
from backend.audit.consumer import AuditConsumer
from backend.audit.publisher import RedisAuditPublisher
from backend.audit.store import (
    AuditEvent, AuditStore,
    CATEGORY_HITL_AUTO_CONTINUE, CATEGORY_HITL_PAUSE, CATEGORY_HITL_RESUME,
    CATEGORY_LLM_CHUNK, CATEGORY_RUN_COMPLETE, CATEGORY_RUN_FAILED,
    CATEGORY_TOOL_CALL, CATEGORY_TOOL_RESULT,
)


# ─── tiny fake of the redis-py methods AuditConsumer + Publisher use ────────

class FakeRedis:
    def __init__(self):
        self._streams: dict[str, list[tuple[str, dict]]] = {}
        self._counter = 0

    def xadd(self, key, fields):
        self._counter += 1
        entry_id = f'{self._counter}-0'
        self._streams.setdefault(key, []).append((entry_id, dict(fields)))
        return entry_id

    def xrange(self, key, min='-', max='+', count=None):
        rows = list(self._streams.get(key, []))
        return rows[:count] if count else rows

    def xdel(self, key, *ids):
        keep = [(i, f) for i, f in self._streams.get(key, []) if i not in set(ids)]
        self._streams[key] = keep
        return len(ids)

    def scan(self, cursor=0, match=None, count=200):
        prefix = (match or '').rstrip('*')
        keys = [k for k in self._streams.keys() if k.startswith(prefix)]
        return 0, keys  # cursor=0 → "we're done"

    def expire(self, key, ttl):
        return True

    def ping(self):
        return True


class _BrokenRedis(FakeRedis):
    """Pretend Redis is reachable for ping() but xadd fails — exercises the
    publisher's synchronous-fallback path.
    """
    def xadd(self, key, fields):
        raise ConnectionError('redis temporarily unavailable')


# ─── tests ──────────────────────────────────────────────────────────────────

class AuditCompletenessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session_store = SessionStore(os.path.join(self._tmp.name, 'sessions'))
        self.audit_store = AuditStore(connect=self.session_store._connect)
        self.fake = FakeRedis()
        self.publisher = RedisAuditPublisher(
            redis_client=self.fake, fallback_store=self.audit_store,
        )
        self.consumer = AuditConsumer(redis_client=self.fake, store=self.audit_store)

    def tearDown(self):
        self._tmp.cleanup()

    def _emit(self, category, payload, *, audit_log_id='audit_1', run_id='run_1', session_id='sess_1'):
        self.publisher.append(AuditEvent(
            audit_log_id=audit_log_id, run_id=run_id, session_id=session_id,
            response_id='resp_1', category=category, payload=payload,
        ))

    def test_full_run_event_categories_drain_to_sqlite(self):
        # Mirror the categories the SDK runner emits over a normal run + HITL.
        run_categories = [
            CATEGORY_LLM_CHUNK,
            CATEGORY_TOOL_CALL,
            CATEGORY_TOOL_RESULT,
            CATEGORY_HITL_PAUSE,
            CATEGORY_HITL_RESUME,
            CATEGORY_HITL_AUTO_CONTINUE,
            CATEGORY_RUN_COMPLETE,
        ]
        for cat in run_categories:
            self._emit(cat, {'note': cat})

        # Nothing in SQLite yet — events are sitting in the Redis stream.
        self.assertEqual(self.audit_store.query('audit_1'), [])

        drained = self.consumer.run_once()
        self.assertEqual(drained, len(run_categories))

        rows = self.audit_store.query('audit_1')
        seen = {row['category'] for row in rows}
        self.assertEqual(seen, set(run_categories))

    def test_failed_run_emits_run_failed_row(self):
        # A run that crashed mid-stream should leave a single ``run_failed``
        # row alongside whatever earlier categories made it through.
        self._emit(CATEGORY_LLM_CHUNK, {'delta': 'hi'})
        self._emit(CATEGORY_RUN_FAILED, {'error': 'boom'})
        self.consumer.run_once()
        rows = self.audit_store.query('audit_1')
        cats = [r['category'] for r in rows]
        self.assertIn(CATEGORY_LLM_CHUNK, cats)
        self.assertIn(CATEGORY_RUN_FAILED, cats)

    def test_secret_patterns_get_redacted_before_persistence(self):
        # ``sk-...`` 20+ chars and api_key=... patterns are both in
        # SECRET_PATTERNS; both must vanish before hitting SQLite.
        self._emit(CATEGORY_TOOL_CALL, {
            'name': 'http_get',
            'args': {
                'token': 'sk-abcdef0123456789ABCDEF01',
                'header': 'api_key=topsecret123456',
            },
            'note': 'leading text sk-zzzzzzzzzzzzzzzzzzzz trailing',
        })
        self.consumer.run_once()
        rows = self.audit_store.query('audit_1')
        self.assertEqual(len(rows), 1)
        payload = rows[0]['payload']
        # The non-sensitive scaffolding survives unchanged.
        self.assertEqual(payload['name'], 'http_get')
        # The args dict still exists, but the sensitive value either was
        # redacted in place or replaced wholesale by ``[REDACTED]``.
        flat = repr(payload).lower()
        self.assertNotIn('sk-abcdef0123456789abcdef01', flat)
        self.assertNotIn('topsecret123456', flat)

    def test_fallback_writes_directly_when_redis_xadd_fails(self):
        broken = _BrokenRedis()
        publisher = RedisAuditPublisher(redis_client=broken, fallback_store=self.audit_store)
        publisher.append(AuditEvent(
            audit_log_id='audit_2', run_id='run_2', session_id='sess_2',
            response_id=None, category=CATEGORY_LLM_CHUNK,
            payload={'delta': 'still recorded'},
        ))
        # No drain needed — publisher wrote straight to SQLite.
        rows = self.audit_store.query('audit_2')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['payload']['delta'], 'still recorded')

    def test_unknown_category_raises_before_redis_xadd(self):
        # Catch typos in the runner before they cause silent gaps in the audit
        # log. The publisher reuses ``ALL_CATEGORIES`` from the store.
        with self.assertRaises(ValueError):
            self.publisher.append(AuditEvent(
                audit_log_id='audit_3', run_id='r', session_id='s',
                response_id=None, category='not_a_real_category',
                payload={},
            ))


if __name__ == '__main__':
    unittest.main()
