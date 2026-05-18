"""``agent_run_states`` roundtrip + GC tests.

Covers the SQLite layer that owns the SDK ``RunState`` blob: byte-equivalent
storage across an upsert→get cycle, expiry handling, and the GC sweep that
the server-side daemon calls every 5 minutes.

The SDK's ``RunState.to_string()/from_string()`` round-trip itself is not the
contract this layer owns — we only need to prove the blob lands and comes
back unchanged so that ``Runner.run_streamed(state, ...)`` resumes with the
exact bytes the previous run wrote.
"""
import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone

from session_store import SERVER_USER_ID, SessionStore
from backend.agents_sdk.lifecycle import (
    RUN_STATE_COMPLETED,
    RUN_STATE_EXPIRED,
    RUN_STATE_REQUIRES_ACTION,
    RUN_STATE_RUNNING,
)
from backend.agents_sdk.run_state_store import RunStateStore


SAMPLE_BLOB = (
    '{"current_turn": 3, "current_agent": {"name": "pai-rag"}, '
    '"generated_items": [{"type": "tool_call_item", "raw_item": '
    '{"call_id": "call_42", "name": "file_read", "arguments": "{\\"path\\":\\"x\\"}"}}], '
    '"trace": null, "max_turns": 40, "noop_coalesced_assistant_text": "", '
    '"context_wrapper": {"context": null, "usage": {"requests": 1, "input_tokens": 100}}, '
    '"_pending_tool_runs": [], "_input_items": []}'
)


class RunStateRoundtripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session_store = SessionStore(os.path.join(self._tmp.name, 'sessions'))
        self.store = RunStateStore(connect=self.session_store._connect)

    def tearDown(self):
        self._tmp.cleanup()

    def _upsert(self, **overrides):
        kwargs = dict(
            id='resp_1', session_id='sess_1', run_id='run_1',
            response_id='resp_1', user_id=SERVER_USER_ID, model='qwen-test',
            status=RUN_STATE_RUNNING, run_state_blob=SAMPLE_BLOB,
            pending_interruption_json=None, last_event_id=None,
            audit_log_id='audit_1',
        )
        kwargs.update(overrides)
        self.store.upsert(**kwargs)

    def test_blob_roundtrip_byte_equivalent(self):
        # Upsert a non-trivial JSON blob, sleep across a clock boundary, read
        # it back; resume requires byte-for-byte equality (RunState.from_string
        # is sensitive to any normalization).
        self._upsert(run_state_blob=SAMPLE_BLOB)
        time.sleep(1.1)
        row = self.store.get('resp_1')
        self.assertIsNotNone(row)
        self.assertEqual(row['run_state_blob'], SAMPLE_BLOB)
        # Sanity: the unicode-rich fields survive too.
        unicode_blob = SAMPLE_BLOB + ' "中文":"你好"'
        self._upsert(run_state_blob=unicode_blob)
        row2 = self.store.get('resp_1')
        self.assertEqual(row2['run_state_blob'], unicode_blob)

    def test_upsert_overwrites_blob_and_status(self):
        self._upsert(run_state_blob='v1', status=RUN_STATE_RUNNING)
        self._upsert(run_state_blob='v2', status=RUN_STATE_REQUIRES_ACTION,
                     pending_interruption_json='[{"call_id":"c1"}]')
        row = self.store.get('resp_1')
        self.assertEqual(row['run_state_blob'], 'v2')
        self.assertEqual(row['status'], RUN_STATE_REQUIRES_ACTION)
        self.assertEqual(row['pending_interruption_json'], '[{"call_id":"c1"}]')

    def test_expires_at_is_set_relative_to_now(self):
        # Default TTL is 7d; the store should populate ``expires_at`` ~7d in
        # the future. We allow a generous skew to avoid CI flakiness.
        self._upsert()
        row = self.store.get('resp_1')
        expires = datetime.fromisoformat(row['expires_at'])
        delta = expires - datetime.now(timezone.utc)
        self.assertGreater(delta, timedelta(days=6, hours=23))
        self.assertLess(delta, timedelta(days=7, hours=1))

    def test_custom_ttl_seconds_respected(self):
        self._upsert(ttl_seconds=60)
        row = self.store.get('resp_1')
        expires = datetime.fromisoformat(row['expires_at'])
        delta = expires - datetime.now(timezone.utc)
        self.assertLess(delta, timedelta(minutes=2))

    def test_gc_expired_marks_overdue_runs_expired(self):
        # Insert an active row, then rewrite ``expires_at`` to the past so GC
        # has something to flip without us having to wait 7 days.
        self._upsert(status=RUN_STATE_REQUIRES_ACTION)
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with self.session_store._connect() as conn:
            conn.execute(
                'UPDATE agent_run_states SET expires_at = ? WHERE id = ?',
                (past, 'resp_1'),
            )

        affected = self.store.gc_expired()
        self.assertGreaterEqual(affected, 1)
        row = self.store.get('resp_1')
        self.assertEqual(row['status'], RUN_STATE_EXPIRED)

    def test_gc_expired_deletes_completed_rows_after_completed_ttl(self):
        # Completed rows past completed_ttl are hard-deleted (not just flipped).
        self._upsert(status=RUN_STATE_COMPLETED)
        long_ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        with self.session_store._connect() as conn:
            conn.execute(
                'UPDATE agent_run_states SET last_active_at = ? WHERE id = ?',
                (long_ago, 'resp_1'),
            )

        self.store.gc_expired(completed_ttl_seconds=24 * 3600)
        self.assertIsNone(self.store.get('resp_1'))

    def test_gc_does_not_touch_fresh_rows(self):
        self._upsert(status=RUN_STATE_RUNNING)
        affected = self.store.gc_expired()
        self.assertEqual(affected, 0)
        row = self.store.get('resp_1')
        self.assertEqual(row['status'], RUN_STATE_RUNNING)

    def test_resume_after_expiry_surfaces_lookup_error(self):
        # The runner reads the row via ``state_store.get`` and refuses to
        # resume anything not in ``requires_action`` — the GC sweep flips the
        # row to ``expired`` precisely so this gate trips.
        self._upsert(status=RUN_STATE_REQUIRES_ACTION)
        past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with self.session_store._connect() as conn:
            conn.execute(
                'UPDATE agent_run_states SET expires_at = ? WHERE id = ?',
                (past, 'resp_1'),
            )
        self.store.gc_expired()

        row = self.store.get('resp_1')
        self.assertEqual(row['status'], RUN_STATE_EXPIRED)
        # This is the exact branch ``runner._resume_run`` enters before
        # ``raise LookupError(f'run not resumable: status={row["status"]}')``.
        self.assertNotEqual(row['status'], RUN_STATE_REQUIRES_ACTION)


if __name__ == '__main__':
    unittest.main()
