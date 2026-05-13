"""Layer 4 verification: token-chunk events coalesce into a small number of
SQLite writes within the configured `SNAPSHOT_FLUSH_INTERVAL_MS` window,
while logical-step events (tool call, ask_user, done) still flush eagerly.
"""
import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from backend.worker import WorkerSession  # noqa: F401
    _IMPORT_ERROR = None
except ImportError as e:
    _IMPORT_ERROR = str(e)


def _build_session(interval_ms):
    from backend.worker import WorkerSession

    session = WorkerSession.__new__(WorkerSession)
    session.run_id = 'r1'
    session.sid = 's1'
    session.user_id = 'u1'
    session.status = 'running'
    session.active_run_id = 'r1'
    session.workspace_path = '/tmp/ws'
    session.ui_msgs = [{'role': 'user', 'content': 'q'}]
    session.client = MagicMock()
    session.client.history = []
    session.handler = MagicMock()
    session.handler.history_info = []
    session.handler.working = {}
    session.handler.todos = []
    session.store = MagicMock()
    session._save_lock = threading.Lock()
    session._save_dirty = False
    session._save_last_flush_ms = 0.0
    session._save_timer = None
    session._save_interval_ms = interval_ms
    return session


@unittest.skipIf(_IMPORT_ERROR is not None,
                 f'backend.worker not importable (likely missing celery): {_IMPORT_ERROR}')
class WorkerSnapshotDebounceTests(unittest.TestCase):
    def test_token_chunks_coalesce(self):
        session = _build_session(interval_ms=200)
        for i in range(50):
            event = {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': f't{i}'}}
            session._record_assistant_event(event)
        # First call flushes immediately (cold start), rest debounce until timer.
        time.sleep(0.5)
        self.assertGreaterEqual(session.store.save_run_snapshot.call_count, 1)
        self.assertLessEqual(session.store.save_run_snapshot.call_count, 4,
                             f'too many writes: {session.store.save_run_snapshot.call_count}')

    def test_logical_step_event_flushes_immediately(self):
        session = _build_session(interval_ms=10_000)
        session._record_assistant_event(
            {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': 'hi'}}
        )
        session._record_assistant_event({'sessionUpdate': 'tool_call', 'toolCallId': 'tc1'})
        # 1 cold-start chunk flush + 1 forced flush from tool_call = 2.
        # Pending timer should be cancelled.
        self.assertEqual(session.store.save_run_snapshot.call_count, 2)
        self.assertIsNone(session._save_timer)


if __name__ == '__main__':
    unittest.main()
