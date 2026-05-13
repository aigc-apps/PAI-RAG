import queue
import threading
import unittest

from fastapi.testclient import TestClient

import backend.server as server
from agent_events import agent_message_chunk, done, stop_reason
from session_store import SERVER_USER_ID


class FakeSession:
    def __init__(self, sid, run_id):
        self.sid = sid
        self._run_id = run_id
        self.user_id = SERVER_USER_ID
        self.display_q: queue.Queue = queue.Queue()
        self.turn_done_evt = threading.Event()
        self.cancel_evt = threading.Event()

    def run_or_answer(self, text, mode="events"):
        self.display_q.put({"event": agent_message_chunk("hi back")})
        self.display_q.put({"event": done(stop_reason({"result": "NO_TOOL_CALL"}))})
        self.turn_done_evt.set()
        return self._run_id


class FakeService:
    def __init__(self):
        self.sessions: dict[str, FakeSession] = {}

    def create_session(self, user_id=SERVER_USER_ID, cwd=None):
        sid = f"sess-{len(self.sessions) + 1}"
        sess = FakeSession(sid, f"run-{len(self.sessions) + 1}")
        self.sessions[sid] = sess
        return sess

    def get_session(self, sid=None, user_id=SERVER_USER_ID, cwd=None):
        if sid and sid not in self.sessions:
            self.sessions[sid] = FakeSession(sid, f"run-{len(self.sessions) + 1}")
        if not sid:
            return self.create_session(user_id=user_id, cwd=cwd)
        return self.sessions[sid]

    def load_session(self, sid, user_id=SERVER_USER_ID):
        return self.sessions.get(sid)

    def cancel_session(self, sid, user_id=SERVER_USER_ID):
        sess = self.sessions.get(sid)
        if sess:
            sess.cancel_evt.set()
        return sess is not None


class RunsStreamTests(unittest.TestCase):
    def setUp(self):
        self.original_service = server.service
        self.original_celery_service = server.celery_service
        self.original_thread_runs = dict(server.THREAD_RUNS)
        server.celery_service = None
        server.THREAD_RUNS.clear()
        self.fake_service = FakeService()
        server.service = self.fake_service
        self.client = TestClient(server.app)

    def tearDown(self):
        server.service = self.original_service
        server.celery_service = self.original_celery_service
        server.THREAD_RUNS.clear()
        server.THREAD_RUNS.update(self.original_thread_runs)

    def test_create_run_without_stream_returns_202_json_with_cursor(self):
        response = self.client.post("/v1/runs", json={"input": "hi"})

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["object"], "agent.run")
        self.assertEqual(body["status"], "started")
        self.assertIn("cursor", body)
        self.assertNotIn("stream_from", body)

    def test_create_run_with_stream_true_returns_sse(self):
        response = self.client.post("/v1/runs", json={"input": "hi", "stream": True})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream"),
            response.headers["content-type"],
        )
        self.assertIn("X-Session-Id", response.headers)
        self.assertIn("X-Run-Id", response.headers)
        body_text = response.text
        self.assertIn("data:", body_text)
        self.assertIn("run.completed", body_text)


if __name__ == "__main__":
    unittest.main()
