import threading
import unittest
from types import SimpleNamespace

import backend.agent_service as agent_service
from backend.agent_service import AgentSession, SESSION_RUNNING
from session_store import SERVER_USER_ID


class FakeStore:
    def __init__(self):
        self.finished = []

    def finish_run(self, session_id, user_id, run_id, status, error=""):
        self.finished.append({
            "session_id": session_id,
            "user_id": user_id,
            "run_id": run_id,
            "status": status,
            "error": error,
        })


class AgentSessionRunRecordTests(unittest.TestCase):
    def test_thread_run_loop_finishes_run_record(self):
        original_runner = agent_service.agent_runner_loop
        original_archive = agent_service.archive_session
        original_prompt = agent_service.build_system_prompt
        store = FakeStore()
        saves = []

        def fake_runner(**kwargs):
            return {"result": "NO_TOOL_CALL"}

        try:
            agent_service.agent_runner_loop = fake_runner
            agent_service.archive_session = lambda *args, **kwargs: None
            agent_service.build_system_prompt = lambda user_id=SERVER_USER_ID: "system"

            session = AgentSession.__new__(AgentSession)
            session.sid = "session-1"
            session.user_id = SERVER_USER_ID
            session.client = object()
            session.handler = object()
            session.status = SESSION_RUNNING
            session.active_run_id = "run-1"
            session.worker = object()
            session._lock = threading.RLock()
            session.turn_done_evt = threading.Event()
            session.service = SimpleNamespace(store=store)
            session.save = lambda: saves.append(True)

            session._run_loop("user input", "task text", mode="events", run_id="run-1")

            self.assertTrue(session.turn_done_evt.is_set())
            self.assertEqual(store.finished, [{
                "session_id": "session-1",
                "user_id": SERVER_USER_ID,
                "run_id": "run-1",
                "status": "completed",
                "error": "",
            }])
            self.assertEqual(session.active_run_id, "")
            self.assertTrue(saves)
        finally:
            agent_service.agent_runner_loop = original_runner
            agent_service.archive_session = original_archive
            agent_service.build_system_prompt = original_prompt

    def test_thread_run_loop_marks_max_turns_as_failed(self):
        original_runner = agent_service.agent_runner_loop
        original_archive = agent_service.archive_session
        original_prompt = agent_service.build_system_prompt
        store = FakeStore()

        def fake_runner(**kwargs):
            return {"result": "MAX_TURNS_EXCEEDED", "data": "partial report"}

        try:
            agent_service.agent_runner_loop = fake_runner
            agent_service.archive_session = lambda *args, **kwargs: None
            agent_service.build_system_prompt = lambda user_id=SERVER_USER_ID: "system"

            session = AgentSession.__new__(AgentSession)
            session.sid = "session-1"
            session.user_id = SERVER_USER_ID
            session.client = object()
            session.handler = object()
            session.status = SESSION_RUNNING
            session.active_run_id = "run-1"
            session.worker = object()
            session._lock = threading.RLock()
            session.turn_done_evt = threading.Event()
            session.service = SimpleNamespace(store=store)
            session.save = lambda: None

            session._run_loop("user input", "task text", mode="events", run_id="run-1")

            self.assertEqual(store.finished[0]["status"], "failed")
            self.assertEqual(store.finished[0]["error"], "MAX_TURNS_EXCEEDED")
        finally:
            agent_service.agent_runner_loop = original_runner
            agent_service.archive_session = original_archive
            agent_service.build_system_prompt = original_prompt


if __name__ == "__main__":
    unittest.main()
