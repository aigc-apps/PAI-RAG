import unittest
from types import SimpleNamespace

import backend.worker as worker
from backend.agent_service import SESSION_RUNNING
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


class WorkerTextCompletionTests(unittest.TestCase):
    def test_text_mode_emits_done_after_normal_completion(self):
        original_runner = worker.agent_runner_loop
        original_archive = worker.archive_session
        original_prompt = worker.build_system_prompt
        original_schema = worker.json_tools_schema
        original_review = worker.schedule_background_memory_review
        original_long_term = worker.long_term_memory_enabled

        events = []
        store = FakeStore()

        def fake_runner(**kwargs):
            kwargs["on_chunk"]("hello")
            return {"result": "NO_TOOL_CALL", "data": "hello"}

        try:
            worker.agent_runner_loop = fake_runner
            worker.archive_session = lambda *args, **kwargs: None
            worker.build_system_prompt = lambda user_id=SERVER_USER_ID: "system"
            worker.json_tools_schema = lambda: []
            worker.schedule_background_memory_review = lambda **kwargs: None
            worker.long_term_memory_enabled = lambda user_id: True

            session = worker.WorkerSession.__new__(worker.WorkerSession)
            session.run_id = "run-1"
            session.sid = "session-1"
            session.user_id = SERVER_USER_ID
            session.task_text = "hello"
            session.mode = "text"
            session.client = SimpleNamespace(history=[])
            session.user_input = "hello"
            session.handler = SimpleNamespace(working={})
            session.memory_scope = SimpleNamespace(root="/memory/root")
            session.status = SESSION_RUNNING
            session.active_run_id = "run-1"
            session.store = store
            session.save = lambda: None
            session.emit_event = lambda event, check_cancel=True: events.append(event)

            session.run()

            self.assertEqual(events[0]["sessionUpdate"], "agent_message_chunk")
            self.assertEqual(events[0]["content"]["text"], "hello")
            self.assertEqual(events[-1]["sessionUpdate"], "done")
            self.assertEqual(events[-1]["stopReason"], "end_turn")
            self.assertEqual(store.finished[0]["status"], "completed")
        finally:
            worker.agent_runner_loop = original_runner
            worker.archive_session = original_archive
            worker.build_system_prompt = original_prompt
            worker.json_tools_schema = original_schema
            worker.schedule_background_memory_review = original_review
            worker.long_term_memory_enabled = original_long_term


if __name__ == "__main__":
    unittest.main()
