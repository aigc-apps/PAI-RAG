import os
import tempfile
import unittest
from unittest import mock

import backend.background_review as background_review
from backend.tool_schemas import background_memory_review_tools_schema, main_tools_schema
from tools import WorkspaceViolation


def tool_names(schemas):
    return {schema["function"]["name"] for schema in schemas}


class BackgroundReviewTests(unittest.TestCase):
    def test_foreground_schema_excludes_long_term_update_tool(self):
        self.assertNotIn("start_long_term_update", tool_names(main_tools_schema()))
        self.assertIn("file_read", tool_names(main_tools_schema()))

    def test_background_review_schema_is_memory_only(self):
        self.assertEqual(
            tool_names(background_memory_review_tools_schema()),
            {"start_long_term_update", "file_read", "file_patch", "file_write"},
        )

    def test_background_review_uses_isolated_memory_workspace(self):
        original_runner = background_review.agent_runner_loop
        calls = {}

        def fake_runner(**kwargs):
            calls.update(kwargs)
            handler = kwargs["handler"]
            self.assertEqual(os.path.abspath(handler.memory_root), memory_root)
            self.assertEqual(os.path.abspath(handler.workspace_root), memory_root)
            with self.assertRaises(WorkspaceViolation):
                handler._write_path(os.path.dirname(memory_root))
            return {"result": "NO_TOOL_CALL"}

        with tempfile.TemporaryDirectory() as tmp:
            memory_root = os.path.abspath(tmp)
            try:
                background_review.agent_runner_loop = fake_runner
                result = background_review.run_background_memory_review(
                    user_id="server",
                    session_id="s1",
                    run_id="r1",
                    task_text="diagnose startup",
                    llm_history=[{"role": "user", "content": "question"}],
                    memory_root=memory_root,
                    active_skill="diagnosis",
                )
            finally:
                background_review.agent_runner_loop = original_runner

        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls["client"].history, [{"role": "user", "content": "question"}])
        self.assertEqual(tool_names(calls["tools_schema"]), tool_names(background_memory_review_tools_schema()))
        self.assertIn("后台长期记忆审查 Agent", calls["system_prompt"])
        self.assertIn("start_long_term_update", calls["user_input"])

    def test_schedule_logs_skip_reasons(self):
        with self.assertLogs("backend.background_review", level="INFO") as logs:
            scheduled = background_review.schedule_background_memory_review(
                session_id="s1",
                run_id="r1",
                memory_root="/memory/root",
                final_status="failed",
                long_term_enabled=True,
            )

        self.assertFalse(scheduled)
        self.assertIn("reason=final_status", "\n".join(logs.output))

        with self.assertLogs("backend.background_review", level="INFO") as logs:
            scheduled = background_review.schedule_background_memory_review(
                session_id="s1",
                run_id="r1",
                memory_root="/memory/root",
                final_status="completed",
                long_term_enabled=False,
            )

        self.assertFalse(scheduled)
        self.assertIn("reason=long_term_disabled", "\n".join(logs.output))

    def test_schedule_logs_thread_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("backend.background_review.threading.Thread") as thread_cls:
                thread = thread_cls.return_value
                with self.assertLogs("backend.background_review", level="INFO") as logs:
                    scheduled = background_review.schedule_background_memory_review(
                        session_id="s1",
                        run_id="r1",
                        memory_root=tmp,
                        final_status="completed",
                        long_term_enabled=True,
                    )

        self.assertTrue(scheduled)
        thread.start.assert_called_once()
        self.assertIn("Background memory review scheduled: backend=thread", "\n".join(logs.output))

    def test_schedule_logs_celery_enqueue(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(background_review.review_memory_task, "delay") as delay:
                with self.assertLogs("backend.background_review", level="INFO") as logs:
                    scheduled = background_review.schedule_background_memory_review(
                        session_id="s1",
                        run_id="r1",
                        memory_root=tmp,
                        final_status="completed",
                        long_term_enabled=True,
                        use_celery=True,
                    )

        self.assertTrue(scheduled)
        delay.assert_called_once()
        self.assertIn("Background memory review enqueued: backend=celery", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
