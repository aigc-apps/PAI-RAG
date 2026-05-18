import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from session_store import SERVER_USER_ID, SessionStore
import backend.server as server


class SessionStoreResponseTests(unittest.TestCase):
    def test_save_load_latest_and_delete_response(self):
        with tempfile.TemporaryDirectory() as root:
            store = SessionStore(os.path.join(root, "sessions"))
            response = {"id": "resp_1", "object": "response", "output": []}
            history = [{"role": "user", "content": "hello"}]

            store.save_response(
                "resp_1",
                response,
                conversation_history=history,
                instructions="be brief",
                session_id="session_1",
                user_id=SERVER_USER_ID,
                conversation="conv_1",
            )

            loaded = store.load_response("resp_1", user_id=SERVER_USER_ID)
            self.assertEqual(loaded["response"], response)
            self.assertEqual(loaded["conversation_history"], history)
            self.assertEqual(loaded["instructions"], "be brief")
            self.assertEqual(store.latest_response_for_conversation("conv_1"), "resp_1")
            self.assertTrue(store.delete_response("resp_1", user_id=SERVER_USER_ID))
            self.assertIsNone(store.load_response("resp_1", user_id=SERVER_USER_ID))


class ResponseThinkingTagTests(unittest.TestCase):
    def test_taking_tag_is_stripped_from_visible_assistant_text(self):
        text = "hello\n<taking>private plan</taking>\nworld"
        self.assertEqual(server._strip_thinking_blocks(text).strip(), "hello\n\nworld")

    def test_working_tag_is_stripped_from_visible_assistant_text(self):
        text = "hello\n<working>private work</working>\nworld"
        self.assertEqual(server._strip_thinking_blocks(text).strip(), "hello\n\nworld")

    def test_skill_context_tag_is_stripped_from_visible_assistant_text(self):
        text = "hello\n<skill_context>private skill context</skill_context>\nworld"
        self.assertEqual(server._strip_thinking_blocks(text).strip(), "hello\n\nworld")

    def test_taking_tag_is_reconstructed_as_agent_step_event(self):
        updates = server._agent_updates_from_output([
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<taking>private plan</taking>\nvisible answer",
                    },
                ],
            },
        ])

        self.assertEqual(updates[0]["sessionUpdate"], "thought_start")
        self.assertEqual(updates[1]["sessionUpdate"], "thought_delta")
        self.assertEqual(updates[1]["content"]["text"], "private plan")
        self.assertEqual(updates[2]["sessionUpdate"], "thought_done")
        self.assertEqual(updates[3]["sessionUpdate"], "agent_message_chunk")
        self.assertEqual(updates[3]["content"]["text"].strip(), "visible answer")

    def test_working_tag_is_reconstructed_as_agent_step_event(self):
        updates = server._agent_updates_from_output([
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<working>private work</working>\nvisible answer",
                    },
                ],
            },
        ])

        self.assertEqual(updates[0]["sessionUpdate"], "thought_start")
        self.assertEqual(updates[1]["sessionUpdate"], "thought_delta")
        self.assertEqual(updates[1]["content"]["text"], "private work")
        self.assertEqual(updates[2]["sessionUpdate"], "thought_done")
        self.assertEqual(updates[3]["sessionUpdate"], "agent_message_chunk")
        self.assertEqual(updates[3]["content"]["text"].strip(), "visible answer")

    def test_skill_context_tag_is_reconstructed_as_agent_step_event(self):
        updates = server._agent_updates_from_output([
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<skill_context>private skill context</skill_context>\nvisible answer",
                    },
                ],
            },
        ])

        self.assertEqual(updates[0]["sessionUpdate"], "thought_start")
        self.assertEqual(updates[1]["sessionUpdate"], "thought_delta")
        self.assertEqual(updates[1]["content"]["text"], "private skill context")
        self.assertEqual(updates[2]["sessionUpdate"], "thought_done")
        self.assertEqual(updates[3]["sessionUpdate"], "agent_message_chunk")
        self.assertEqual(updates[3]["content"]["text"].strip(), "visible answer")

    def test_final_report_contract_text_is_authoritative(self):
        text = server._final_assistant_text([
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "<summary>done</summary>"}],
            },
            {
                "type": "message",
                "metadata": {"pai_final_report": True},
                "content": [{"type": "output_text", "text": "## Report\nReadable result."}],
            },
        ])

        self.assertEqual(text, "## Report\nReadable result.")


class BackgroundReviewSchedulingTests(unittest.TestCase):
    def _session(self, memory_root):
        class _Session:
            sid = "sess_1"
            user_id = SERVER_USER_ID
            memory_scope = SimpleNamespace(root=memory_root)
            ui_msgs = []
            _lock = threading.RLock()

            def save(self):
                pass

        return _Session()

    def test_completed_turn_schedules_background_review_after_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)
            archive_path = os.path.join(tmp, "L4_raw_sessions", "session.md")

            with mock.patch.object(server, "_archive_session_for_replay", return_value=archive_path), \
                 mock.patch.object(server, "schedule_background_memory_review", return_value=True) as schedule:
                result = server._append_turn_to_session(
                    sess,
                    user_text="请诊断",
                    assistant_text="诊断完成",
                    final_status="completed",
                    run_id="resp_1",
                    active_skill="diagnosis",
                )

        self.assertEqual(result, archive_path)
        schedule.assert_called_once()
        kwargs = schedule.call_args.kwargs
        self.assertEqual(kwargs["session_id"], "sess_1")
        self.assertEqual(kwargs["run_id"], "resp_1")
        self.assertEqual(kwargs["memory_root"], tmp)
        self.assertEqual(kwargs["archive_path"], archive_path)
        self.assertTrue(kwargs["long_term_enabled"])
        self.assertEqual(kwargs["active_skill"], "diagnosis")
        self.assertEqual(kwargs["llm_history"], [
            {"role": "user", "content": "请诊断"},
            {"role": "assistant", "content": "诊断完成"},
        ])

    def test_non_completed_turn_does_not_schedule_background_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)
            archive_path = os.path.join(tmp, "L4_raw_sessions", "session.md")

            with mock.patch.object(server, "_archive_session_for_replay", return_value=archive_path), \
                 mock.patch.object(server, "schedule_background_memory_review") as schedule:
                server._append_turn_to_session(
                    sess,
                    user_text="继续",
                    assistant_text="",
                    final_status="requires_action",
                    run_id="resp_pause",
                )

        schedule.assert_not_called()

    def test_empty_archive_path_does_not_schedule_background_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)

            with mock.patch.object(server, "_archive_session_for_replay", return_value=""), \
                 mock.patch.object(server, "schedule_background_memory_review") as schedule:
                server._append_turn_to_session(
                    sess,
                    user_text="请诊断",
                    assistant_text="诊断完成",
                    final_status="completed",
                    run_id="resp_1",
                )

        schedule.assert_not_called()

    def test_background_review_scheduler_failure_does_not_break_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)
            archive_path = os.path.join(tmp, "L4_raw_sessions", "session.md")

            with mock.patch.object(server, "_archive_session_for_replay", return_value=archive_path), \
                 mock.patch.object(server, "schedule_background_memory_review", side_effect=RuntimeError("boom")):
                result = server._append_turn_to_session(
                    sess,
                    user_text="请诊断",
                    assistant_text="诊断完成",
                    final_status="completed",
                    run_id="resp_1",
                )

        self.assertEqual(result, archive_path)
        self.assertEqual(len(sess.ui_msgs), 2)


if __name__ == "__main__":
    unittest.main()
