import os
import tempfile
import unittest

from session_store import SERVER_USER_ID, SessionStore
from backend import server


class ResponseApiHelpersTests(unittest.TestCase):
    def test_build_response_from_run_events_includes_tool_items_and_text(self):
        updates = [
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_1_0",
                "name": "lookup",
                "input": {"query": "abc"},
            },
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_1_0",
                "status": "completed",
                "content": {"text": "found"},
            },
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"text": "final answer"},
            },
            {"sessionUpdate": "done"},
        ]

        response, final_text = server.build_response_from_run_events(
            "resp_test",
            "test-model",
            123,
            "run_test",
            updates,
        )

        self.assertEqual(final_text, "final answer")
        self.assertEqual(response["id"], "resp_test")
        self.assertEqual(response["status"], "completed")
        self.assertEqual([item["type"] for item in response["output"]], [
            "function_call",
            "function_call_output",
            "message",
        ])
        self.assertEqual(response["output"][-1]["content"][0]["text"], "final answer")

    def test_build_response_from_run_events_surfaces_usage_from_done(self):
        updates = [
            {"sessionUpdate": "agent_message_chunk", "content": {"text": "ok"}},
            {
                "sessionUpdate": "done",
                "stopReason": "end_turn",
                "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
            },
        ]

        response, _ = server.build_response_from_run_events(
            "resp_usage", "test-model", 0, "run_usage", updates,
        )

        self.assertEqual(response["usage"], {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "total_tokens": 46,
        })


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


if __name__ == "__main__":
    unittest.main()
