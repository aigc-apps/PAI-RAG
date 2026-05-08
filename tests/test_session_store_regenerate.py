import os
import tempfile
import unittest

from session_store import SERVER_USER_ID, SessionStore


class SessionStoreRegenerateTests(unittest.TestCase):
    def make_store(self, root):
        return SessionStore(os.path.join(root, "sessions"))

    def test_regenerate_restores_pre_run_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            store = self.make_store(root)
            base_ui = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer", "events": []},
            ]
            base_history = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ]
            base_state = {"history_info": ["old summary"], "working": {"key_info": "keep"}}
            store.save(
                "s1",
                llm_history=base_history,
                ui_messages=base_ui,
                handler_state=base_state,
                user_id=SERVER_USER_ID,
                status="idle",
                workspace_path="/workspace/s1",
            )

            started = store.try_start_run("s1", SERVER_USER_ID, "r1", "events", "new question", "/workspace/s1")
            self.assertEqual(started["status"], "started")
            run = store.load_run("r1", SERVER_USER_ID)
            self.assertEqual(run["metadata"]["pre_run_snapshot"]["input_text"], "new question")
            self.assertEqual(run["metadata"]["pre_run_snapshot"]["ui_message_count"], 2)

            final_ui = base_ui + [
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer", "events": []},
            ]
            final_history = base_history + [
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]
            store.save_run_snapshot(
                "s1",
                SERVER_USER_ID,
                "r1",
                llm_history=final_history,
                ui_messages=final_ui,
                handler_state={"history_info": ["old summary", "new summary"], "working": {"key_info": "changed"}},
                status="completed",
                active_run_id="r1",
                workspace_path="/workspace/s1",
            )
            store.finish_run("s1", SERVER_USER_ID, "r1", "completed")

            regenerated = store.try_start_regenerate_run("s1", SERVER_USER_ID, "r2", "events", "/workspace/s1")

            self.assertEqual(regenerated["status"], "started")
            self.assertEqual(regenerated["input_text"], "new question")
            self.assertEqual(regenerated["regenerated_from_run_id"], "r1")
            loaded = store.load("s1", SERVER_USER_ID)
            self.assertEqual(loaded["status"], "running")
            self.assertEqual(loaded["active_run_id"], "r2")
            self.assertEqual(loaded["llm_history"], base_history)
            self.assertEqual(loaded["handler_state"], base_state)
            self.assertEqual(
                loaded["ui_messages"],
                base_ui + [
                    {"role": "user", "content": "new question"},
                    {"role": "assistant", "content": "", "events": []},
                ],
            )
            regen_run = store.load_run("r2", SERVER_USER_ID)
            self.assertTrue(regen_run["metadata"]["regenerate"])
            self.assertEqual(regen_run["metadata"]["regenerated_from_run_id"], "r1")

    def test_regenerate_uses_legacy_fallback_without_run_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            store = self.make_store(root)
            ui_messages = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer", "events": []},
                {"role": "user", "content": "latest question"},
                {"role": "assistant", "content": "latest answer", "events": []},
            ]
            llm_history = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "latest question"},
                {"role": "assistant", "content": "latest answer"},
            ]
            store.save("legacy", llm_history, ui_messages, handler_state={"history_info": ["current"]}, status="idle")

            regenerated = store.try_start_regenerate_run("legacy", SERVER_USER_ID, "r2", "events")

            self.assertEqual(regenerated["status"], "started")
            self.assertEqual(regenerated["input_text"], "latest question")
            loaded = store.load("legacy", SERVER_USER_ID)
            self.assertEqual(
                loaded["ui_messages"],
                ui_messages[:2] + [
                    {"role": "user", "content": "latest question"},
                    {"role": "assistant", "content": "", "events": []},
                ],
            )
            self.assertEqual(loaded["llm_history"], llm_history[:2])

    def test_regenerate_rejects_active_session(self):
        with tempfile.TemporaryDirectory() as root:
            store = self.make_store(root)
            store.save(
                "busy",
                llm_history=[],
                ui_messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                status="running",
                active_run_id="r1",
            )

            result = store.try_start_regenerate_run("busy", SERVER_USER_ID, "r2", "events")

            self.assertEqual(result["status"], "busy")
            self.assertEqual(store.load("busy", SERVER_USER_ID)["active_run_id"], "r1")

    def test_regenerate_rejects_session_without_completed_answer(self):
        with tempfile.TemporaryDirectory() as root:
            store = self.make_store(root)
            store.save("empty", llm_history=[], ui_messages=[{"role": "user", "content": "q"}], status="idle")

            result = store.try_start_regenerate_run("empty", SERVER_USER_ID, "r2", "events")

            self.assertEqual(result["status"], "no_regeneratable_answer")


if __name__ == "__main__":
    unittest.main()
