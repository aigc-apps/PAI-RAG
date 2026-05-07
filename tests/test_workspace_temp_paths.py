import os
import tempfile
import unittest
from types import SimpleNamespace

from tools import GenericHandler, WorkspaceViolation, code_run


class WorkspaceTemporaryPathTests(unittest.TestCase):
    def make_handler(self, root):
        workspace = os.path.join(root, "workspace")
        os.makedirs(workspace, exist_ok=True)
        return GenericHandler(
            cwd=workspace,
            mini_agent_root=root,
            workspace_root=workspace,
            memory_root=os.path.join(root, "memory"),
        ), workspace

    def write_file(self, handler, path, content="ok"):
        response = SimpleNamespace(content=f"<file_content>{content}</file_content>")
        return handler.do_file_write({"path": path, "mode": "overwrite"}, response)

    def test_system_tmp_write_maps_to_workspace_tmp(self):
        with tempfile.TemporaryDirectory() as root:
            handler, workspace = self.make_handler(root)

            outcome = self.write_file(handler, "/tmp/embedding_config.json", '{"ok": true}')

            self.assertEqual(outcome.data["status"], "success")
            mapped = os.path.join(workspace, ".tmp", "embedding_config.json")
            self.assertTrue(os.path.exists(mapped))
            with open(mapped, encoding="utf-8") as file:
                self.assertEqual(file.read(), '{"ok": true}')

    def test_relative_write_stays_in_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            handler, workspace = self.make_handler(root)

            outcome = self.write_file(handler, "embedding_config.json", "relative")

            self.assertEqual(outcome.data["status"], "success")
            with open(os.path.join(workspace, "embedding_config.json"), encoding="utf-8") as file:
                self.assertEqual(file.read(), "relative")

    def test_tmp_parent_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            handler, _ = self.make_handler(root)

            with self.assertRaises(WorkspaceViolation):
                self.write_file(handler, "/tmp/../outside.json", "bad")

    def test_python_code_run_temp_script_is_created_in_cwd(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = code_run(
                "import os\nprint(os.path.dirname(os.path.abspath(__file__)))",
                code_type="python",
                cwd=workspace,
            )

            self.assertEqual(result["status"], "success")
            self.assertIn(workspace, result["stdout"])


if __name__ == "__main__":
    unittest.main()
