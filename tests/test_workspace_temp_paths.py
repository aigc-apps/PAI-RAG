import contextlib
import io
import json
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

    def run_pwd(self, handler, cwd):
        return handler.do_code_run(
            {
                "cwd": cwd,
                "script": "import os\nprint(os.getcwd())",
                "type": "python",
            },
            SimpleNamespace(content=""),
        )

    def workspace_path(self, workspace, ref_path):
        rel_path = ref_path[2:] if ref_path.startswith("./") else ref_path
        return os.path.join(workspace, rel_path)

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

    def test_code_run_project_root_cwd_maps_to_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            handler, workspace = self.make_handler(root)

            outcome = self.run_pwd(handler, root)

            self.assertEqual(outcome.data["status"], "success")
            self.assertEqual(os.path.realpath(outcome.data["stdout"].strip()), os.path.realpath(workspace))

    def test_code_run_skill_cwd_maps_to_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            handler, workspace = self.make_handler(root)
            skill_dir = os.path.join(root, "skills", "example")
            os.makedirs(skill_dir, exist_ok=True)
            handler.allow_readonly_root(skill_dir)

            outcome = self.run_pwd(handler, skill_dir)

            self.assertEqual(outcome.data["status"], "success")
            self.assertEqual(os.path.realpath(outcome.data["stdout"].strip()), os.path.realpath(workspace))

    def test_code_run_other_external_cwd_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            handler, _ = self.make_handler(root)

            with self.assertRaises(WorkspaceViolation):
                self.run_pwd(handler, outside)

    def test_large_code_run_stdout_is_saved_in_workspace_tmp(self):
        with tempfile.TemporaryDirectory() as root:
            handler, workspace = self.make_handler(root)

            with contextlib.redirect_stdout(io.StringIO()):
                outcome = handler.do_code_run(
                    {
                        "script": "print('x' * 12050)",
                        "type": "python",
                        "_tool_call_id": "tool-large-stdout",
                    },
                    SimpleNamespace(content=""),
                )

            self.assertEqual(outcome.data["status"], "success")
            self.assertTrue(outcome.data["stdout_truncated"])
            self.assertIn("[OUTPUT TRUNCATED:", outcome.data["stdout"])
            self.assertEqual(outcome.data["stdout_path"], "./.tmp/code_run_outputs/tool-large-stdout.stdout")

            saved_path = self.workspace_path(workspace, outcome.data["stdout_path"])
            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, encoding="utf-8") as file:
                full_stdout = file.read()
            self.assertEqual(full_stdout, ("x" * 12050) + "\n")

    def test_large_code_run_json_can_be_parsed_from_saved_stdout_path(self):
        with tempfile.TemporaryDirectory() as root:
            handler, workspace = self.make_handler(root)

            with contextlib.redirect_stdout(io.StringIO()):
                outcome = handler.do_code_run(
                    {
                        "script": "import json\nprint(json.dumps({'payload': 'x' * 12000}))",
                        "type": "python",
                        "_tool_call_id": "tool-json-stdout",
                    },
                    SimpleNamespace(content=""),
                )

            saved_path = self.workspace_path(workspace, outcome.data["stdout_path"])
            with open(saved_path, encoding="utf-8") as file:
                parsed = json.loads(file.read())

            self.assertEqual(len(parsed["payload"]), 12000)
            self.assertTrue(outcome.data["stdout_truncated"])
            self.assertIn("stdout is a preview only", outcome.data["stdout_note"])


if __name__ == "__main__":
    unittest.main()
