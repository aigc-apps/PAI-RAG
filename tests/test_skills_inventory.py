import os
import tempfile
import unittest
from types import SimpleNamespace

import backend.server as server
from backend.skills_inventory import skills_inventory


class SkillsInventoryTests(unittest.TestCase):
    def make_repo(self, root):
        skill_dir = os.path.join(root, "skills", "official-one")
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as file:
            file.write(
                "---\n"
                "name: official-one\n"
                "description: Official diagnostic flow\n"
                "trigger: /official-one\n"
                "allowed-tools: file_read, code_run\n"
                "---\n"
                "# Official Skill\n"
            )

        memory_root = os.path.join(root, "memory")
        os.makedirs(os.path.join(memory_root, "L4_raw_sessions"), exist_ok=True)
        os.makedirs(os.path.join(memory_root, "users", "u1"), exist_ok=True)
        os.makedirs(os.path.join(memory_root, "sessions"), exist_ok=True)
        files = {
            "global_index.txt": "index\n",
            "global_facts.txt": "facts\n",
            "service_sop.md": "# Service SOP\nUse this flow.\n",
            "helper.py": '"""Helper script."""\n',
            "notes.txt": "ignore\n",
            os.path.join("L4_raw_sessions", "raw.md"): "# raw\n",
            os.path.join("users", "u1", "private_sop.md"): "# private\n",
            os.path.join("sessions", "run.md"): "# session\n",
        }
        for rel, content in files.items():
            path = os.path.join(memory_root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                file.write(content)
        return memory_root

    def test_inventory_classifies_official_and_evolved_skills(self):
        with tempfile.TemporaryDirectory() as root:
            memory_root = self.make_repo(root)

            payload = skills_inventory(root, memory_root)

            self.assertEqual([item["name"] for item in payload["official"]], ["official-one"])
            self.assertEqual(payload["official"][0]["source"], "skills/official-one/SKILL.md")
            self.assertEqual(payload["official"][0]["allowed_tools"], ["file_read", "code_run"])
            evolved_sources = {item["source"] for item in payload["evolved"]}
            self.assertEqual(evolved_sources, {"memory/helper.py", "memory/service_sop.md"})
            kinds = {item["source"]: item["kind"] for item in payload["evolved"]}
            self.assertEqual(kinds["memory/helper.py"], "script")
            self.assertEqual(kinds["memory/service_sop.md"], "sop")
            for section in ("official", "evolved"):
                for item in payload[section]:
                    self.assertFalse(os.path.isabs(item["source"]))

    def test_skills_endpoint_returns_inventory(self):
        original_root = server.ROOT
        original_scope = server.handler_memory_scope
        try:
            with tempfile.TemporaryDirectory() as root:
                memory_root = self.make_repo(root)
                server.ROOT = root
                server.handler_memory_scope = lambda user_id: SimpleNamespace(root=memory_root)

                payload = server.skills()
                self.assertEqual(payload["official"][0]["name"], "official-one")
                self.assertEqual(
                    {item["source"] for item in payload["evolved"]},
                    {"memory/helper.py", "memory/service_sop.md"},
                )
        finally:
            server.ROOT = original_root
            server.handler_memory_scope = original_scope


if __name__ == "__main__":
    unittest.main()
