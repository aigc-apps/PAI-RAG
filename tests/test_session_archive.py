import os
import tempfile
import unittest
from datetime import datetime

from backend.session_archive import archive_filename, archive_session_record


class SessionArchiveTests(unittest.TestCase):
    def test_archive_filename_matches_l4_timestamp_style(self):
        self.assertEqual(
            archive_filename('abcdef12-3456-7890', '2026-05-18T17:28:31.123456'),
            '20260518_172831_abcdef12.md',
        )

    def test_archive_session_record_writes_markdown_history(self):
        with tempfile.TemporaryDirectory() as root:
            record = {
                'session_id': 'abcdef12-3456-7890',
                'user_id': '__server__',
                'title': 'hello',
                'status': 'completed',
                'created_at': '2026-05-18T17:28:31',
                'updated_at': '2026-05-18T17:29:01',
                'workspace_path': '/tmp/workspace',
                'ui_messages': [
                    {'role': 'user', 'content': 'hello'},
                    {
                        'role': 'assistant',
                        'content': 'world',
                        'events': [{'sessionUpdate': 'thought_delta', 'content': {'text': 'private'}}],
                    },
                ],
            }

            path = archive_session_record(record, os.path.join(root, 'L4_raw_sessions'))

            self.assertTrue(path.endswith('20260518_172831_abcdef12.md'))
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('# Task (20260518_172831)', content)
            self.assertIn('## Transcript', content)
            self.assertIn('### 1. User\nhello', content)
            self.assertIn('### 2. Assistant\nworld', content)
            self.assertIn('## History', content)
            self.assertIn('"role": "user"', content)
            self.assertIn('## UI Messages', content)
            self.assertIn('"sessionUpdate": "thought_delta"', content)

    def test_empty_session_is_not_archived(self):
        with tempfile.TemporaryDirectory() as root:
            path = archive_session_record(
                {'session_id': 'empty', 'created_at': '2026-05-18T17:28:31', 'ui_messages': []},
                os.path.join(root, 'L4_raw_sessions'),
            )

            self.assertEqual(path, '')
            self.assertFalse(os.path.exists(os.path.join(root, 'L4_raw_sessions')))

    def test_created_after_skips_old_sessions_without_existing_archive(self):
        with tempfile.TemporaryDirectory() as root:
            archive_dir = os.path.join(root, 'L4_raw_sessions')
            record = {
                'session_id': 'old-session',
                'created_at': '2026-05-18T10:00:00',
                'ui_messages': [{'role': 'user', 'content': 'old'}],
            }

            path = archive_session_record(
                record,
                archive_dir,
                created_after=datetime.fromisoformat('2026-05-18T11:00:00'),
            )

            self.assertEqual(path, '')
            self.assertEqual(os.listdir(archive_dir), [])


if __name__ == '__main__':
    unittest.main()
