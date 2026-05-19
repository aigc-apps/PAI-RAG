import json
import os
import subprocess
import stat
import tempfile
import unittest
from unittest.mock import patch

from backend.aliyun_credentials import (
    AliyunConfigError,
    AliyunCredentialError,
    AliyunCredentials,
    parse_aliyun_credentials,
    write_temporary_profile,
)


class AliyunCredentialParsingTests(unittest.TestCase):
    def test_parse_credentials_defaults_region(self):
        creds = parse_aliyun_credentials({
            'access_key_id': 'test-ak',
            'access_key_secret': 'test-secret',
        })

        self.assertEqual(creds.access_key_id, 'test-ak')
        self.assertEqual(creds.access_key_secret, 'test-secret')
        self.assertEqual(creds.region_id, 'cn-beijing')

    def test_parse_credentials_rejects_partial_payload(self):
        with self.assertRaises(AliyunCredentialError):
            parse_aliyun_credentials({'access_key_id': 'test-ak'})

    def test_parse_credentials_rejects_non_object(self):
        with self.assertRaises(AliyunCredentialError):
            parse_aliyun_credentials(None)


class AliyunConfigProfileTests(unittest.TestCase):
    def read_config(self, path):
        with open(path, encoding='utf-8') as file:
            return json.load(file)

    def test_write_and_cleanup_temporary_profile_preserves_existing_config(self):
        with tempfile.TemporaryDirectory() as root:
            config_dir = os.path.join(root, '.aliyun')
            os.makedirs(config_dir)
            config_path = os.path.join(config_dir, 'config.json')
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump({
                    'current': 'default',
                    'profiles': [{
                        'name': 'default',
                        'mode': 'AK',
                        'access_key_id': 'existing-ak',
                        'access_key_secret': 'existing-secret',
                        'region_id': 'cn-hangzhou',
                    }],
                }, file)

            lease = write_temporary_profile(
                AliyunCredentials('request-ak', 'request-secret', 'cn-beijing'),
                config_path=config_path,
            )

            data = self.read_config(config_path)
            self.assertEqual(data['current'], lease.profile_name)
            profiles = {item['name']: item for item in data['profiles']}
            self.assertIn('default', profiles)
            self.assertIn(lease.profile_name, profiles)
            self.assertTrue(lease.profile_name.startswith('pai-rag-runtime-'))
            self.assertEqual(profiles[lease.profile_name]['access_key_id'], 'request-ak')
            self.assertEqual(profiles[lease.profile_name]['access_key_secret'], 'request-secret')
            self.assertEqual(profiles[lease.profile_name]['region_id'], 'cn-beijing')
            self.assertEqual(lease.env['ALIBABA_CLOUD_PROFILE'], lease.profile_name)
            self.assertEqual(lease.env['PAI_RAG_REQUEST_ALIYUN_CREDENTIALS'], '1')

            self.assertTrue(lease.cleanup())

            data = self.read_config(config_path)
            profiles = {item['name']: item for item in data['profiles']}
            self.assertEqual(data['current'], 'default')
            self.assertIn('default', profiles)
            self.assertNotIn(lease.profile_name, profiles)

    def test_write_profile_creates_private_config_file(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, '.aliyun', 'config.json')

            lease = write_temporary_profile(
                AliyunCredentials('request-ak', 'request-secret', 'cn-beijing'),
                config_path=config_path,
            )

            self.assertTrue(os.path.exists(config_path))
            self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(config_path)).st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(os.stat(config_path).st_mode), 0o600)
            lease.cleanup()

    def test_configure_set_is_attempted_and_cli_current_profile_is_kept(self):
        with tempfile.TemporaryDirectory() as root:
            config_dir = os.path.join(root, '.aliyun')
            os.makedirs(config_dir)
            config_path = os.path.join(config_dir, 'config.json')
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump({
                    'current': 'default',
                    'profiles': [{
                        'name': 'default',
                        'mode': 'AK',
                        'access_key_id': 'existing-ak',
                        'access_key_secret': 'existing-secret',
                        'region_id': 'cn-hangzhou',
                    }],
                }, file)

            def fake_run(cmd, **kwargs):
                profile_name = cmd[cmd.index('--profile') + 1]
                with open(config_path, encoding='utf-8') as file:
                    data = json.load(file)
                data['current'] = profile_name
                data['profiles'].append({
                    'name': profile_name,
                    'mode': 'AK',
                    'access_key_id': 'cli-ak',
                    'access_key_secret': 'cli-secret',
                    'region_id': 'cn-shanghai',
                })
                with open(config_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file)
                return subprocess.CompletedProcess(cmd, 0)

            with patch('backend.aliyun_credentials.subprocess.run', side_effect=fake_run) as run:
                lease = write_temporary_profile(
                    AliyunCredentials('request-ak', 'request-secret', 'cn-beijing'),
                    config_path=config_path,
                    configure_with_cli=True,
                )

            run.assert_called_once()
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[:3], ['aliyun', 'configure', 'set'])
            self.assertIn('--profile', cmd)
            self.assertIn(lease.profile_name, cmd)
            self.assertIn('--access-key-id', cmd)
            self.assertIn('request-ak', cmd)
            self.assertIn('--access-key-secret', cmd)
            self.assertIn('request-secret', cmd)
            self.assertIs(run.call_args.kwargs['stdout'], subprocess.DEVNULL)
            self.assertIs(run.call_args.kwargs['stderr'], subprocess.DEVNULL)

            data = self.read_config(config_path)
            profiles = {item['name']: item for item in data['profiles']}
            self.assertEqual(data['current'], lease.profile_name)
            self.assertEqual(profiles[lease.profile_name]['access_key_id'], 'request-ak')
            self.assertEqual(profiles[lease.profile_name]['access_key_secret'], 'request-secret')
            self.assertEqual(profiles[lease.profile_name]['region_id'], 'cn-beijing')

    def test_invalid_config_json_fails_without_overwriting(self):
        with tempfile.TemporaryDirectory() as root:
            config_dir = os.path.join(root, '.aliyun')
            os.makedirs(config_dir)
            config_path = os.path.join(config_dir, 'config.json')
            with open(config_path, 'w', encoding='utf-8') as file:
                file.write('not json{{')

            with self.assertRaises(AliyunConfigError):
                write_temporary_profile(
                    AliyunCredentials('request-ak', 'request-secret', 'cn-beijing'),
                    config_path=config_path,
                )

            with open(config_path, encoding='utf-8') as file:
                self.assertEqual(file.read(), 'not json{{')


if __name__ == '__main__':
    unittest.main()
