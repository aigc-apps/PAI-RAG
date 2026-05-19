import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import backend.server as server
from backend.agents_sdk.lifecycle import RUN_STATE_COMPLETED, RUN_STATE_REQUIRES_ACTION
from backend.agents_sdk.run_state_store import RunStateStore
from backend.agents_sdk.runner import StreamFrame
from session_store import SERVER_USER_ID, SessionStore


class _Request:
    def __init__(self, body, headers=None):
        self.body = body
        self.headers = headers or {}

    async def json(self):
        return self.body


def _response(text):
    return {
        'id': 'resp_prev',
        'object': 'response',
        'status': 'completed',
        'output': [{
            'type': 'message',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': text}],
        }],
    }


def _minimal_sse(**_kwargs):
    yield 'event: response.created\ndata: {"type":"response.created","id":"resp_new"}\n\n'
    yield 'event: response.completed\ndata: {"type":"response.completed","id":"resp_new","status":"completed"}\n\n'
    yield 'data: [DONE]\n\n'


async def _minimal_chat_sse(**_kwargs):
    yield 'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1,"model":"qwen-test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    yield 'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1,"model":"qwen-test","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
    yield 'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1,"model":"qwen-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    yield 'data: [DONE]\n\n'


class _Service:
    def __init__(self, store):
        self.store = store


class _Lease:
    profile_name = 'pai-rag-runtime-test'
    env = {'ALIBABA_CLOUD_PROFILE': profile_name}

    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True
        return True


class PreviousResponseIdHttpRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(os.path.join(self.tmp.name, 'sessions'))
        self.original_service = server.service
        self.original_run_state_store = server._sdk_run_state_store
        self.original_audit_store = server._sdk_audit_store
        self.original_audit_publisher = server._sdk_audit_publisher
        server.service = _Service(self.store)
        self.run_state_store = RunStateStore(connect=self.store._connect)
        server._sdk_run_state_store = self.run_state_store
        server._sdk_audit_store = None
        server._sdk_audit_publisher = None
        self.stores_patch = patch.object(server, '_sdk_stores', return_value=(self.run_state_store, None))
        self.stores_patch.start()

    def tearDown(self):
        self.stores_patch.stop()
        server.service = self.original_service
        server._sdk_run_state_store = self.original_run_state_store
        server._sdk_audit_store = self.original_audit_store
        server._sdk_audit_publisher = self.original_audit_publisher
        self.tmp.cleanup()

    async def test_plain_previous_response_id_does_not_require_run_state(self):
        self.store.save_response(
            'resp_prev',
            _response('hello'),
            conversation_history=[{'role': 'user', 'content': 'hi'}],
            session_id='sess_prev',
            user_id=SERVER_USER_ID,
        )

        with patch.object(server, '_sdk_response_stream', side_effect=_minimal_sse) as fake:
            response = await server.create_response(_Request({
                'previous_response_id': 'resp_prev',
                'input': 'follow up',
                'stream': True,
            }))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake.call_args.kwargs['previous_response_id'], 'resp_prev')
        self.assertEqual(fake.call_args.kwargs['session_id'], 'sess_prev')
        self.assertFalse(fake.call_args.kwargs['allow_hitl'])

    async def test_allow_hitl_request_flag_is_forwarded(self):
        with patch.object(server, '_sdk_response_stream', side_effect=_minimal_sse) as fake:
            response = await server.create_response(_Request({
                'input': 'hello',
                'stream': True,
                'allow_hitl': True,
            }))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(fake.call_args.kwargs['allow_hitl'])

    async def test_responses_aliyun_credentials_create_profile_and_are_stripped(self):
        lease = _Lease()
        with patch.object(server, 'write_temporary_profile', return_value=lease) as write_profile, \
             patch.object(server, '_sdk_response_stream', side_effect=_minimal_sse) as fake:
            response = await server.create_response(_Request({
                'input': 'hello',
                'stream': True,
                'aliyun_credentials': {
                    'access_key_id': 'request-ak',
                    'access_key_secret': 'request-secret',
                    'region_id': 'cn-beijing',
                },
            }))

        self.assertEqual(response.status_code, 200)
        write_profile.assert_called_once()
        self.assertEqual(write_profile.call_args.args[0].access_key_id, 'request-ak')
        self.assertNotIn('aliyun_credentials', fake.call_args.kwargs['body'])
        self.assertEqual(fake.call_args.kwargs['tool_env'], lease.env)

    async def test_invalid_aliyun_credentials_are_rejected_before_profile_write(self):
        with patch.object(server, 'write_temporary_profile') as write_profile:
            response = await server.create_response(_Request({
                'input': 'hello',
                'stream': True,
                'aliyun_credentials': {'access_key_id': 'request-ak'},
            }))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)['error']['code'], 'invalid_aliyun_credentials')
        write_profile.assert_not_called()

    async def test_conversation_uses_existing_session_for_plain_turn(self):
        self.store.save(
            session_id='sess_frontend',
            user_id=SERVER_USER_ID,
            llm_history=[],
            ui_messages=[],
        )

        with patch.object(server, '_sdk_response_stream', side_effect=_minimal_sse) as fake:
            response = await server.create_response(_Request({
                'conversation': 'sess_frontend',
                'input': 'hello',
                'stream': True,
            }))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake.call_args.kwargs['session_id'], 'sess_frontend')
        self.assertEqual(fake.call_args.kwargs['conversation'], 'sess_frontend')

    async def test_legacy_responses_fields_are_rejected(self):
        for field, value in (
            ('session_id', 'sess_legacy'),
            ('conversation_history', [{'role': 'user', 'content': 'old'}]),
            ('messages', [{'role': 'user', 'content': 'old'}]),
        ):
            response = await server.create_response(_Request({
                field: value,
                'input': 'hello',
                'stream': True,
            }))
            self.assertEqual(response.status_code, 400, field)
            self.assertEqual(json.loads(response.body)['error']['code'], 'unsupported_legacy_field')

    async def test_resume_input_defaults_to_allow_hitl(self):
        self.run_state_store.upsert(
            id='resp_pause',
            session_id='sess_pause',
            run_id='run_pause',
            response_id='resp_pause',
            user_id=SERVER_USER_ID,
            model='qwen-test',
            status=RUN_STATE_REQUIRES_ACTION,
            run_state_blob='blob',
            pending_interruption_json='[]',
            last_event_id=None,
            audit_log_id='audit_pause',
        )

        with patch.object(server, '_sdk_response_stream', side_effect=_minimal_sse) as fake:
            response = await server.create_response(_Request({
                'previous_response_id': 'resp_pause',
                'input': [{'type': 'function_call_output', 'call_id': 'call_1', 'output': 'A'}],
                'stream': True,
            }))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(fake.call_args.kwargs['allow_hitl'])

    async def test_resume_input_requires_paused_run_state(self):
        self.store.save_response(
            'resp_done',
            _response('done'),
            conversation_history=[{'role': 'user', 'content': 'hi'}],
            session_id='sess_done',
            user_id=SERVER_USER_ID,
        )
        self.run_state_store.upsert(
            id='resp_done',
            session_id='sess_done',
            run_id='run_done',
            response_id='resp_done',
            user_id=SERVER_USER_ID,
            model='qwen-test',
            status=RUN_STATE_COMPLETED,
            run_state_blob='blob',
            pending_interruption_json=None,
            last_event_id=None,
            audit_log_id='audit_done',
        )

        response = await server.create_response(_Request({
            'previous_response_id': 'resp_done',
            'input': [{'type': 'function_call_output', 'call_id': 'call_1', 'output': 'A'}],
            'stream': True,
        }))

        self.assertEqual(response.status_code, 409)
        self.assertEqual(json.loads(response.body)['error']['code'], 'not_resumable')

    async def test_paused_response_with_plain_text_is_invalid_resume(self):
        self.run_state_store.upsert(
            id='resp_pause',
            session_id='sess_pause',
            run_id='run_pause',
            response_id='resp_pause',
            user_id=SERVER_USER_ID,
            model='qwen-test',
            status=RUN_STATE_REQUIRES_ACTION,
            run_state_blob='blob',
            pending_interruption_json='[]',
            last_event_id=None,
            audit_log_id='audit_pause',
        )

        response = await server.create_response(_Request({
            'previous_response_id': 'resp_pause',
            'input': 'plain text',
            'stream': True,
        }))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)['error']['code'], 'invalid_resume')

    async def test_chat_completions_non_stream_is_public_api(self):
        with patch.object(server, '_sdk_chat_stream', side_effect=_minimal_chat_sse) as fake:
            response = await server.chat_completions(_Request({
                'messages': [{'role': 'user', 'content': 'hi'}],
                'stream': False,
            }))

        self.assertEqual(response['object'], 'chat.completion')
        self.assertEqual(response['choices'][0]['message']['content'], 'ok')
        self.assertEqual(fake.call_args.kwargs['cwd'], None)

    async def test_chat_completions_aliyun_credentials_create_profile_and_cleanup(self):
        lease = _Lease()
        with patch.object(server, 'write_temporary_profile', return_value=lease) as write_profile, \
             patch.object(server, '_sdk_chat_stream', side_effect=_minimal_chat_sse) as fake:
            response = await server.chat_completions(_Request({
                'messages': [{'role': 'user', 'content': 'hi'}],
                'stream': False,
                'aliyun_credentials': {
                    'access_key_id': 'request-ak',
                    'access_key_secret': 'request-secret',
                    'region_id': 'cn-beijing',
                },
            }))

        self.assertEqual(response['object'], 'chat.completion')
        write_profile.assert_called_once()
        self.assertNotIn('aliyun_credentials', fake.call_args.kwargs['body'])
        self.assertEqual(fake.call_args.kwargs['tool_env'], lease.env)
        self.assertTrue(lease.cleaned)

    async def test_chat_completions_rejects_legacy_session_header(self):
        response = await server.chat_completions(_Request({
            'messages': [{'role': 'user', 'content': 'hi'}],
            'stream': False,
        }, headers={'x-session-id': 'legacy'}))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)['error']['code'], 'unsupported_legacy_header')


class PreviousResponseIdStreamRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(os.path.join(self.tmp.name, 'sessions'))
        self.original_service = server.service
        server.service = _Service(self.store)

    def tearDown(self):
        server.service = self.original_service
        self.tmp.cleanup()

    async def test_plain_previous_response_builds_history_and_starts_new_run(self):
        self.store.save_response(
            'resp_prev',
            _response('hello'),
            conversation_history=[{'role': 'user', 'content': 'hi'}],
            session_id='sess_prev',
            user_id=SERVER_USER_ID,
        )
        captured = {}

        async def fake_runner(**kwargs):
            captured.update(kwargs)
            yield StreamFrame(terminal=True, response_object={
                'id': 'resp_new',
                'object': 'response',
                'status': 'completed',
                'model': 'qwen-test',
                'output': [],
            })

        fake_session = SimpleNamespace(
            sid='sess_prev',
            workspace_path='/tmp',
            cwd='/tmp',
            workspace_root=None,
            readonly_roots=[],
            ui_msgs=[],
            _lock=threading.RLock(),
            save=lambda: None,
        )

        with patch.object(server, '_sdk_stores', return_value=(object(), None)), \
             patch.object(server, 'ensure_session', return_value=fake_session), \
             patch('backend.agents_sdk.runtime_setup.ensure_sdk_runtime', return_value=None), \
             patch('backend.tools.wrappers.build_tool_list', return_value=[]), \
             patch('tools.GenericHandler', return_value=object()), \
             patch('backend.agents_sdk.runner.stream_responses_run', side_effect=fake_runner):
            chunks = []
            async for chunk in server._sdk_response_stream(
                body={'input': 'follow up', 'stream': True},
                model='qwen-test',
                model_override=None,
                instructions='',
                conversation='',
                session_id='',
                cwd=None,
                store=True,
                previous_response_id='resp_prev',
            ):
                chunks.append(chunk)

        self.assertIn('data: [DONE]', ''.join(chunks))
        self.assertIsNone(captured['previous_response_id'])
        self.assertIsNone(captured['resume'])
        self.assertEqual(captured['input_items'], [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
            {'role': 'user', 'content': 'follow up'},
        ])

    async def test_session_id_plain_turn_builds_session_history(self):
        captured = {}

        async def fake_runner(**kwargs):
            captured.update(kwargs)
            yield StreamFrame(terminal=True, response_object={
                'id': 'resp_new',
                'object': 'response',
                'status': 'completed',
                'model': 'qwen-test',
                'output': [],
            })

        fake_session = SimpleNamespace(
            sid='sess_frontend',
            workspace_path='/tmp',
            cwd='/tmp',
            workspace_root=None,
            readonly_roots=[],
            ui_msgs=[
                {'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': 'hello'},
            ],
            _lock=threading.RLock(),
            save=lambda: None,
        )

        with patch.object(server, '_sdk_stores', return_value=(object(), None)), \
             patch.object(server, 'ensure_session', return_value=fake_session), \
             patch('backend.agents_sdk.runtime_setup.ensure_sdk_runtime', return_value=None), \
             patch('backend.tools.wrappers.build_tool_list', return_value=[]), \
             patch('tools.GenericHandler', return_value=object()), \
             patch('backend.agents_sdk.runner.stream_responses_run', side_effect=fake_runner):
            chunks = []
            async for chunk in server._sdk_response_stream(
                body={'input': 'follow up', 'stream': True},
                model='qwen-test',
                model_override=None,
                instructions='',
                conversation='',
                session_id='sess_frontend',
                cwd=None,
                store=True,
                previous_response_id='',
            ):
                chunks.append(chunk)

        self.assertIn('data: [DONE]', ''.join(chunks))
        self.assertIsNone(captured['previous_response_id'])
        self.assertIsNone(captured['resume'])
        self.assertEqual(captured['input_items'], [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
            {'role': 'user', 'content': 'follow up'},
        ])


if __name__ == '__main__':
    unittest.main()
