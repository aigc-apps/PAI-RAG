"""HTTP-layer regression tests for ``POST /v1/sessions/{id}/regenerate``.

The endpoint used to be a JSON metadata return — the client then opened a
second SSE stream against ``/v1/runs/{id}/events`` to receive tokens. With
the OpenAI Agents SDK migration we deleted the ``/v1/runs`` family and
collapsed regenerate to a single SSE stream. These tests pin the new
contract: 200 + ``text/event-stream`` body + ``[DONE]`` sentinel on
success, and 409 when the session has no answer to regenerate.
"""
import unittest
import asyncio
from unittest.mock import patch

from fastapi import HTTPException

from backend.agent_service import NoRegeneratableAnswerError
import backend.server as server
from session_store import SERVER_USER_ID


class _FakeService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def regenerate_session(self, session_id, user_id):
        self.calls.append((session_id, user_id))
        if self.error:
            raise self.error
        return self.result


async def _fake_stream(**kwargs):
    """Stand-in for ``_sdk_response_stream`` — emits a minimal valid SSE
    skeleton so we can prove the regenerate route forwards the bytes
    without driving a real LLM."""
    yield 'event: response.created\ndata: {"type":"response.created","id":"resp_1"}\n\n'
    yield 'event: response.completed\ndata: {"type":"response.completed","id":"resp_1"}\n\n'
    yield 'data: [DONE]\n\n'


class _Request:
    headers = {'content-length': '2'}

    async def json(self):
        return {}


async def _read_streaming_response(response):
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode('utf-8'))
        else:
            chunks.append(str(chunk))
    return ''.join(chunks)


class RegenerateApiTests(unittest.TestCase):
    def setUp(self):
        self.original_service = server.service

    def tearDown(self):
        server.service = self.original_service

    def test_regenerate_streams_directly_as_sse(self):
        server.service = _FakeService({'session_id': 'session-1', 'user_text': 'hi'})

        with patch.object(server, '_sdk_response_stream', side_effect=_fake_stream):
            response = asyncio.run(server.regenerate_session_answer('session-1', _Request()))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, 'text/event-stream')
        self.assertNotIn('X-Session-Id', response.headers)
        body = asyncio.run(_read_streaming_response(response))
        self.assertIn('event: response.created', body)
        self.assertIn('event: response.completed', body)
        self.assertTrue(body.rstrip().endswith('data: [DONE]'))
        self.assertEqual(server.service.calls, [('session-1', SERVER_USER_ID)])

    def test_regenerate_returns_409_when_no_answer_exists(self):
        server.service = _FakeService(error=NoRegeneratableAnswerError('session-1'))

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(server.regenerate_session_answer('session-1', _Request()))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail['code'], 'no_regeneratable_answer')

    def test_regenerate_returns_404_when_session_missing(self):
        server.service = _FakeService(result=None)

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(server.regenerate_session_answer('missing', _Request()))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == '__main__':
    unittest.main()
