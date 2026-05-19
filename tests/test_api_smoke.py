"""End-to-end smoke test against a running PAI-RAG backend.

Drives the public Responses and Chat Completions surfaces to make sure the SDK runtime is actually
serving requests after the agent_loop → OpenAI Agents SDK rewrite:

- ``/health`` + ``/health/detailed`` (verifies the SDK runner is selected)
- ``/v1/sessions`` (create / get / list / delete)
- ``/v1/responses`` (non-stream + stream + ``previous_response_id`` continuation/resume)
- ``/v1/chat/completions`` (stream)
- ``/v1/models`` and ``/v1/models/active``

The legacy ``/v1/runs`` family was removed when we collapsed onto the SDK
runtime; the smoke suite now asserts every former path is a flat 404.

The whole suite is gated on a live URL — set ``PAIRAG_SMOKE_URL`` to point
at a running server (``http://127.0.0.1:18765`` is the convention used in
local dev). Without it, all tests are skipped so this file stays harmless
in CI environments without a deployed backend.

The test does *not* spin up the server itself: starting uvicorn under
``unittest`` complicates teardown and obscures the very signals we want
this test to surface (port collisions, missing API keys, etc.). Bring up
the service the same way an operator would, then run::

    PAIRAG_SMOKE_URL=http://127.0.0.1:18765 \
        python -m pytest tests/test_api_smoke.py -v
"""
from __future__ import annotations

import json
import os
import unittest
import urllib.error
import urllib.request


SMOKE_URL = os.environ.get('PAIRAG_SMOKE_URL', '').rstrip('/')
SMOKE_TIMEOUT = float(os.environ.get('PAIRAG_SMOKE_TIMEOUT', '60'))


def _request(method, path, *, body=None, headers=None, timeout=None):
    """Plain stdlib HTTP — no ``requests`` dependency. Returns
    ``(status, headers, body_bytes)``; never raises on non-2xx so callers
    can assert on the status directly.
    """
    url = f'{SMOKE_URL}{path}'
    data = None
    h = {'Accept': 'application/json'}
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        h['Content-Type'] = 'application/json'
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout or SMOKE_TIMEOUT) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


def _stream_lines(method, path, *, body=None, headers=None, timeout=None,
                  max_lines=2000):
    """Iterate SSE lines from a streaming endpoint until ``[DONE]`` or
    ``max_lines``. Returns the list of lines actually read so callers can
    assert on event prefixes / payloads.
    """
    url = f'{SMOKE_URL}{path}'
    data = None
    h = {'Accept': 'text/event-stream'}
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        h['Content-Type'] = 'application/json'
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    lines: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout or SMOKE_TIMEOUT) as resp:
        for raw in resp:
            line = raw.decode('utf-8', errors='replace').rstrip('\n').rstrip('\r')
            lines.append(line)
            if line == 'data: [DONE]' or len(lines) >= max_lines:
                break
    return lines


def _parse_sse(lines):
    """Group SSE lines into ``[(event_or_None, data_str), ...]``. ``data:
    [DONE]`` is preserved so the caller can confirm end-of-stream."""
    events = []
    cur_event = None
    cur_data: list[str] = []
    for line in lines:
        if line.startswith('event: '):
            cur_event = line[len('event: '):]
        elif line.startswith('data: '):
            cur_data.append(line[len('data: '):])
        elif line == '':
            if cur_data:
                events.append((cur_event, '\n'.join(cur_data)))
            cur_event = None
            cur_data = []
    if cur_data:
        events.append((cur_event, '\n'.join(cur_data)))
    return events


@unittest.skipUnless(SMOKE_URL, 'set PAIRAG_SMOKE_URL=http://host:port to run smoke tests')
class ApiSmokeTests(unittest.TestCase):
    """One ``setUp`` creates a fresh session per test so case bleed-through
    can't mask state-machine bugs.
    """

    @classmethod
    def setUpClass(cls):
        # Confirm the server is up before any test runs — gives a clean
        # error if the operator forgot to start uvicorn instead of
        # cascading mysterious 502s through every test.
        status, _h, body = _request('GET', '/health')
        if status != 200:
            raise unittest.SkipTest(f'/health unhealthy: {status} {body!r}')

    def setUp(self):
        status, _h, body = _request('POST', '/v1/sessions', body={})
        self.assertEqual(status, 200, body)
        self.session = json.loads(body)
        self.session_id = self.session['session_id']

    def tearDown(self):
        # Best-effort cleanup; failures here shouldn't fail the test.
        _request('DELETE', f'/v1/sessions/{self.session_id}')

    # ─── health ─────────────────────────────────────────────────────────

    def test_health_endpoints(self):
        status, _h, body = _request('GET', '/health')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['status'], 'ok')

        status, _h, body = _request('GET', '/health/detailed')
        self.assertEqual(status, 200)
        detail = json.loads(body)
        # The whole point of the rewrite — confirm the SDK runner is what's
        # actually serving traffic.
        self.assertEqual(detail['runner_backend'], 'sdk')
        self.assertEqual(detail['checks']['runner'], 'sdk')

    # ─── sessions ───────────────────────────────────────────────────────

    def test_session_lifecycle(self):
        status, _h, body = _request('GET', f'/v1/sessions/{self.session_id}')
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)['status'], 'idle')

        status, _h, body = _request('GET', '/v1/sessions')
        self.assertEqual(status, 200, body)
        listing = json.loads(body)
        self.assertEqual(listing['object'], 'list')
        self.assertTrue(any(s['session_id'] == self.session_id for s in listing['data']))

    # ─── responses ──────────────────────────────────────────────────────

    def test_responses_non_stream_returns_400(self):
        # SDK runner is stream-first; non-stream Responses is intentionally
        # unsupported (documented as ``unsupported_mode``). This guards
        # against a future regression that might silently re-enable a
        # buffered code path with stale semantics.
        status, _h, body = _request('POST', '/v1/responses', body={
            'conversation': self.session_id,
            'input': 'hi',
            'stream': False,
        })
        self.assertEqual(status, 400, body)
        err = json.loads(body)['error']
        self.assertEqual(err['code'], 'unsupported_mode')

    def test_responses_get_after_stream(self):
        # Stream a response, then confirm the saved artifact is fetchable.
        # ``store=true`` is the default — no need to opt in.
        lines = _stream_lines('POST', '/v1/responses', body={
            'conversation': self.session_id,
            'input': 'Reply with the single word: ok',
            'stream': True,
        })
        events = _parse_sse(lines)
        # Find the response.created event to grab the id.
        created = next((json.loads(d) for ev, d in events
                        if ev == 'response.created'), None)
        self.assertIsNotNone(created, lines)
        rid = created['id']

        status, _h, body = _request('GET', f'/v1/responses/{rid}')
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)['id'], rid)

    def test_responses_stream(self):
        lines = _stream_lines('POST', '/v1/responses', body={
            'conversation': self.session_id,
            'input': 'Reply with the single word: ok',
            'stream': True,
        })
        events = _parse_sse(lines)
        types = {ev for ev, _ in events if ev}
        # The minimum SSE skeleton the wire contract guarantees: a created
        # event opens the stream, exactly one terminal closes it. We don't
        # assert ``completed`` specifically — an upstream LLM hiccup can
        # legitimately downgrade to ``response.failed`` and the wire still
        # has to remain well-formed.
        self.assertIn('response.created', types)
        terminal = types & {'response.completed', 'response.failed'}
        self.assertTrue(terminal, lines)
        self.assertEqual(lines[-1], 'data: [DONE]')
        # Every emitted ``response.*`` payload must be valid JSON with
        # a matching ``type`` field — frontends rely on this invariant.
        for ev, data in events:
            if ev and ev.startswith('response.') and data != '[DONE]':
                payload = json.loads(data)
                self.assertEqual(payload.get('type'), ev)

    def test_responses_conversation_chains_multi_turn(self):
        # The React client sends its session id as the Responses conversation
        # id, without using the removed legacy session_id request field.
        for prompt in ('Reply with the word: alpha',
                       'Reply with the word: beta'):
            lines = _stream_lines('POST', '/v1/responses', body={
                'conversation': self.session_id,
                'input': prompt,
                'stream': True,
            })
            events = _parse_sse(lines)
            terminal = {ev for ev, _ in events
                        if ev in ('response.completed', 'response.failed')}
            self.assertTrue(terminal, f'no terminal event in {lines!r}')
            self.assertEqual(lines[-1], 'data: [DONE]')

    def test_previous_response_id_without_resume_payload_continues_conversation(self):
        # Bare ``previous_response_id`` is normal Responses-style
        # continuation. It must start a new response rather than being
        # interpreted as HITL resume.
        lines = _stream_lines('POST', '/v1/responses', body={
            'conversation': self.session_id,
            'input': 'Reply with the word: ok',
            'stream': True,
        })
        first = next((json.loads(d) for ev, d in _parse_sse(lines)
                      if ev == 'response.created'), None)
        self.assertIsNotNone(first, lines)

        lines = _stream_lines('POST', '/v1/responses', body={
            'previous_response_id': first['id'],
            'input': 'follow up',
            'stream': True,
        })
        events = _parse_sse(lines)
        second = next((json.loads(d) for ev, d in events
                       if ev == 'response.created'), None)
        self.assertIsNotNone(second, lines)
        self.assertNotEqual(second['id'], first['id'])
        terminal = {ev for ev, _ in events if ev in ('response.completed', 'response.failed')}
        self.assertTrue(terminal, lines)
        self.assertEqual(lines[-1], 'data: [DONE]')

    # ─── chat completions ──────────────────────────────────────────────

    def test_chat_completions_stream(self):
        lines = _stream_lines('POST', '/v1/chat/completions', body={
            'model': 'pairag-agent',
            'messages': [{'role': 'user', 'content': 'Reply with the word ok'}],
            'stream': True,
        })
        self.assertEqual(lines[-1], 'data: [DONE]')
        chunks = [json.loads(l[len('data: '):]) for l in lines
                  if l.startswith('data: ') and l != 'data: [DONE]']
        self.assertTrue(any(c.get('object') == 'chat.completion.chunk' for c in chunks))
        self.assertTrue(any(
            (c.get('choices') or [{}])[0].get('finish_reason') == 'stop'
            for c in chunks
        ), chunks)

    # ─── runs (deleted) ────────────────────────────────────────────────

    def test_runs_endpoints_all_404(self):
        # The /v1/runs family was deleted in the SDK migration. We don't
        # keep a 410 fallback — every former path must be a clean 404 so
        # routing-layer mistakes can't silently re-introduce the old
        # surface (which would skew clients into two-stage mode).
        for method, path in (
            ('POST', '/v1/runs'),
            ('GET', '/v1/runs/run_x'),
            ('GET', '/v1/runs/run_x/events'),
            ('POST', '/v1/runs/run_x/stop'),
        ):
            status, _h, _body = _request(
                method, path,
                body={} if method == 'POST' else None,
            )
            self.assertEqual(status, 404, f'{method} {path}: expected 404, got {status}')

    # ─── regenerate ────────────────────────────────────────────────────

    def test_regenerate_streams_directly(self):
        # Set the session up with a completed turn so regenerate has
        # something to trim.
        lines = _stream_lines('POST', '/v1/responses', body={
            'conversation': self.session_id,
            'input': 'Reply with the word: ok',
            'stream': True,
        })
        # Confirm we got a terminal — otherwise regenerate has no
        # assistant message to trim.
        terminal = {ev for ev, _ in _parse_sse(lines)
                    if ev in ('response.completed', 'response.failed')}
        self.assertTrue(terminal, lines)

        # The new contract: regenerate returns SSE directly (no JSON
        # 202 + cursor handshake). Content-type and the [DONE] sentinel
        # are the two invariants we pin.
        url = f'{SMOKE_URL}/v1/sessions/{self.session_id}/regenerate'
        req = urllib.request.Request(
            url, data=b'{}', method='POST',
            headers={'Content-Type': 'application/json',
                     'Accept': 'text/event-stream'},
        )
        body_lines: list[str] = []
        with urllib.request.urlopen(req, timeout=SMOKE_TIMEOUT) as resp:
            self.assertEqual(resp.status, 200)
            self.assertTrue(resp.headers.get('content-type', '').startswith('text/event-stream'),
                            resp.headers.get('content-type'))
            for raw in resp:
                line = raw.decode('utf-8', errors='replace').rstrip('\n').rstrip('\r')
                body_lines.append(line)
                if line == 'data: [DONE]' or len(body_lines) >= 2000:
                    break
        self.assertEqual(body_lines[-1], 'data: [DONE]')

    # ─── session detail ────────────────────────────────────────────────

    def test_session_detail_carries_pending_hitl_field(self):
        # The frontend uses ``pending_hitl`` to recover an ``ask_user``
        # state on page reload. The field must always be present (null
        # for an idle session) so the client can rely on its shape.
        status, _h, body = _request('GET', f'/v1/sessions/{self.session_id}')
        self.assertEqual(status, 200, body)
        payload = json.loads(body)
        self.assertIn('pending_hitl', payload)
        # Idle session — no pause should be reported.
        self.assertIsNone(payload['pending_hitl'])

    # ─── models ────────────────────────────────────────────────────────

    def test_models_listing_carries_active_flag(self):
        # /v1/models is the only public read for model state — it embeds
        # ``active_model`` plus per-row ``active`` booleans, so we don't
        # need a separate GET /v1/models/active (which is POST-only by
        # design: setter, not getter).
        status, _h, body = _request('GET', '/v1/models')
        self.assertEqual(status, 200, body)
        models = json.loads(body)
        self.assertEqual(models['object'], 'list')
        self.assertTrue(models['data'])
        self.assertIn('active_model', models)
        actives = [m for m in models['data'] if m.get('active')]
        self.assertEqual(len(actives), 1, models)
        self.assertEqual(actives[0]['id'], models['active_model'])


if __name__ == '__main__':
    unittest.main()
