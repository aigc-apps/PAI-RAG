import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import server  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = text

    def json(self):
        return self._data

    async def aread(self):
        return self.text.encode()


class FakeStream:
    def __init__(self, lines):
        self.status_code = 200
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


class FakeOpenClawClient:
    def __init__(self):
        self.requests = []

    async def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        if url.endswith("/api/auth/csrf"):
            return FakeResponse(data={"csrfToken": "csrf_test"})
        if "/api/sessions/" in url and url.endswith("/messages?limit=10"):
            return FakeResponse(data=[
                {"role": "USER", "content": "ping"},
                {"role": "ASSISTANT", "content": "OK"},
            ])
        raise AssertionError(f"unexpected GET {url}")

    async def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        if url.endswith("/api/auth/register"):
            return FakeResponse(data={"ok": True})
        if url.endswith("/api/auth/callback/credentials"):
            return FakeResponse(data={"url": "http://localhost:3000"})
        if url.endswith("/api/sessions"):
            return FakeResponse(data={"id": "sess_123", "title": "AgentArena"})
        raise AssertionError(f"unexpected POST {url}")

    def stream(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        if method == "POST" and url.endswith("/api/chat"):
            return FakeStream([
                "event: started",
                'data: {"assistantMessageId":"msg_1","runStartedAt":"2026-05-27T08:31:10Z"}',
                "",
                "event: status",
                'data: {"phase":"thinking"}',
                "",
                "event: token",
                'data: {"text":"O"}',
                "",
                "event: token",
                'data: {"text":"K"}',
                "",
                "event: done",
                'data: {"messageId":"msg_1","runState":"DONE"}',
                "",
            ])
        raise AssertionError(f"unexpected stream {method} {url}")


class OpenClawAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_openclaw_mode_logs_in_creates_session_streams_trace_and_content(self):
        client = FakeOpenClawClient()
        agent = server.AgentConfig(
            name="OpenClaw",
            base_url="http://openclaw.test",
            api_key="",
            model="openclaw",
            trace_mode="openclaw",
            runs_base_url="",
        )

        result = await server.call_agent(
            client,
            agent,
            [{"role": "user", "content": "ping"}],
            temperature=0.2,
            max_tokens=32,
        )

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.content, "OK")
        self.assertTrue(result.trace_supported)
        self.assertEqual(result.raw_finish_reason, "stop")
        self.assertIn("run.completed", [event.event for event in result.trace_events])
        self.assertIn("message.delta", [event.event for event in result.trace_events])
        self.assertEqual(
            [method for method, url, _kwargs in client.requests if url.endswith("/api/chat")],
            ["POST"],
        )

    async def test_openclaw_auth_error_redacts_email_and_password(self):
        text = "login failed for user@example.com with password Secret123"

        redacted = server.redact_openclaw_auth_text(text, "user@example.com", "Secret123")

        self.assertNotIn("user@example.com", redacted)
        self.assertNotIn("Secret123", redacted)
        self.assertIn("[redacted-email]", redacted)
        self.assertIn("[redacted-password]", redacted)


if __name__ == "__main__":
    unittest.main()
