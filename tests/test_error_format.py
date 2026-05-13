import unittest

from fastapi.testclient import TestClient

from backend.agent_service import NoRegeneratableAnswerError
import backend.server as server


class FakeRegenerateService:
    def __init__(self, error=None):
        self.error = error

    def regenerate_session(self, session_id, user_id):
        if self.error:
            raise self.error


class ErrorFormatTests(unittest.TestCase):
    def setUp(self):
        self.original_service = server.service
        self.original_celery_service = server.celery_service
        server.celery_service = None
        self.client = TestClient(server.app)

    def tearDown(self):
        server.service = self.original_service
        server.celery_service = self.original_celery_service

    def _assert_error_shape(self, body, code=None):
        self.assertIn("error", body, body)
        err = body["error"]
        self.assertIn("message", err)
        self.assertIn("type", err)
        self.assertIn("code", err)
        if code is not None:
            self.assertEqual(err["code"], code)

    def test_plain_http_exception_with_string_detail_is_wrapped(self):
        response = self.client.post("/v1/sessions/no-such-session/cancel")

        self.assertEqual(response.status_code, 404)
        body = response.json()
        self._assert_error_shape(body)
        self.assertEqual(body["error"]["message"], "Session not found")

    def test_backend_error_dict_detail_is_wrapped_into_error(self):
        server.service = FakeRegenerateService(error=NoRegeneratableAnswerError("session-1"))

        response = self.client.post("/v1/sessions/session-1/regenerate", json={})

        self.assertEqual(response.status_code, 409)
        self._assert_error_shape(response.json(), code="no_regeneratable_answer")

    def test_invalid_json_body_uses_unified_error_shape(self):
        response = self.client.post(
            "/v1/chat/completions",
            data="not json",
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self._assert_error_shape(response.json())


if __name__ == "__main__":
    unittest.main()
