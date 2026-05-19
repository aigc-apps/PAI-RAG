import asyncio
import json
import unittest

from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.agent_service import NoRegeneratableAnswerError
import backend.server as server


class ErrorFormatTests(unittest.TestCase):
    def _assert_error_shape(self, body, code=None):
        self.assertIn("error", body, body)
        err = body["error"]
        self.assertIn("message", err)
        self.assertIn("type", err)
        self.assertIn("code", err)
        if code is not None:
            self.assertEqual(err["code"], code)

    def test_plain_http_exception_with_string_detail_is_wrapped(self):
        response = asyncio.run(server.http_exception_handler(
            None,
            StarletteHTTPException(status_code=404, detail="Session not found"),
        ))

        self.assertEqual(response.status_code, 404)
        body = json.loads(response.body)
        self._assert_error_shape(body)
        self.assertEqual(body["error"]["message"], "Session not found")

    def test_backend_error_dict_detail_is_wrapped_into_error(self):
        exc = server.backend_error(NoRegeneratableAnswerError("session-1"))
        response = asyncio.run(server.http_exception_handler(None, exc))

        self.assertEqual(response.status_code, 409)
        self._assert_error_shape(json.loads(response.body), code="no_regeneratable_answer")

    def test_invalid_json_body_uses_unified_error_shape(self):
        class _BadJsonRequest:
            headers = {}

            async def json(self):
                raise ValueError("invalid")

        response = asyncio.run(server.create_response(_BadJsonRequest()))

        self.assertEqual(response.status_code, 400)
        self._assert_error_shape(json.loads(response.body))


if __name__ == "__main__":
    unittest.main()
