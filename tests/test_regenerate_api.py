import unittest

from fastapi.testclient import TestClient

from backend.auth import AuthContext
from backend.agent_service import NoRegeneratableAnswerError
import backend.server as server


class FakeRegenerateService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def regenerate_session(self, session_id, user_id):
        self.calls.append((session_id, user_id))
        if self.error:
            raise self.error
        return self.result


class RegenerateApiTests(unittest.TestCase):
    def setUp(self):
        self.original_service = server.service
        self.original_celery_service = server.celery_service
        self.original_thread_runs = dict(server.THREAD_RUNS)
        server.celery_service = None
        server.THREAD_RUNS.clear()
        server.app.dependency_overrides[server.require_auth] = lambda: AuthContext(
            user_id="user-1",
            username="tester",
        )
        self.client = TestClient(server.app)

    def tearDown(self):
        server.service = self.original_service
        server.celery_service = self.original_celery_service
        server.THREAD_RUNS.clear()
        server.THREAD_RUNS.update(self.original_thread_runs)
        server.app.dependency_overrides.pop(server.require_auth, None)

    def test_regenerate_endpoint_returns_run_payload_and_headers(self):
        fake_service = FakeRegenerateService({
            "session_id": "session-1",
            "run_id": "run-2",
            "stream_from": "0-0",
            "regenerated_from_run_id": "run-1",
        })
        server.service = fake_service

        response = self.client.post("/v1/sessions/session-1/regenerate", json={})

        self.assertEqual(response.status_code, 202)
        self.assertEqual(fake_service.calls, [("session-1", "user-1")])
        self.assertEqual(response.headers["X-Session-Id"], "session-1")
        self.assertEqual(response.headers["X-Run-Id"], "run-2")
        self.assertEqual(response.json(), {
            "id": "run-2",
            "object": "agent.run",
            "run_id": "run-2",
            "session_id": "session-1",
            "status": "started",
            "stream_from": "0-0",
            "regenerated_from_run_id": "run-1",
        })
        self.assertEqual(server.THREAD_RUNS["run-2"]["session_id"], "session-1")

    def test_regenerate_endpoint_returns_409_when_no_answer_exists(self):
        server.service = FakeRegenerateService(error=NoRegeneratableAnswerError("session-1"))

        response = self.client.post("/v1/sessions/session-1/regenerate", json={})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "no_regeneratable_answer")


if __name__ == "__main__":
    unittest.main()
