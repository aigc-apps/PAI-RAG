import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.injection import get_thread_service, get_message_service, get_llm_service
from app.main import app


class TestThreadAPI:
    @pytest.fixture(autouse=True)
    def setup_thread_services(self, api_client):
        self.mock_thread_service = AsyncMock()
        self.mock_message_service = AsyncMock()
        self.mock_llm_service = AsyncMock()
        app.dependency_overrides[get_thread_service] = lambda: self.mock_thread_service
        app.dependency_overrides[get_message_service] = lambda: self.mock_message_service
        app.dependency_overrides[get_llm_service] = lambda: self.mock_llm_service
        self.client = api_client

    def _make_thread_entity(self, thread_id="t1", title="Test Thread"):
        entity = MagicMock()
        entity.id = thread_id
        entity.title = title
        entity.user_id = "user1"
        entity.tenant_id = "test-tenant"
        entity.created_at = "2024-01-01T00:00:00Z"
        entity.updated_at = "2024-01-01T00:00:00Z"
        entity.model_dump = MagicMock(return_value={
            "id": thread_id,
            "title": title,
            "user_id": "user1",
            "tenant_id": "test-tenant",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        })
        entity.__class__ = type("ThreadEntity", (), {})
        # Support model_validate
        return entity

    def test_create_thread_success(self, mock_session):
        mock_entity = self._make_thread_entity()
        self.mock_thread_service.create_thread.return_value = mock_entity
        mock_session.refresh = AsyncMock()

        response = self.client.post("/v1/threads", json={
            "title": "Test Thread",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_create_thread_value_error(self, mock_session):
        self.mock_thread_service.create_thread.side_effect = ValueError("invalid")
        response = self.client.post("/v1/threads", json={
            "title": "Test Thread",
        })
        assert response.status_code == 400

    def test_create_thread_error(self, mock_session):
        self.mock_thread_service.create_thread.side_effect = Exception("db error")
        response = self.client.post("/v1/threads", json={
            "title": "Test Thread",
        })
        assert response.status_code == 400

    def test_get_threads_success(self):
        mock_entity = self._make_thread_entity()
        self.mock_thread_service.list_threads.return_value = [mock_entity]

        response = self.client.get("/v1/threads")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_get_threads_error(self):
        self.mock_thread_service.list_threads.side_effect = Exception("db error")
        response = self.client.get("/v1/threads")
        assert response.status_code == 400

    def test_delete_thread_success(self, mock_session):
        self.mock_thread_service.delete_thread.return_value = None
        self.mock_message_service.delete_related_attachments.return_value = None

        response = self.client.delete("/v1/threads/t1")
        assert response.status_code == 200

    def test_delete_thread_not_found(self, mock_session):
        self.mock_message_service.delete_related_attachments.return_value = None
        self.mock_thread_service.delete_thread.side_effect = ValueError("Thread not found")

        response = self.client.delete("/v1/threads/nonexistent")
        assert response.status_code == 404

    def test_delete_thread_error(self, mock_session):
        self.mock_message_service.delete_related_attachments.return_value = None
        self.mock_thread_service.delete_thread.side_effect = Exception("db error")

        response = self.client.delete("/v1/threads/t1")
        assert response.status_code == 400

    def test_get_thread_messages_success(self):
        mock_msg = MagicMock()
        mock_msg.id = "m1"
        mock_msg.thread_id = "t1"
        mock_msg.role = "user"
        mock_msg.content = [{"type": "text", "text": "hello"}]
        mock_msg.attachments = []
        mock_msg.local_id = "local1"
        mock_msg.token_usage = None
        mock_msg.created_at = "2024-01-01T00:00:00Z"
        mock_msg.updated_at = "2024-01-01T00:00:00Z"
        self.mock_message_service.list_messages.return_value = [mock_msg]

        response = self.client.get("/v1/threads/t1/messages")
        assert response.status_code == 200

    def test_get_thread_messages_error(self):
        self.mock_message_service.list_messages.side_effect = Exception("db error")
        response = self.client.get("/v1/threads/t1/messages")
        assert response.status_code == 400

    def test_create_thread_message_success(self, mock_session):
        mock_thread = self._make_thread_entity()
        self.mock_thread_service.get_thread.return_value = mock_thread

        mock_msg = MagicMock()
        mock_msg.model_dump = MagicMock(return_value={
            "id": "m1",
            "thread_id": "t1",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        })
        mock_msg.__class__ = type("MessageEntity", (), {})
        self.mock_message_service.create_message.return_value = mock_msg
        mock_session.refresh = AsyncMock()

        response = self.client.post("/v1/threads/t1/messages", json={
            "thread_id": "t1",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        })
        assert response.status_code == 200

    def test_create_thread_message_thread_not_found(self, mock_session):
        self.mock_thread_service.get_thread.return_value = None

        response = self.client.post("/v1/threads/t1/messages", json={
            "thread_id": "t1",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        })
        assert response.status_code == 404
