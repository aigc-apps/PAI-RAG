"""Integration tests for thread and message CRUD."""

import pytest
from tests.integration.conftest import pytestmark  # noqa: F401


class TestThreadCrud:
    """Tests for thread CRUD via /v1/threads."""

    def test_create_thread(self, client):
        """Create a thread and verify response."""
        payload = {"user_id": "integ-test-user", "title": "Test Thread"}
        response = client.post("/v1/threads", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        thread = data["data"]
        assert "id" in thread

        # Cleanup
        client.delete(f"/v1/threads/{thread['id']}")

    def test_create_thread_with_title(self, client):
        """Create a thread with a custom title."""
        payload = {"user_id": "integ-test-user", "title": "My Test Thread"}
        response = client.post("/v1/threads", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        thread = data["data"]
        assert thread.get("title") == "My Test Thread"

        # Cleanup
        client.delete(f"/v1/threads/{thread['id']}")

    def test_list_threads(self, client):
        """Create a thread then verify it appears in the list."""
        # Create
        payload = {"user_id": "integ-test-user", "title": "List Test Thread"}
        create_resp = client.post("/v1/threads", json=payload)
        assert create_resp.status_code == 200
        thread_id = create_resp.json()["data"]["id"]

        # List
        response = client.get("/v1/threads")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        threads = data["data"]
        assert isinstance(threads, list)
        thread_ids = [t["id"] for t in threads]
        assert thread_id in thread_ids

        # Cleanup
        client.delete(f"/v1/threads/{thread_id}")

    def test_delete_thread(self, client):
        """Create then delete a thread."""
        payload = {"user_id": "integ-test-user", "title": "Delete Test Thread"}
        create_resp = client.post("/v1/threads", json=payload)
        assert create_resp.status_code == 200
        thread_id = create_resp.json()["data"]["id"]

        delete_resp = client.delete(f"/v1/threads/{thread_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["code"] == 200

    def test_delete_nonexistent_thread(self, client):
        """Deleting a non-existent thread should return an error."""
        response = client.delete("/v1/threads/nonexistent-thread-id-99999")
        data = response.json()
        assert response.status_code != 200 or data.get("code") != 200


class TestMessageCrud:
    """Tests for message CRUD within threads."""

    @pytest.fixture()
    def thread(self, client):
        """Create a thread for message tests, clean up after."""
        payload = {"user_id": "integ-test-user", "title": "Message Test Thread"}
        resp = client.post("/v1/threads", json=payload)
        assert resp.status_code == 200, f"Failed to create thread: {resp.json()}"
        thread_data = resp.json()["data"]
        yield thread_data
        client.delete(f"/v1/threads/{thread_data['id']}")

    def test_create_message(self, client, thread):
        """Create a message in a thread."""
        payload = {
            "thread_id": thread["id"],
            "role": "user",
            "content": [{"type": "text", "text": "Hello from integration test"}],
        }
        response = client.post(
            f"/v1/threads/{thread['id']}/messages", json=payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        message = data["data"]
        assert "id" in message
        assert message["role"] == "user"

    def test_list_messages(self, client, thread):
        """Create a message then list messages in the thread."""
        # Create a message
        payload = {
            "thread_id": thread["id"],
            "role": "user",
            "content": [{"type": "text", "text": "Test message for listing"}],
        }
        client.post(f"/v1/threads/{thread['id']}/messages", json=payload)

        # List messages
        response = client.get(f"/v1/threads/{thread['id']}/messages")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        messages = data["data"]
        assert isinstance(messages, list)
        assert len(messages) >= 1

    def test_create_assistant_message(self, client, thread):
        """Create an assistant-role message."""
        payload = {
            "thread_id": thread["id"],
            "role": "assistant",
            "content": [{"type": "text", "text": "I am the assistant response."}],
        }
        response = client.post(
            f"/v1/threads/{thread['id']}/messages", json=payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["role"] == "assistant"
