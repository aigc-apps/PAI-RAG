"""Integration tests for chat completions with real DashScope LLM."""

import json
import pytest
from tests.integration.conftest import pytestmark  # noqa: F401


class TestChatCompletions:
    """Tests for POST /v1/chat/completions with real qwen3.5-plus."""

    def test_non_streaming_chat(self, client, test_llm_model):
        """Non-streaming chat returns a valid OpenAI-format response."""
        payload = {
            "model": test_llm_model["model_id"],
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "stream": False,
            "enable_input_guardrail": False,
            "enable_output_guardrail": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]
        assert len(choice["message"]["content"]) > 0

    def test_streaming_chat(self, client, test_llm_model):
        """Streaming chat returns SSE events."""
        payload = {
            "model": test_llm_model["model_id"],
            "messages": [{"role": "user", "content": "Say hi."}],
            "stream": True,
            "enable_input_guardrail": False,
            "enable_output_guardrail": False,
        }
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type

            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event_data = line[len("data: "):]
                    if event_data.strip() == "[DONE]":
                        break
                    try:
                        parsed = json.loads(event_data)
                        events.append(parsed)
                    except json.JSONDecodeError:
                        pass

            assert len(events) > 0
            # First event should have choices with delta
            first_event = events[0]
            assert "choices" in first_event

    def test_chat_with_kb_context(self, client, test_llm_model, test_knowledgebase):
        """Chat with KB IDs triggers RAG retrieval."""
        import io

        kb_id = test_knowledgebase["id"]

        # Upload a file and add a chunk for context
        file_content = b"RAG context test file."
        files = {"files": ("rag_chat_test.txt", io.BytesIO(file_content), "text/plain")}
        upload_resp = client.post(
            f"/v1/config/knowledgebases/{kb_id}/files",
            files=files,
            data={"auto_parse": "false"},
        )
        if upload_resp.status_code == 200:
            file_list = upload_resp.json().get("data", [])
            if file_list:
                file_id = file_list[0]["id"] if isinstance(file_list, list) else file_list["id"]
                client.post(
                    f"/v1/config/knowledgebases/{kb_id}/files/{file_id}/chunks",
                    json={
                        "text": "The capital of France is Paris. Paris is known for the Eiffel Tower.",
                        "chunk_metadata": {},
                    },
                )

        payload = {
            "model": test_llm_model["model_id"],
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "stream": False,
            "kb_ids": [kb_id],
            "enable_input_guardrail": False,
            "enable_output_guardrail": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        content = data["choices"][0]["message"]["content"].lower()
        assert "paris" in content

    def test_empty_messages_returns_greeting(self, client, test_llm_model):
        """Empty messages list returns a fallback greeting response."""
        payload = {
            "model": test_llm_model["model_id"],
            "messages": [],
            "stream": False,
            "enable_input_guardrail": False,
            "enable_output_guardrail": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_non_streaming_response_structure(self, client, test_llm_model):
        """Verify the complete response structure of a non-streaming chat."""
        payload = {
            "model": test_llm_model["model_id"],
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
            "enable_input_guardrail": False,
            "enable_output_guardrail": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        # OpenAI-format fields
        assert "choices" in data
        assert "model" in data
        choice = data["choices"][0]
        assert "message" in choice
        assert "role" in choice["message"]
        assert "content" in choice["message"]

    def test_chat_with_system_message(self, client, test_llm_model):
        """Chat with a system message and user message."""
        payload = {
            "model": test_llm_model["model_id"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Reply in exactly one word."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "stream": False,
            "enable_input_guardrail": False,
            "enable_output_guardrail": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_chat_with_temperature(self, client, test_llm_model):
        """Chat with explicit temperature parameter."""
        payload = {
            "model": test_llm_model["model_id"],
            "messages": [{"role": "user", "content": "Say yes."}],
            "stream": False,
            "temperature": 0.0,
            "enable_input_guardrail": False,
            "enable_output_guardrail": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
