"""Chat API Tests based on PAI-RAG API documentation.
Reference: https://help.aliyun.com/zh/pai/use-cases/rag-api-interface-for-v0-4-x
"""
import os
from typing import Generator, Any
from fastapi.testclient import TestClient
import pytest
from httpx import Client
import json
from loguru import logger


class TestChatAPI:
    """Test cases for Chat API (OpenAI-compatible)."""

    @pytest.mark.skip(reason="Requires LLM model configuration")
    def test_chat_completions_basic(self, client: Client, test_llm_model: Any):
        """Test POST /v1/chat/completions - Basic chat request."""
        chat_payload = {
            "model": "qwen-plus",  # Should be a configured chat app
            "messages": [
                {
                    "role": "user",
                    "content": "你好"
                }
            ],
            "stream": False
        }
        
        response = client.post("/v1/chat/completions", json=chat_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert "choices" in resp_json
        assert len(resp_json["choices"]) > 0
        assert "message" in resp_json["choices"][0]

    def test_chat_completions_stream(self, client: Client, test_llm_model: Any):
        """Test POST /v1/chat/completions - Streaming chat request."""
        chat_payload = {
            "model": "qwen-plus",
            "messages": [
                {
                    "role": "user",
                    "content": "PAI-RAG 有哪些核心功能？"
                }
            ],
            "stream": True
        }
        
        with client.stream("POST", "/v1/chat/completions", json=chat_payload) as response:
            assert response.status_code == 200
            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        chunks.append(json.loads(data))
            assert len(chunks) > 0
            logger.debug(f"####: {chunks}")
            assert "not found" not in chunks[0]["choices"][0]["delta"]["content"]

    def test_chat_completions_with_history(self, client: Client, test_llm_model: Any):
        """Test chat with conversation history."""
        chat_payload = {
            "model": "qwen-plus",
            "messages": [
                {
                    "role": "user",
                    "content": "我叫小明"
                },
                {
                    "role": "assistant",
                    "content": "你好小明，有什么可以帮助你的吗？"
                },
                {
                    "role": "user",
                    "content": "我叫什么名字？"
                }
            ],
            "stream": False
        }
        
        response = client.post("/v1/chat/completions", json=chat_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert "choices" in resp_json
        assert "not found" not in resp_json["choices"][0]["message"]["content"]
        logger.debug(f"####: {resp_json}")

    def test_chat_completions_invalid_model(self, client: Client):
        """Test chat with non-existent model should fail."""
        chat_payload = {
            "model": "non_existent_model",
            "messages": [
                {
                    "role": "user",
                    "content": "测试"
                }
            ],
            "stream": False
        }
        
        response = client.post("/v1/chat/completions", json=chat_payload)
        # Should return error for non-existent model
        response_json = response.json()
        assert "Model `non_existent_model` not found." in response_json["choices"][0]["message"]["content"]

    def test_chat_completions_missing_messages(self, client: Client):
        """Test chat without messages should fail validation."""
        chat_payload = {
            "model": "qwen-max"
            # Missing messages field
        }
        
        response = client.post("/v1/chat/completions", json=chat_payload)
        assert response.status_code == 400

    def test_chat_completions_empty_messages(self, client: Client):
        """Test chat with empty messages array."""
        chat_payload = {
            "model": "my_assistant",
            "messages": [],
            "stream": False
        }
        
        response = client.post("/v1/chat/completions", json=chat_payload)
        response_json = response.json()
        # Should fail or return error
        assert "Hi, how can I help you." in response_json["choices"][0]["message"]["content"]


class TestChatAppsAPI:
    """Test cases for Chat Apps configuration."""

    def test_list_chat_apps(self, client: Client):
        """Test GET /v1/config/apps - List chat applications."""
        response = client.get("/v1/config/apps")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert "data" in resp_json

    def test_create_chat_app(self, client: Client):
        """Test POST /v1/config/apps - Create a new chat application."""
        create_payload = {
            "app_id": "my_assistant",
            "description": "测试聊天应用",
            "model_id": "qwen-max",
            "system_prompt": "你是一个有帮助的助手",
            "temperature": 0.7
        }
        
        response = client.post("/v1/config/apps", json=create_payload)
        if response.status_code == 200:
            resp_json = response.json()
            assert resp_json["code"] == 200
            app_id = resp_json["data"]["id"]
            # Cleanup
            client.delete(f"/v1/config/apps/{app_id}")

    def test_get_chat_app(self, client: Client):
        """Test GET /v1/config/apps/{app_id} - Get chat app details."""
        # First create an app
        create_payload = {
            "model_id": "qwen-max",
            "app_id": "test_chat_app_get",
            "description": "测试获取应用详情"
        }
        create_response = client.post("/v1/config/apps", json=create_payload)
        
        if create_response.status_code == 200:
            _id = create_response.json()["data"]["id"]
            # Get the app
            get_response = client.get(f"/v1/config/apps?app_id=test_chat_app_get")
            assert get_response.status_code == 200
            
            # Cleanup
            client.delete(f"/v1/config/apps/{_id}")

    def test_delete_chat_app(self, client: Client):
        """Test DELETE /v1/config/apps/{app_id} - Delete chat application."""
        # First create an app
        create_payload = {
            "app_id": "test_chat_app_delete",
            "description": "测试删除应用",
            "model_id": "qwen-max"
        }
        create_response = client.post("/v1/config/apps", json=create_payload)
        
        if create_response.status_code == 200:
            app_id = create_response.json()["data"]["id"]
            
            # Delete the app
            delete_response = client.delete(f"/v1/config/apps/{app_id}")
            assert delete_response.status_code == 200

