import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

from rag.rerank.reranker import OpenAICompatibleReranker


class TestEndpointConstruction:
    def test_no_v1_suffix(self):
        reranker = OpenAICompatibleReranker(base_url="http://localhost:8000")
        assert reranker.endpoint == "http://localhost:8000/v1/rerank"

    def test_with_v1_suffix(self):
        reranker = OpenAICompatibleReranker(base_url="http://localhost:8000/v1")
        assert reranker.endpoint == "http://localhost:8000/v1/rerank"

    def test_with_v1_rerank_suffix(self):
        reranker = OpenAICompatibleReranker(base_url="http://localhost:8000/v1/rerank")
        assert reranker.endpoint == "http://localhost:8000/v1/rerank"

    def test_trailing_slash_stripped(self):
        reranker = OpenAICompatibleReranker(base_url="http://localhost:8000/")
        assert reranker.endpoint == "http://localhost:8000/v1/rerank"

    def test_api_key_sets_header(self):
        reranker = OpenAICompatibleReranker(
            base_url="http://localhost:8000",
            api_key="test-key",
        )
        assert reranker.headers["Authorization"] == "test-key"

    def test_no_api_key_no_auth_header(self):
        reranker = OpenAICompatibleReranker(base_url="http://localhost:8000")
        assert "Authorization" not in reranker.headers


class TestQwen3ModelDetection:
    def test_qwen3_reranker_8b(self):
        reranker = OpenAICompatibleReranker(
            base_url="http://localhost:8000",
            model="Qwen3-Reranker-8B",
        )
        assert reranker.model in OpenAICompatibleReranker.QWEN3_RERANKER_MODELS

    def test_qwen3_reranker_4b(self):
        reranker = OpenAICompatibleReranker(
            base_url="http://localhost:8000",
            model="Qwen3-Reranker-4B",
        )
        assert reranker.model in OpenAICompatibleReranker.QWEN3_RERANKER_MODELS

    def test_regular_model_not_qwen3(self):
        reranker = OpenAICompatibleReranker(
            base_url="http://localhost:8000",
            model="BAAI/bge-reranker-base",
        )
        assert reranker.model not in OpenAICompatibleReranker.QWEN3_RERANKER_MODELS
