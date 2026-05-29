"""MultimodalDashscopeReranker 单元测试。"""
import pytest
from unittest.mock import patch, AsyncMock
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from rag.rerank.multimodal_dashscope_reranker import (
    MultimodalDashscopeReranker,
    _node_documents_with_images,
)
from rag.rerank.reranker import RerankResult


def test_endpoint_normalization():
    r1 = MultimodalDashscopeReranker(api_key="sk", base_url="")
    assert r1.endpoint.endswith("/text-rerank/text-rerank")

    r2 = MultimodalDashscopeReranker(
        api_key="sk",
        base_url="https://example.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
    assert r2.endpoint == "https://example.com/api/v1/services/rerank/text-rerank/text-rerank"

    r3 = MultimodalDashscopeReranker(
        api_key="sk",
        base_url="https://example.com/api/v1/services/rerank/text-rerank",
    )
    assert r3.endpoint.endswith("/text-rerank/text-rerank")


def test_authorization_header():
    r1 = MultimodalDashscopeReranker(api_key="sk-xxx")
    assert r1.headers["Authorization"] == "Bearer sk-xxx"

    r2 = MultimodalDashscopeReranker(api_key="Bearer sk-yyy")
    assert r2.headers["Authorization"] == "Bearer sk-yyy"


def test_node_documents_with_images_pure_text():
    n = TextNode(text="pure text", metadata={"images_info": []})
    docs = _node_documents_with_images([n])
    assert docs == [{"text": "pure text"}]


def test_node_documents_with_images_image_only():
    n = TextNode(
        text="",
        metadata={"images_info": [{"url": "http://x/a.png", "desc": "a"}]},
    )
    docs = _node_documents_with_images([n])
    assert docs == [{"image": "http://x/a.png"}]


def test_node_documents_with_images_text_priority_when_both_present():
    n = TextNode(
        text="text+img",
        metadata={"images_info": [{"url": "http://x/b.png", "desc": "b"}]},
    )
    docs = _node_documents_with_images([n])
    assert docs == [{"text": "text+img"}]


@pytest.mark.asyncio
async def test_rerank_normalizes_string_query_and_documents():
    r = MultimodalDashscopeReranker(api_key="sk", model="qwen3-vl-rerank")
    fake_response = {
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.9, "document": {"text": "doc B"}},
                {"index": 0, "relevance_score": 0.5, "document": {"text": "doc A"}},
            ]
        }
    }

    async def fake_post(self, *args, **kwargs):
        return fake_response

    captured = {}

    async def capture_post(*args, **kwargs):
        # ClientSession.post returns a context manager; we just need to record payload
        return fake_response

    with patch("aiohttp.ClientSession.post") as mock_post:
        # Build a manual context manager that returns our response
        from unittest.mock import MagicMock

        class _Resp:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self):
                return fake_response

        def _post_capture(url, **kwargs):
            captured["url"] = url
            captured["payload"] = kwargs.get("json")
            return _Resp()

        mock_post.side_effect = _post_capture

        results = await r.rerank(
            query="hello",
            documents=["doc A", "doc B"],
            top_n=2,
        )

    assert len(results) == 2
    assert results[0].score == 0.9
    assert results[0].index == 1
    # ensure query/documents normalized to dict form
    payload = captured["payload"]
    assert payload["input"]["query"] == {"text": "hello"}
    assert payload["input"]["documents"] == [{"text": "doc A"}, {"text": "doc B"}]
    assert payload["parameters"]["return_documents"] is True
    assert payload["parameters"]["top_n"] == 2


@pytest.mark.asyncio
async def test_vector_store_rerank_passthrough_on_few_nodes():
    r = MultimodalDashscopeReranker(api_key="sk")
    # 0 or 1 node short-circuits without API call
    n = TextNode(text="solo", metadata={})
    vr = VectorStoreQueryResult(nodes=[n], ids=["x"], similarities=[0.5])
    result = await r.vector_store_rerank(query="q", vector_result=vr)
    assert result is vr


@pytest.mark.asyncio
async def test_vector_store_rerank_uses_node_images():
    r = MultimodalDashscopeReranker(api_key="sk")
    n1 = TextNode(id_="a", text="text doc", metadata={"images_info": []})
    n2 = TextNode(
        id_="b",
        text="",
        metadata={"images_info": [{"url": "http://x/a.png", "desc": ""}]},
    )
    vr = VectorStoreQueryResult(
        nodes=[n1, n2], ids=["a", "b"], similarities=[0.1, 0.1]
    )
    fake_results = [
        RerankResult(index=1, score=0.9, doc="http://x/a.png"),
        RerankResult(index=0, score=0.7, doc="text doc"),
    ]
    with patch.object(r, "rerank", new=AsyncMock(return_value=fake_results)) as m:
        out = await r.vector_store_rerank(query="dog", vector_result=vr, top_n=2)
        assert out.ids == ["b", "a"]
        assert out.similarities == [0.9, 0.7]
        kwargs = m.await_args.kwargs
        assert kwargs["documents"] == [{"text": "text doc"}, {"image": "http://x/a.png"}]
