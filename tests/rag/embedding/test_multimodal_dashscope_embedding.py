"""MultimodalDashscopeEmbedding 单元测试。"""
import pytest
from unittest.mock import patch, AsyncMock
from rag.embedding.multimodal_dashscope_embedding import MultimodalDashscopeEmbedding


def test_endpoint_normalization():
    e1 = MultimodalDashscopeEmbedding(api_key="sk", base_url="")
    assert e1._endpoint.endswith("/multimodal-embedding/multimodal-embedding")

    e2 = MultimodalDashscopeEmbedding(
        api_key="sk",
        base_url="https://example.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
    )
    assert e2._endpoint == (
        "https://example.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    )

    e3 = MultimodalDashscopeEmbedding(
        api_key="sk",
        base_url="https://example.com/api/v1/services/embeddings/multimodal-embedding",
    )
    assert e3._endpoint.endswith("/multimodal-embedding/multimodal-embedding")


def test_authorization_header():
    e1 = MultimodalDashscopeEmbedding(api_key="sk-xxx")
    assert e1._headers["Authorization"] == "Bearer sk-xxx"

    e2 = MultimodalDashscopeEmbedding(api_key="Bearer sk-yyy")
    assert e2._headers["Authorization"] == "Bearer sk-yyy"


def test_build_payload_fusion_capable_model():
    e = MultimodalDashscopeEmbedding(
        api_key="sk", model_name="qwen3-vl-embedding", dimension=1024
    )
    payload = e._build_payload(
        contents=[{"text": "hi"}, {"image": "http://x/y.png"}],
        prefer_fusion=True,
    )
    assert payload["model"] == "qwen3-vl-embedding"
    assert payload["input"]["contents"] == [{"text": "hi"}, {"image": "http://x/y.png"}]
    assert payload["parameters"]["enable_fusion"] is True
    assert payload["parameters"]["dimension"] == 1024


def test_build_payload_non_fusion_model_drops_enable_fusion():
    e = MultimodalDashscopeEmbedding(
        api_key="sk", model_name="tongyi-embedding-vision-plus"
    )
    payload = e._build_payload(
        contents=[{"text": "hi"}], prefer_fusion=True
    )
    assert "enable_fusion" not in payload.get("parameters", {})


def test_pick_vector_priority():
    embeddings = [
        {"index": 0, "embedding": [1.0], "type": "text"},
        {"index": 1, "embedding": [2.0], "type": "image"},
        {"index": 2, "embedding": [3.0], "type": "fusion"},
    ]
    assert MultimodalDashscopeEmbedding._pick_vector(embeddings, ["fusion"]) == [3.0]
    assert MultimodalDashscopeEmbedding._pick_vector(embeddings, ["text"]) == [1.0]
    # fallback to first
    assert MultimodalDashscopeEmbedding._pick_vector(embeddings, ["nonexistent"]) == [1.0]


def test_pick_vector_empty_raises():
    with pytest.raises(RuntimeError):
        MultimodalDashscopeEmbedding._pick_vector([], ["text"])


@pytest.mark.asyncio
async def test_aget_text_embedding_invokes_api():
    e = MultimodalDashscopeEmbedding(api_key="sk", model_name="qwen3-vl-embedding")
    fake_response = {
        "output": {
            "embeddings": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3], "type": "text"}
            ]
        }
    }
    with patch.object(e, "_post", new=AsyncMock(return_value=fake_response)) as m:
        v = await e.aget_text_embedding("hello world")
        assert v == [0.1, 0.2, 0.3]
        m.assert_awaited_once()
        called_payload = m.await_args.args[0]
        assert called_payload["input"]["contents"] == [{"text": "hello world"}]
        # text-only path should not request fusion
        assert "enable_fusion" not in called_payload.get("parameters", {})


@pytest.mark.asyncio
async def test_aget_multimodal_embedding_with_images_fusion():
    e = MultimodalDashscopeEmbedding(api_key="sk", model_name="qwen3-vl-embedding")
    fake_response = {
        "output": {
            "embeddings": [
                {"index": 0, "embedding": [9.0, 9.0], "type": "fusion"}
            ]
        }
    }
    with patch.object(e, "_post", new=AsyncMock(return_value=fake_response)) as m:
        v = await e.aget_multimodal_embedding(
            text="caption",
            images=["http://x/y.png", "http://x/z.png"],
        )
        assert v == [9.0, 9.0]
        called_payload = m.await_args.args[0]
        assert called_payload["parameters"]["enable_fusion"] is True
        assert called_payload["input"]["contents"] == [
            {"text": "caption"},
            {"image": "http://x/y.png"},
            {"image": "http://x/z.png"},
        ]


@pytest.mark.asyncio
async def test_aget_multimodal_embedding_truncates_images_over_qwen3_vl_cap():
    """qwen3-vl-embedding 单请求最多 5 张图，超出应截断而非抛错。"""
    e = MultimodalDashscopeEmbedding(api_key="sk", model_name="qwen3-vl-embedding")
    fake_response = {
        "output": {
            "embeddings": [{"index": 0, "embedding": [1.0], "type": "fusion"}]
        }
    }
    with patch.object(e, "_post", new=AsyncMock(return_value=fake_response)) as m:
        v = await e.aget_multimodal_embedding(
            text="caption",
            images=[f"http://x/{i}.png" for i in range(8)],
        )
        assert v == [1.0]
        called_payload = m.await_args.args[0]
        image_items = [c for c in called_payload["input"]["contents"] if "image" in c]
        assert len(image_items) == 5


def test_truncate_modalities_respects_total_element_cap():
    """text + image + video 总数不能超过 20。"""
    e = MultimodalDashscopeEmbedding(api_key="sk", model_name="qwen3-vl-embedding")
    imgs, vids = e._truncate_modalities(
        images=[f"i{i}" for i in range(50)],
        videos=[f"v{i}" for i in range(5)],
        has_text=True,
    )
    # image 先被截到 5（qwen3-vl-embedding 单请求上限），再受总元素 20 - 1(text) - 3(video) 限制
    assert len(imgs) <= 5
    assert len(vids) <= 3
    assert len(imgs) + len(vids) + 1 <= 20


@pytest.mark.asyncio
async def test_aget_multimodal_embedding_batch_returns_per_item_vector():
    e = MultimodalDashscopeEmbedding(api_key="sk", model_name="qwen3-vl-embedding")

    async def fake_single(self, text=None, images=None, videos=None):
        return [len(text or ""), len(images or [])]

    with patch.object(
        MultimodalDashscopeEmbedding,
        "aget_multimodal_embedding",
        new=fake_single,
    ):
        items = [
            {"text": "a", "images": []},
            {"text": "bb", "images": ["u1", "u2"]},
        ]
        out = await e.aget_multimodal_embedding_batch(items)
        assert out == [[1, 0], [2, 2]]
