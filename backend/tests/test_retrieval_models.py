# ruff: noqa: E402
"""DashScope-native embedding/rerank client tests — offline, against a fake
`httpx.AsyncClient` (no network). Asserts request-body shape (model / input /
parameters, batching ≤10) and response parsing (embeddings placed back by
text_index, rerank results parsed in server order)."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.retrieval_models as rm
from app.retrieval_models import (
    DashScopeEmbedder,
    DashScopeReranker,
    LocalHashEmbedder,
    OpenAICompatibleEmbedder,
    OpenAICompatibleReranker,
    build_embedder,
)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, responder, calls, **_kw):
        self._responder = responder
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def post(self, url, headers=None, json=None):
        self._calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResp(self._responder(json))


def _install(monkeypatch, responder):
    calls: list = []
    monkeypatch.setattr(
        rm.httpx, "AsyncClient", lambda **kw: _FakeClient(responder, calls, **kw)
    )
    return calls


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def test_embed_batches_by_ten_and_places_by_text_index(monkeypatch):
    def responder(body):
        texts = body["input"]["texts"]
        # Return embeddings in REVERSED order to prove they're re-placed by
        # text_index, not by response order. Each vector encodes its text ("t<n>").
        entries = [
            {"text_index": i, "embedding": [float(int(t[1:]))]}
            for i, t in enumerate(texts)
        ]
        return {"output": {"embeddings": list(reversed(entries))}}

    calls = _install(monkeypatch, responder)
    emb = DashScopeEmbedder(
        base_url="https://ds/emb", api_key="k", model="text-embedding-v4", dimension=1024
    )
    texts = [f"t{n}" for n in range(23)]
    vecs = asyncio.run(emb.embed(texts, text_type="document"))

    # 23 texts -> 3 requests of 10, 10, 3
    assert [len(c["json"]["input"]["texts"]) for c in calls] == [10, 10, 3]
    # request body shape
    first = calls[0]["json"]
    assert first["model"] == "text-embedding-v4"
    assert first["parameters"] == {"dimension": 1024, "text_type": "document"}
    assert calls[0]["headers"]["Authorization"] == "Bearer k"
    # response re-placed correctly despite reversed order
    assert vecs == [[float(n)] for n in range(23)]


def test_embed_query_text_type_and_empty(monkeypatch):
    def responder(body):
        return {
            "output": {
                "embeddings": [
                    {"text_index": i, "embedding": [1.0]}
                    for i, _ in enumerate(body["input"]["texts"])
                ]
            }
        }

    calls = _install(monkeypatch, responder)
    emb = DashScopeEmbedder(base_url="u", api_key="k", model="m", dimension=8)
    assert asyncio.run(emb.embed([], text_type="query")) == []
    assert calls == []  # no request for an empty batch
    asyncio.run(emb.embed(["hi"], text_type="query"))
    assert calls[0]["json"]["parameters"]["text_type"] == "query"


def test_embed_raises_when_vector_missing(monkeypatch):
    _install(monkeypatch, lambda body: {"output": {"embeddings": []}})
    emb = DashScopeEmbedder(base_url="u", api_key="k", model="m", dimension=8)
    try:
        asyncio.run(emb.embed(["a", "b"]))
        assert False, "expected RuntimeError"
    except RuntimeError as ex:
        assert "no vector" in str(ex)


def test_dashscope_embed_runs_api_batches_concurrently_and_reuses_client():
    class ConcurrentClient:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.calls = []
            self.closed = False

        async def post(self, url, headers=None, json=None):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(json)
            await asyncio.sleep(0.01)
            self.active -= 1
            entries = [
                {
                    "text_index": i,
                    "embedding": [float(text.removeprefix("t"))],
                }
                for i, text in enumerate(json["input"]["texts"])
            ]
            return _FakeResp({"output": {"embeddings": entries}})

        async def aclose(self):
            self.closed = True

    async def scenario():
        client = ConcurrentClient()
        embedder = DashScopeEmbedder(
            base_url="https://ds/emb",
            api_key="k",
            model="text-embedding-v4",
            dimension=8,
            concurrency_gate=asyncio.Semaphore(2),
            client=client,
        )

        vectors = await embedder.embed([f"t{i}" for i in range(25)])

        assert vectors == [[float(i)] for i in range(25)]
        assert sorted(len(call["input"]["texts"]) for call in client.calls) == [5, 10, 10]
        assert client.max_active == 2
        await embedder.aclose()
        assert client.closed is False  # injected clients remain caller-owned

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Rerank
# --------------------------------------------------------------------------- #
def test_rerank_parses_results_and_sends_top_n(monkeypatch):
    def responder(body):
        docs = body["input"]["documents"]
        # server returns best-first, index pointing into input docs
        return {
            "output": {
                "results": [
                    {"index": len(docs) - 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            }
        }

    calls = _install(monkeypatch, responder)
    rr = DashScopeReranker(base_url="https://ds/rr", api_key="k", model="qwen3-rerank")
    out = asyncio.run(rr.rerank("q", ["a", "b", "c"], top_n=2))

    body = calls[0]["json"]
    assert body["model"] == "qwen3-rerank"
    assert body["input"]["query"] == "q"
    assert body["parameters"] == {"return_documents": False, "top_n": 2}
    assert out == [(2, 0.9), (0, 0.1)]


def test_rerank_empty_documents_no_request(monkeypatch):
    calls = _install(monkeypatch, lambda body: {"output": {"results": []}})
    rr = DashScopeReranker(base_url="u", api_key="k", model="m")
    assert asyncio.run(rr.rerank("q", [])) == []
    assert calls == []


# --------------------------------------------------------------------------- #
# OpenAI-compatible embedding + rerank (every non-DashScope provider)
# --------------------------------------------------------------------------- #
def test_compatible_embed_appends_endpoint_and_parses_data(monkeypatch):
    def responder(body):
        # openai response shape: data[].{index, embedding}, returned reversed
        entries = [
            {"index": i, "embedding": [float(int(t[1:]))]}
            for i, t in enumerate(body["input"])
        ]
        return {"data": list(reversed(entries))}

    calls = _install(monkeypatch, responder)
    emb = OpenAICompatibleEmbedder(
        base_url="https://api.vendor.com/v1", api_key="k", model="bge-m3", dimension=512
    )
    vecs = asyncio.run(emb.embed([f"t{n}" for n in range(3)]))
    assert calls[0]["url"] == "https://api.vendor.com/v1/embeddings"
    body = calls[0]["json"]
    assert body["model"] == "bge-m3" and body["input"] == ["t0", "t1", "t2"]
    assert body["dimensions"] == 512 and body["encoding_format"] == "float"
    assert vecs == [[0.0], [1.0], [2.0]]


def test_compatible_embed_omits_dimensions_when_unset(monkeypatch):
    calls = _install(
        monkeypatch,
        lambda body: {"data": [{"index": i, "embedding": [1.0]} for i, _ in enumerate(body["input"])]},
    )
    emb = OpenAICompatibleEmbedder(base_url="https://v/v1", api_key="k", model="m")
    asyncio.run(emb.embed(["a"]))
    assert "dimensions" not in calls[0]["json"]


def test_compatible_rerank_flat_body_and_endpoint(monkeypatch):
    def responder(body):
        return {"results": [{"index": 2, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}]}

    calls = _install(monkeypatch, responder)
    rr = OpenAICompatibleReranker(base_url="https://api.jina.ai/v1", api_key="k", model="jina-reranker")
    out = asyncio.run(rr.rerank("q", ["a", "b", "c"], top_n=2))
    assert calls[0]["url"] == "https://api.jina.ai/v1/rerank"
    body = calls[0]["json"]
    # flat body (not nested under input/parameters like DashScope)
    assert body == {"model": "jina-reranker", "query": "q", "documents": ["a", "b", "c"],
                    "return_documents": False, "top_n": 2}
    assert out == [(2, 0.8), (0, 0.2)]


# --------------------------------------------------------------------------- #
# Local fallback + factory
# --------------------------------------------------------------------------- #
def test_local_hash_embedder_dimension_and_values():
    emb = LocalHashEmbedder(dimension=32)
    vecs = asyncio.run(emb.embed(["hello world", "hello world"]))
    assert emb.dimension == 32
    assert len(vecs) == 2 and len(vecs[0]) == 32
    assert vecs[0] == vecs[1]  # deterministic


class _StubRouter:
    def __init__(self):
        self.asked = None

    def get_embedder(self, model_id):
        self.asked = model_id
        return "DASHSCOPE_EMBEDDER"


def test_build_embedder_dispatch():
    router = _StubRouter()
    # dashscope config -> router.get_embedder
    got = build_embedder(
        {"provider_id": "dashscope", "model": "dashscope/text-embedding-v4"}, router
    )
    assert got == "DASHSCOPE_EMBEDDER" and router.asked == "dashscope/text-embedding-v4"
    # local / legacy config -> LocalHashEmbedder at the stored dimension
    local = build_embedder({"provider_id": "local_hash", "dimension": 64}, router)
    assert isinstance(local, LocalHashEmbedder) and local.dimension == 64
    # no router -> local, even if config says dashscope
    fallback = build_embedder({"provider_id": "dashscope", "model": "x"}, None)
    assert isinstance(fallback, LocalHashEmbedder)
