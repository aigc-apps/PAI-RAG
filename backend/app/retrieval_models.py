"""Retrieval-side model clients: embedding + rerank.

The chat path uses `openai.AsyncOpenAI` (`app/llm.py:LeanLLM`); embedding and
rerank use DashScope's **native** protocol, which is not OpenAI-compatible, so
these are the repo's first raw-HTTP model clients (`httpx.AsyncClient`).

An embedder's contract is ``async embed(texts, *, text_type) -> list[list[float]]``
plus a ``.dimension`` attribute; a reranker's is
``async rerank(query, documents, *, top_n) -> list[tuple[int, float]]``.

`LocalHashEmbedder` wraps the deterministic local hash embedding so old KBs and
offline tests keep a real vector path. `build_embedder` resolves a KB's frozen
``embedding_config`` to the right client, guaranteeing ingest and query use the
same embedder for that KB.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import httpx

# DashScope native embedding dimensions. v4 supports these; we default to 1024.
DASHSCOPE_EMBED_BATCH = 10  # native API caps a single request at 10 texts
DEFAULT_HTTP_TIMEOUT = 60.0


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    async def embed(
        self, texts: Sequence[str], *, text_type: str = "document"
    ) -> List[List[float]]: ...


class DashScopeEmbedder:
    """DashScope-native text embedding (e.g. ``text-embedding-v4``).

    ``POST {base_url}`` with ``{model, input:{texts:[...]}, parameters:{dimension,
    text_type}}``; the response carries ``output.embeddings[].{text_index,
    embedding}``. ``text_index`` is relative to each batch's input order, so
    results are placed back by ``batch_start + text_index``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int = 1024,
        batch_size: int = DASHSCOPE_EMBED_BATCH,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self._batch = max(1, min(batch_size, DASHSCOPE_EMBED_BATCH))
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def embed(
        self, texts: Sequence[str], *, text_type: str = "document"
    ) -> List[List[float]]:
        items = list(texts)
        if not items:
            return []
        out: List[Optional[List[float]]] = [None] * len(items)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for start in range(0, len(items), self._batch):
                batch = items[start : start + self._batch]
                payload = {
                    "model": self.model,
                    "input": {"texts": batch},
                    "parameters": {
                        "dimension": self.dimension,
                        "text_type": text_type,
                    },
                }
                resp = await client.post(
                    self.base_url, headers=self._headers(), json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = (data.get("output") or {}).get("embeddings") or []
                for entry in embeddings:
                    idx = int(entry.get("text_index", 0))
                    out[start + idx] = entry.get("embedding") or []
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RuntimeError(
                f"DashScope embedding returned no vector for {len(missing)} "
                f"of {len(items)} texts (model '{self.model}')"
            )
        return [v for v in out if v is not None]


class DashScopeReranker:
    """DashScope-native rerank (e.g. ``qwen3-rerank``).

    ``POST {base_url}`` with ``{model, input:{query, documents:[...]},
    parameters:{top_n, return_documents:false}}``; the response carries
    ``output.results[].{index, relevance_score}`` (already sorted best-first,
    ``index`` pointing back into the input ``documents``).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        docs = list(documents)
        if not docs:
            return []
        parameters: dict = {"return_documents": False}
        if top_n is not None:
            parameters["top_n"] = int(top_n)
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": docs},
            "parameters": parameters,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                self.base_url, headers=self._headers(), json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        results = (data.get("output") or {}).get("results") or []
        return [
            (int(r["index"]), float(r.get("relevance_score", 0.0)))
            for r in results
            if r.get("index") is not None
        ]


class OpenAICompatibleEmbedder:
    """OpenAI-compatible embeddings (everyone that isn't DashScope-native).

    ``POST {base_url}/embeddings`` with ``{model, input:[...], encoding_format,
    dimensions?}``; the response carries ``data[].{index, embedding}``. ``base_url``
    is the provider's ``/v1`` root (the ``/embeddings`` suffix is appended).
    ``dimension`` is sent as ``dimensions`` only when set (text-embedding-3 &
    compatible servers); left off otherwise so servers that reject it still work.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimension: Optional[int] = None,
        batch_size: int = 32,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self._batch = max(1, batch_size)
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def embed(
        self, texts: Sequence[str], *, text_type: str = "document"
    ) -> List[List[float]]:
        # text_type has no OpenAI equivalent — accepted for interface parity, unused.
        items = list(texts)
        if not items:
            return []
        out: List[Optional[List[float]]] = [None] * len(items)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for start in range(0, len(items), self._batch):
                batch = items[start : start + self._batch]
                payload: dict = {
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float",
                }
                if self.dimension is not None:
                    payload["dimensions"] = self.dimension
                resp = await client.post(
                    self.url, headers=self._headers(), json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                for entry in data.get("data") or []:
                    idx = int(entry.get("index", 0))
                    out[start + idx] = entry.get("embedding") or []
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RuntimeError(
                f"embedding endpoint returned no vector for {len(missing)} of "
                f"{len(items)} texts (model '{self.model}')"
            )
        return [v for v in out if v is not None]


class OpenAICompatibleReranker:
    """Cohere/Jina-style rerank (the de-facto "compatible" rerank shape used by
    Xinference, vLLM, one-api, SiliconFlow, Jina, Cohere).

    ``POST {base_url}/rerank`` with ``{model, query, documents:[...], top_n,
    return_documents:false}``; the response carries
    ``results[].{index, relevance_score}``. ``base_url`` is the provider `/v1` root.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        self.url = base_url.rstrip("/") + "/rerank"
        self.api_key = api_key
        self.model = model
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        docs = list(documents)
        if not docs:
            return []
        payload: dict = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = int(top_n)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self.url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results") or []
        return [
            (int(r["index"]), float(r.get("relevance_score", 0.0)))
            for r in results
            if r.get("index") is not None
        ]


class LocalHashEmbedder:
    """The deterministic local hash embedding as an `Embedder`. Default for old
    KBs (dim 64) and offline / test paths; never hits the network."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    async def embed(
        self, texts: Sequence[str], *, text_type: str = "document"
    ) -> List[List[float]]:
        from app.knowledge import embed_text  # lazy: avoids import cycle

        return [embed_text(t, dimension=self.dimension) for t in texts]


def build_embedder(embedding_config: Optional[dict], router) -> Embedder:
    """Resolve a KB's frozen ``embedding_config`` to an embedder client.

    Any catalogued provider (``provider_id`` other than ``local_hash``) + a
    router → the catalogued embedder, whose wire protocol (dashscope-native or
    openai-compatible) the router picks from the model's ``protocol``. The
    local_hash default and rows created before this feature → `LocalHashEmbedder`.
    """
    cfg = embedding_config or {}
    provider_id = cfg.get("provider_id") or cfg.get("provider")
    model = cfg.get("model")
    if provider_id and provider_id != "local_hash" and router is not None and model:
        return router.get_embedder(model)
    return LocalHashEmbedder(dimension=int(cfg.get("dimension") or 64))
