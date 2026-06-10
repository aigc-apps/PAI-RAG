"""DashScope 多模态向量模型适配器。

兼容 LlamaIndex `BaseEmbedding` 接口的同时，额外暴露
`aget_multimodal_embedding`，让索引侧可以把节点中的图片直接喂入向量化。

参考: https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference
端点: POST https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
支持模型: qwen3-vl-embedding / qwen2.5-vl-embedding /
        tongyi-embedding-vision-* / multimodal-embedding-v1
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from loguru import logger

from utils.http_session import HttpSessionShared


DEFAULT_DASHSCOPE_MM_EMBED_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)

# 默认开启融合向量的模型；其余模型即使传入 enable_fusion=true 也会被忽略
_FUSION_CAPABLE_MODELS = {
    "qwen3-vl-embedding",
    "qwen2.5-vl-embedding",
}

# 单请求图片数上限（参考 DashScope qwen3-vl-embedding / qwen2.5-vl-embedding 限制：
# 单请求最多 20 个元素、其中最多 5 张图片）。超出时截断并打 warning，
# 避免索引整批失败。
_PER_REQUEST_IMAGE_LIMITS: Dict[str, int] = {
    "qwen3-vl-embedding": 5,
    "qwen2.5-vl-embedding": 5,
}
_PER_REQUEST_VIDEO_LIMIT_DEFAULT = 3
_PER_REQUEST_TOTAL_ELEMENTS_LIMIT = 20


class MultimodalDashscopeEmbedding(BaseEmbedding):
    """DashScope 多模态向量化客户端。

    - 当输入只有文本时（``aget_text_embedding`` / ``aget_query_embedding``），
      请求 ``contents=[{"text": ...}]``，返回首个向量。
    - 当输入是 ``parts`` 列表时（``aget_multimodal_embedding``），将文本与图
      片放入同一个 ``contents`` 中，并对支持融合的模型设置
      ``enable_fusion=true``，从而返回 1 个融合向量；对仅支持独立向量的模型
      则取首个文本/融合向量作为该节点的向量。
    """

    base_url: str = Field(default=DEFAULT_DASHSCOPE_MM_EMBED_ENDPOINT)
    model_name: str = Field(default="qwen3-vl-embedding")
    timeout: int = Field(default=60)
    dimension: Optional[int] = Field(default=None)
    embed_batch_size: int = Field(default=10)

    _api_key: Optional[str] = PrivateAttr()
    _endpoint: str = PrivateAttr()
    _headers: Dict[str, str] = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "qwen3-vl-embedding",
        base_url: str = DEFAULT_DASHSCOPE_MM_EMBED_ENDPOINT,
        timeout: int = 60,
        dimension: Optional[int] = None,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout=timeout,
            dimension=dimension,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._api_key = api_key
        # 允许传入完整 endpoint 或仅根域名
        url = (base_url or DEFAULT_DASHSCOPE_MM_EMBED_ENDPOINT).rstrip("/")
        if url.endswith("/multimodal-embedding/multimodal-embedding"):
            self._endpoint = url
        elif url.endswith("/multimodal-embedding"):
            self._endpoint = f"{url}/multimodal-embedding"
        else:
            self._endpoint = (
                f"{url}/api/v1/services/embeddings/"
                "multimodal-embedding/multimodal-embedding"
            )

        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            if api_key.startswith("Bearer "):
                self._headers["Authorization"] = api_key
            else:
                self._headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def class_name(cls) -> str:
        return "multimodal_dashscope_embedding"

    # ------------------------------------------------------------------
    # 内部 HTTP 调用
    # ------------------------------------------------------------------
    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            session = await HttpSessionShared.ensure_session()
            async with session.post(
                self._endpoint,
                headers=self._headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    try:
                        err = await response.json()
                        msg = err.get("message", f"HTTP {response.status}")
                    except Exception:
                        msg = f"HTTP {response.status}"
                    raise RuntimeError(
                        f"DashScope multimodal embedding request failed: {msg}"
                    )
                data = await response.json()
                if data.get("code"):
                    raise RuntimeError(
                        f"DashScope multimodal embedding error: "
                        f"{data.get('message', 'unknown')} (code: {data['code']})"
                    )
                return data
        except aiohttp.ClientError as e:
            raise RuntimeError(f"DashScope request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"DashScope response parsing failed: {e}") from e

    def _build_payload(
        self,
        contents: List[Dict[str, Any]],
        *,
        prefer_fusion: bool,
    ) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {}
        if self.dimension is not None and self.dimension > 0:
            parameters["dimension"] = self.dimension
        # qwen3-vl-embedding 默认输出独立向量，需显式开启融合
        if prefer_fusion and self.model_name in _FUSION_CAPABLE_MODELS:
            parameters["enable_fusion"] = True

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": {"contents": contents},
        }
        if parameters:
            payload["parameters"] = parameters
        return payload

    @staticmethod
    def _pick_vector(
        embeddings: List[Dict[str, Any]],
        prefer_types: List[str],
    ) -> List[float]:
        """按优先级在返回结果中选择代表节点的向量。"""
        if not embeddings:
            raise RuntimeError("DashScope multimodal embedding returned empty result")
        for t in prefer_types:
            for item in embeddings:
                if item.get("type") == t:
                    return item.get("embedding", [])
        # fallback: 第一个
        return embeddings[0].get("embedding", [])

    # ------------------------------------------------------------------
    # 文本路径（兼容现有索引/检索调用）
    # ------------------------------------------------------------------
    async def _aget_single_text_embedding(self, text: str) -> List[float]:
        payload = self._build_payload(
            contents=[{"text": text or ""}],
            prefer_fusion=False,
        )
        data = await self._post(payload)
        out = data.get("output", {})
        return self._pick_vector(out.get("embeddings", []), prefer_types=["text", "fusion", "fused", "vl"])

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_single_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._aget_single_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # DashScope 每次 contents 内每条返回独立向量；这里逐个发请求即可保证
        # 顺序与 batch 大小一致（避免不同模型对独立/融合的限制）
        coros = [self._aget_single_text_embedding(t) for t in texts]
        return await asyncio.gather(*coros)

    # ------------------------------------------------------------------
    # 同步占位（PAI-RAG 全异步路径，但 BaseEmbedding 抽象方法要求实现）
    # ------------------------------------------------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        return asyncio.get_event_loop().run_until_complete(
            self._aget_query_embedding(query)
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        return asyncio.get_event_loop().run_until_complete(
            self._aget_text_embedding(text)
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return asyncio.get_event_loop().run_until_complete(
            self._aget_text_embeddings(texts)
        )

    # ------------------------------------------------------------------
    # 多模态扩展
    # ------------------------------------------------------------------
    def _truncate_modalities(
        self,
        images: Optional[List[str]],
        videos: Optional[List[str]],
        has_text: bool,
    ) -> tuple[List[str], List[str]]:
        """按 DashScope 单请求上限截断 images/videos，并打 warning。

        - 不同模型对单请求图片数有上限（如 qwen3-vl-embedding 为 5）；
        - 同时单请求总元素数（text + image + video）不超过 20。

        截断而非抛错，避免整批索引失败；调用方可读取日志并复跑。
        """
        image_limit = _PER_REQUEST_IMAGE_LIMITS.get(self.model_name)
        imgs = [u for u in (images or []) if u]
        vids = [u for u in (videos or []) if u]

        if image_limit is not None and len(imgs) > image_limit:
            logger.warning(
                "[MultimodalDashscopeEmbedding] model={model} received {n} images "
                "in a single request, truncating to {limit} (DashScope per-request "
                "image cap). Consider splitting nodes upstream.",
                model=self.model_name,
                n=len(imgs),
                limit=image_limit,
            )
            imgs = imgs[:image_limit]

        if len(vids) > _PER_REQUEST_VIDEO_LIMIT_DEFAULT:
            logger.warning(
                "[MultimodalDashscopeEmbedding] model={model} received {n} videos "
                "in a single request, truncating to {limit}.",
                model=self.model_name,
                n=len(vids),
                limit=_PER_REQUEST_VIDEO_LIMIT_DEFAULT,
            )
            vids = vids[:_PER_REQUEST_VIDEO_LIMIT_DEFAULT]

        # 总元素 ≤ 20（text 占用一个 slot）
        max_elements = _PER_REQUEST_TOTAL_ELEMENTS_LIMIT - (1 if has_text else 0)
        if len(imgs) + len(vids) > max_elements:
            logger.warning(
                "[MultimodalDashscopeEmbedding] model={model} total elements "
                "{total} exceeds per-request cap {cap}, further truncating images.",
                model=self.model_name,
                total=len(imgs) + len(vids) + (1 if has_text else 0),
                cap=_PER_REQUEST_TOTAL_ELEMENTS_LIMIT,
            )
            # 先保留 video，再截 image
            vid_keep = min(len(vids), max_elements)
            img_keep = max(0, max_elements - vid_keep)
            imgs = imgs[:img_keep]
            vids = vids[:vid_keep]

        return imgs, vids

    async def aget_multimodal_embedding(
        self,
        text: Optional[str] = None,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
    ) -> List[float]:
        """对单个节点执行多模态向量化，返回 1 个向量。

        融合策略：
        - 模型支持融合（qwen3-vl-embedding / qwen2.5-vl-embedding）：直接合并
          所有模态到一次请求并设置 ``enable_fusion=true``；
        - 模型仅支持独立向量（如 tongyi-embedding-vision-plus）：仍合并到一次
          请求，返回时优先取文本向量作为节点代表向量（与现网用户对纯文本查询
          的检索行为一致）。

        注意：超过 DashScope 单请求上限的 images/videos 会被截断并打 warning，
        以避免单个节点导致整批索引失败。
        """
        has_text = bool(text)
        imgs, vids = self._truncate_modalities(images, videos, has_text=has_text)

        contents: List[Dict[str, Any]] = []
        if has_text:
            contents.append({"text": text})
        for img in imgs:
            contents.append({"image": img})
        for v in vids:
            contents.append({"video": v})
        if not contents:
            raise ValueError("aget_multimodal_embedding requires at least one of text/images/videos")

        payload = self._build_payload(contents=contents, prefer_fusion=True)
        data = await self._post(payload)
        out = data.get("output", {})
        embeddings = out.get("embeddings", [])
        # 融合模型优先取 fusion/fused，否则回退到 text，再否则首个
        return self._pick_vector(
            embeddings,
            prefer_types=["fusion", "fused", "text", "vl", "image"],
        )

    async def aget_multimodal_embedding_batch(
        self,
        items: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """并发批量多模态向量化。

        ``items[i]`` 形如 ``{"text": str | None, "images": [url, ...]}``。
        """
        sem = asyncio.Semaphore(max(1, self.embed_batch_size))

        async def _run(item: Dict[str, Any]) -> List[float]:
            async with sem:
                return await self.aget_multimodal_embedding(
                    text=item.get("text"),
                    images=item.get("images"),
                    videos=item.get("videos"),
                )

        results = await asyncio.gather(*[_run(it) for it in items])
        if show_progress:
            logger.info(
                f"MultimodalDashscopeEmbedding produced {len(results)} vectors"
            )
        return results
