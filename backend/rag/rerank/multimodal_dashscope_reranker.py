"""DashScope 多模态 rerank 适配器（如 qwen3-vl-rerank）。

参考: https://help.aliyun.com/zh/model-studio/text-rerank-api
端点: POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank

与 ``DashscopeReranker`` 的差异：
- ``query`` 与 ``documents`` 中的元素均为 ``{"text"|"image"|"video": value}``
  对象，能把节点的图片一并送入排序模型；
- 当节点仅有文本时，会向后兼容地传入对象形式 ``{"text": ...}``。
"""
import json
from typing import Any, Dict, List, Optional, Union

import aiohttp
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from loguru import logger

from extensions.trace.rag_wrapper import reranker_wrapper
from rag.rerank.reranker import RerankResult
from utils.http_session import HttpSessionShared


DEFAULT_DASHSCOPE_MM_RERANK_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


def _node_has_renderable_content(node) -> bool:
    """节点是否有可送入 rerank 的内容：非空文本 或 至少一张图片 URL。"""
    if (node.text or "").strip():
        return True
    images_info = (node.metadata or {}).get("images_info") or []
    for img in images_info:
        if isinstance(img, dict) and img.get("url"):
            return True
    return False


def _node_documents_with_images(nodes) -> List[Dict[str, Any]]:
    """将节点转换为多模态 rerank 可接受的 documents。

    每个节点产生一条 doc。若节点 ``metadata.images_info`` 非空，则附带其首张
    图片的 URL（多模态 rerank 当前每条 doc 只能携带一种模态，因此不便同时塞
    入文本+图片）。文本节点回落为 ``{"text": node.text}``。
    """
    documents: List[Dict[str, Any]] = []
    for node in nodes:
        images_info = (node.metadata or {}).get("images_info") or []
        first_image_url = None
        for img in images_info:
            url = img.get("url") if isinstance(img, dict) else None
            if url:
                first_image_url = url
                break

        text = node.text or ""
        if first_image_url and not text.strip():
            documents.append({"image": first_image_url})
        elif first_image_url:
            # 文本+图片：当前模型每个 doc 仅支持单模态，文本优先
            documents.append({"text": text})
        else:
            documents.append({"text": text})
    return documents


class MultimodalDashscopeReranker:
    """支持图片/视频文档的 DashScope rerank 客户端。"""

    def __init__(
        self,
        base_url: str = DEFAULT_DASHSCOPE_MM_RERANK_ENDPOINT,
        model: str = "qwen3-vl-rerank",
        timeout: int = 60,
        api_key: Optional[str] = None,
    ):
        self.base_url = (base_url or DEFAULT_DASHSCOPE_MM_RERANK_ENDPOINT).rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            if api_key.startswith("Bearer "):
                self.headers["Authorization"] = api_key
            else:
                self.headers["Authorization"] = f"Bearer {api_key}"

        if self.base_url.endswith("/text-rerank/text-rerank"):
            self.endpoint = self.base_url
        elif self.base_url.endswith("/text-rerank"):
            self.endpoint = f"{self.base_url}/text-rerank"
        else:
            self.endpoint = f"{self.base_url}/text-rerank/text-rerank"

    async def rerank(
        self,
        query: Union[str, Dict[str, str]],
        documents: List[Union[str, Dict[str, str]]],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        similarity_threshold: float = 0,
    ) -> List[RerankResult]:
        if not query:
            raise ValueError("Query content cannot be empty")
        if not documents:
            raise ValueError("Document list cannot be empty")
        if len(documents) > 500:
            raise ValueError("Document count cannot exceed 500")

        # 统一文档格式：字符串视为 {"text": ...}
        normalized_docs: List[Dict[str, str]] = []
        for d in documents:
            if isinstance(d, dict):
                normalized_docs.append(d)
            else:
                normalized_docs.append({"text": str(d) if d is not None else ""})

        # 查询同样支持 str 与 dict
        normalized_query = query if isinstance(query, dict) else {"text": str(query)}

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "input": {
                "query": normalized_query,
                "documents": normalized_docs,
            },
            "parameters": {
                "return_documents": True,
            },
        }
        if top_n is not None:
            payload["parameters"]["top_n"] = top_n

        try:
            session = await HttpSessionShared.ensure_session()
            async with session.post(
                self.endpoint,
                headers=self.headers,
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
                        f"Multimodal rerank request failed: {msg}"
                    )
                data = await response.json()

                if data.get("code"):
                    raise RuntimeError(
                        f"DashScope multimodal rerank error: "
                        f"{data.get('message', 'unknown')} (code: {data['code']})"
                    )

                if "output" in data and "results" in data["output"]:
                    raw_results = data["output"]["results"]
                elif "results" in data:
                    raw_results = data["results"]
                else:
                    raise RuntimeError(
                        f"Response format error: results field not found from {data}"
                    )

                if not isinstance(raw_results, list):
                    raise RuntimeError(
                        f"Response format error: results should be a list from {data}"
                    )

                rerank_results: List[RerankResult] = []
                for item in raw_results:
                    index = item.get("index")
                    if index is None:
                        raise RuntimeError(
                            "Response format error: index field missing in result"
                        )

                    score = item.get("relevance_score", 0.0)
                    if score < similarity_threshold:
                        continue

                    doc_obj = item.get("document")
                    doc_text: str = ""
                    if isinstance(doc_obj, dict):
                        doc_text = (
                            doc_obj.get("text")
                            or doc_obj.get("image")
                            or doc_obj.get("video")
                            or ""
                        )
                    elif isinstance(doc_obj, str):
                        doc_text = doc_obj
                    if not doc_text and 0 <= index < len(normalized_docs):
                        entry = normalized_docs[index]
                        doc_text = (
                            entry.get("text")
                            or entry.get("image")
                            or entry.get("video")
                            or ""
                        )

                    rerank_results.append(
                        RerankResult(index=index, score=score, doc=doc_text)
                    )

                rerank_results.sort(key=lambda x: x.score, reverse=True)
                return rerank_results

        except aiohttp.ClientError as e:
            raise RuntimeError(f"API request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Response parsing failed: {e}") from e

    @reranker_wrapper
    async def vector_store_rerank(
        self,
        query: str,
        vector_result: VectorStoreQueryResult,
        top_n: Optional[int] = None,
        similarity_threshold: float = 0,
        model: Optional[str] = None,
    ) -> VectorStoreQueryResult:
        if not query:
            raise ValueError("Query content cannot be empty")
        if not vector_result:
            raise ValueError("VectorStoreQueryResult list cannot be empty")
        # Rerank API rejects empty documents; keep only nodes with text or an image
        # while preserving each survivor's original similarity. Apply
        # similarity_threshold to the sole survivor too.
        nodes = vector_result.nodes or []
        sims = vector_result.similarities or [0.0] * len(nodes)
        kept = [(n, s) for n, s in zip(nodes, sims) if _node_has_renderable_content(n)]
        if not kept:
            return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])
        if len(kept) == 1:
            only, sim = kept[0]
            if sim < similarity_threshold:
                return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])
            return VectorStoreQueryResult(nodes=[only], ids=[only.node_id], similarities=[sim])

        origin_nodes = [n for n, _ in kept]
        documents = _node_documents_with_images(origin_nodes)

        rerank_results = await self.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n,
            similarity_threshold=similarity_threshold,
        )

        return_nodes = []
        return_ids = []
        return_similarities = []
        for r in rerank_results:
            node = origin_nodes[r.index]
            node.metadata["rerank"] = True
            return_nodes.append(node)
            return_ids.append(node.node_id)
            return_similarities.append(r.score)

        logger.info(
            f"MultimodalDashscopeReranker reranked {len(origin_nodes)} -> "
            f"{len(return_nodes)} nodes with model={model or self.model}"
        )
        return VectorStoreQueryResult(
            nodes=return_nodes, ids=return_ids, similarities=return_similarities
        )
