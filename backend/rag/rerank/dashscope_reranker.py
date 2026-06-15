import json
from typing import List, Optional
from llama_index.core.vector_stores.types import VectorStoreQueryResult
import aiohttp
from utils.http_session import HttpSessionShared
from rag.rerank.reranker import RerankResult
from extensions.trace.rag_wrapper import reranker_wrapper
from loguru import logger


class DashscopeReranker:
    """
    DashScope兼容的Reranker

    支持阿里云DashScope文本排序API
    参考文档: https://help.aliyun.com/zh/model-studio/text-rerank-api
    """

    def __init__(
        self,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        model: str = "qwen3-rerank",
        timeout: int = 30,
        api_key: Optional[str] = None
    ):
        """
        初始化DashScope Reranker客户端

        Args:
            base_url: API基础URL，默认为DashScope端点
            model: 默认模型名称，支持 "qwen3-rerank" 或 "gte-rerank-v2"
            timeout: 请求超时时间（秒）
            api_key: DashScope API密钥，格式为 "sk-xxxx" 或完整 "Bearer sk-xxxx"
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if api_key:
            # 确保Authorization格式为 "Bearer {api_key}"
            if api_key.startswith("Bearer "):
                self.headers["Authorization"] = api_key
            else:
                self.headers["Authorization"] = f"Bearer {api_key}"

        # 预先设置endpoint
        # DashScope API端点
        # 完整端点为: https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
        # 确保URL结尾有两个/text-rerank
        # 如果base_url已经包含完整的endpoint路径，直接使用
        if self.base_url.endswith("/text-rerank/text-rerank"):
            self.endpoint = self.base_url
        elif self.base_url.endswith("/text-rerank"):
            self.endpoint = f"{self.base_url}/text-rerank"
        else:
            self.endpoint = f"{self.base_url}/text-rerank/text-rerank"

    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        similarity_threshold: float = 0,
    ) -> List[RerankResult]:
        """
        执行文档重排序

        Args:
            query: 查询语句，最大长度不能超过4,000个Token
            documents: 需要排序的文档列表，最多包含500个文档，每个文档长度不超过4,000个Token
            model: 覆盖默认模型，支持 "qwen3-rerank" 或 "gte-rerank-v2"
            top_n: 返回的最相关文档数量
            similarity_threshold: 相似度阈值
        Returns:
            排序好的结果列表，每个结果包含index, score, doc字段

        Raises:
            ValueError: 参数验证失败时
            RuntimeError: API返回错误时
        """
        # Parameter validation
        if not query:
            raise ValueError("Query content cannot be empty")
        if not documents:
            raise ValueError("Document list cannot be empty")
        if len(documents) > 500:
            raise ValueError("Document count cannot exceed 500")

        # 构造DashScope格式的请求数据
        payload = {
            "model": model or self.model,
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "return_documents": True
            }
        }

        if top_n is not None:
            payload["parameters"]["top_n"] = top_n

        # 发送异步请求
        try:
            session = await HttpSessionShared.ensure_session()
            async with session.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                # 如果HTTP状态码不是200，抛出异常
                if response.status != 200:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get("message", f"HTTP {response.status}")
                        raise RuntimeError(f"API request failed: {error_msg} from {response.status}")
                    except Exception as e:
                        logger.error(f"API请求失败: {str(e)}")
                        raise
                response_data = await response.json()

                # 检查API返回的错误
                if "code" in response_data and response_data["code"]:
                    error_msg = response_data.get("message", "Unknown error")
                    raise RuntimeError(f"DashScope API error: {error_msg} (code: {response_data['code']})")

                # 转换DashScope响应格式为兼容格式
                # DashScope返回: {"output": {"results": [...]}, "usage": {...}, "request_id": "..."}
                if "output" in response_data and "results" in response_data["output"]:
                    raw_results = response_data["output"]["results"]
                elif "results" in response_data:
                    raw_results = response_data["results"]
                else:
                    raise RuntimeError(f"Response format error: results field not found from {response_data}")

                # 验证结果格式
                if not isinstance(raw_results, list):
                    raise RuntimeError(f"Response format error: results should be a list from {response_data}")

                # 解析并构建结构化结果
                rerank_results = []
                for item in raw_results:
                    index = item.get("index")
                    if index is None:
                        raise RuntimeError("Response format error: index field missing in result")

                    score = item.get("relevance_score", 0.0)

                    if score < similarity_threshold:
                        continue

                    # 提取文档文本
                    if "document" in item and isinstance(item["document"], dict):
                        doc = item["document"].get("text", "")
                    else:
                        # 如果没有document字段，使用原始documents中的文本
                        doc = documents[index] if 0 <= index < len(documents) else ""

                    rerank_results.append(RerankResult(
                        index=index,
                        score=score,
                        doc=doc
                    ))

                # 确保结果按score降序排序
                rerank_results.sort(key=lambda x: x.score, reverse=True)
                return rerank_results

        except aiohttp.ClientError as e:
            raise RuntimeError(f"API request failed: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Response parsing failed: {str(e)}") from e

    @reranker_wrapper
    async def vector_store_rerank(
        self,
        query: str,
        vector_result: VectorStoreQueryResult,
        top_n: Optional[int] = None,
        similarity_threshold: float = 0,
        model: Optional[str] = None,
    ) -> VectorStoreQueryResult:
        """
        执行vector store query result重排序

        Args:
            query: 查询语句
            result: 需要排序的vector store query result
            top_n: 返回的最相关node数量
            model: 覆盖默认模型

        Returns:
            重排序后的VectorStoreQueryResult

        Raises:
            ValueError: 参数验证失败时
            RuntimeError: API返回错误时
        """
        # Parameter validation
        if not query:
            raise ValueError("Query content cannot be empty")
        if not vector_result:
            raise ValueError("VectorStoreQueryResult list cannot be empty")

        # DashScope rejects empty documents; drop empty-text nodes first while
        # preserving each survivor's original similarity. Apply similarity_threshold
        # to the sole survivor too — otherwise a low-score node bypasses the gate.
        nodes = vector_result.nodes or []
        sims = vector_result.similarities or [0.0] * len(nodes)
        kept = [(n, s) for n, s in zip(nodes, sims) if (n.text or "").strip()]
        if not kept:
            return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])
        if len(kept) == 1:
            only, sim = kept[0]
            if sim < similarity_threshold:
                return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])
            return VectorStoreQueryResult(nodes=[only], ids=[only.node_id], similarities=[sim])

        origin_nodes = [n for n, _ in kept]
        documents = [node.text for node in origin_nodes]
        rerank_results = await self.rerank(query, documents, model, top_n, similarity_threshold)

        return_nodes = []
        return_ids = []
        return_similarities = []
        for rerank_result in rerank_results:
            node = origin_nodes[rerank_result.index]
            node.metadata["rerank"] = True
            return_nodes.append(node)
            return_ids.append(node.node_id)
            return_similarities.append(rerank_result.score)
        return VectorStoreQueryResult(nodes=return_nodes, ids=return_ids, similarities=return_similarities)
