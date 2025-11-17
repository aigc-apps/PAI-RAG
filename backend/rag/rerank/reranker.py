import json
from typing import List, Optional
from dataclasses import dataclass
from llama_index.core.vector_stores.types import VectorStoreQueryResult
import aiohttp
from utils.http_session import HttpSessionShared


@dataclass
class RerankResult:
    """重排序结果"""
    index: int
    score: float
    doc: str

class OpenAICompatibleReranker:
    """
    OpenAI兼容的Reranker

    支持与Jina/Cohere兼容的rerank API
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        model: str = "BAAI/bge-reranker-base",
        timeout: int = 30,
        api_key: Optional[str] = None
    ):
        """
        初始化Reranker客户端

        Args:
            base_url: API基础URL
            model: 默认模型名称
            timeout: 请求超时时间（秒）
            api_key: 可选API密钥（如果服务需要）
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
            self.headers["Authorization"] = api_key

        if self.base_url.endswith("/v1/rerank"):
            self.endpoint = self.base_url
        elif self.base_url.endswith("/v1"):
            self.endpoint = f"{self.base_url}/rerank"
        else:
            self.endpoint = f"{self.base_url}/v1/rerank"

    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """
        执行文档重排序

        Args:
            query: 查询语句
            documents: 需要排序的文档列表
            model: 覆盖默认模型
            top_n: 返回的最相关文档数量

        Returns:
            排序好的结果列表，每个结果包含index, score, doc字段

        Raises:
            ValueError: 参数验证失败时
            requests.exceptions.RequestException: 网络请求相关异常
            RuntimeError: API返回错误时
        """
        # 参数验证
        if not query:
            raise ValueError("查询内容不能为空")
        if not documents:
            raise ValueError("文档列表不能为空")
        # 构造请求数据
        payload = {
            "model": model or self.model,
            "query": query,
            "documents": documents,
        }

        if top_n is not None:
            payload["top_n"] = top_n

        # 发送异步请求
        try:
            session = await HttpSessionShared.ensure_session()
            async with session.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                response_data = await response.json()

                # 解析响应并返回排序好的结果
                if "results" not in response_data:
                    raise RuntimeError("响应格式错误: 未找到results字段")

                raw_results = response_data["results"]
                if not isinstance(raw_results, list):
                    raise RuntimeError("响应格式错误: results应该是列表")

                # 解析并构建结构化结果
                rerank_results = []
                for item in raw_results:
                    index = item.get("index")
                    if index is None:
                        raise RuntimeError("响应格式错误: 结果中缺少index字段")

                    score = item.get("relevance_score", 0.0)
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
            raise RuntimeError(f"API请求失败: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"响应解析失败: {str(e)}") from e

    async def vector_store_rerank(
        self,
        query: str,
        result: VectorStoreQueryResult,
        top_n: Optional[int] = None,
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
            API响应结果

        Raises:
            ValueError: 参数验证失败时
            requests.exceptions.RequestException: 网络请求相关异常
            RuntimeError: API返回错误时
        """
        # 参数验证
        if not query:
            raise ValueError("查询内容不能为空")
        if not result:
            raise ValueError("VectorStoreQueryResult列表不能为空")

        origin_nodes = result.nodes
        documents=[node.text for node in origin_nodes]
        rerank_results = await self.rerank(query, documents, model, top_n)

        return_nodes = []
        return_similarities = []
        for rerank_result in rerank_results:
            node = origin_nodes[rerank_result.index]
            node.metadata["rerank"] = True
            return_nodes.append(node)
            return_similarities.append(rerank_result.score)
        return VectorStoreQueryResult(nodes=return_nodes, similarities=return_similarities)
