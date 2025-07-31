import json
from typing import List, Dict, Any, Optional
from llama_index.core.vector_stores.types import VectorStoreQueryResult
import aiohttp
import asyncio

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

    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行文档重排序

        Args:
            query: 查询语句
            documents: 需要排序的文档列表
            model: 覆盖默认模型
            top_n: 返回的最相关文档数量

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
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/rerank",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()
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
        response_data = await self.rerank(query, documents, model, top_n)

        try:
            return_nodes = []
            return_similarities = []
            for result in response_data["results"]:
                node = origin_nodes[result["index"]]
                node.metadata["rerank"] = True
                return_nodes.append(node)
                return_similarities.append(result["relevance_score"])
            return VectorStoreQueryResult(nodes=return_nodes, similarities=return_similarities)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"响应解析失败: {str(e)}") from e


async def test_rerank():
    reranker = OpenAICompatibleReranker(
        base_url="http:/demo.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/qwen3_reranker",
        model="Qwen3-Reranker-4B",
        timeout=60,
        api_key="=="
    )

    try:
        result = await reranker.rerank(
            query="中国首都是哪儿?",
            documents=[
                "美国首都是华盛顿。",
                "中国首都是北京。",
                "今天是星期五。",
            ],
            top_n=3
        )
        print("重排序结果:", result)
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_rerank())
