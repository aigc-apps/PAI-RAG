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
                if self.base_url.endswith("/v1/rerank"):
                    endpoint = self.base_url
                elif self.base_url.endswith("/v1"):
                    endpoint = f"{self.base_url}/rerank"
                else:
                    endpoint = f"{self.base_url}/v1/rerank"
                async with session.post(
                    endpoint,
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
        base_url="http://demo.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/ranxia_qwen3_rerank_8b",
        model="Qwen3-Reranker-8B",
        timeout=60,
        api_key="api_key"
    )

    try:
        result = await reranker.rerank(
            query="如何优化数据库查询性能？有哪些具体的优化方法和技巧？",
            documents=[
                "数据库查询性能优化是提升应用响应速度的关键。可以通过创建合适的索引、优化SQL语句结构、使用查询缓存、分析执行计划等方式来提升查询效率。索引应该建立在经常用于WHERE、JOIN和ORDER BY的列上，但要避免过度索引。",
        "Python是一种高级编程语言，具有简洁的语法和强大的功能。它广泛应用于Web开发、数据分析、人工智能等领域。Python的生态系统非常丰富，有大量的第三方库可以使用。",
        "在MySQL中，可以通过EXPLAIN命令来分析SQL查询的执行计划。执行计划显示了数据库如何执行查询，包括使用的索引、表连接方式等信息。通过分析执行计划，可以找出性能瓶颈并进行优化。",
        "数据库索引是一种数据结构，用于快速定位和访问数据库表中的数据。常见的索引类型包括B树索引、哈希索引等。索引可以显著提高查询速度，但会增加写入操作的开销，因为每次插入、更新或删除数据时都需要维护索引。",
        "Redis是一个开源的内存数据结构存储系统，可以用作数据库、缓存和消息中间件。它支持多种数据结构，如字符串、列表、集合、有序集合等。Redis的读写性能非常高，常用于缓存热点数据。",
        "SQL查询优化技巧包括：避免使用SELECT *，只查询需要的列；使用LIMIT限制返回结果数量；合理使用JOIN，避免笛卡尔积；在WHERE子句中使用索引列；避免在WHERE子句中使用函数，这会导致索引失效。",
        "微服务架构是一种将应用程序构建为一套小型服务的方法，每个服务运行在自己的进程中，并通过轻量级机制（通常是HTTP API）进行通信。这种架构模式有助于提高系统的可扩展性和可维护性。",
            ],
            top_n=6
        )
        print("重排序结果:", result)
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_rerank())
