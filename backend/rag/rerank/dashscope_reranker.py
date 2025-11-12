import json
from typing import List, Dict, Any, Optional
from llama_index.core.vector_stores.types import VectorStoreQueryResult
import aiohttp
import asyncio


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
            query: 查询语句，最大长度不能超过4,000个Token
            documents: 需要排序的文档列表，最多包含500个文档，每个文档长度不超过4,000个Token
            model: 覆盖默认模型，支持 "qwen3-rerank" 或 "gte-rerank-v2"
            top_n: 返回的最相关文档数量

        Returns:
            API响应结果，格式为 {"results": [{"index": int, "relevance_score": float, "document": {"text": str}}]}

        Raises:
            ValueError: 参数验证失败时
            RuntimeError: API返回错误时
        """
        # 参数验证
        if not query:
            raise ValueError("查询内容不能为空")
        if not documents:
            raise ValueError("文档列表不能为空")
        if len(documents) > 500:
            raise ValueError("文档数量不能超过500个")

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
            async with aiohttp.ClientSession() as session:
                # DashScope API端点
                # 完整端点为: https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
                # 确保URL结尾有两个/text-rerank
                if self.base_url.endswith("/text-rerank/text-rerank"):
                    endpoint = self.base_url
                elif self.base_url.endswith("/text-rerank"):
                    endpoint = f"{self.base_url}/text-rerank"
                else:
                    endpoint = f"{self.base_url}/text-rerank/text-rerank"

                async with session.post(
                    endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_data = await response.json()

                    # 检查API返回的错误
                    if "code" in response_data and response_data["code"]:
                        error_msg = response_data.get("message", "未知错误")
                        raise RuntimeError(f"DashScope API错误: {error_msg} (code: {response_data['code']})")

                    # 如果HTTP状态码不是200，抛出异常
                    if response.status != 200:
                        error_msg = response_data.get("message", f"HTTP {response.status}")
                        raise RuntimeError(f"API请求失败: {error_msg}")

                    # 转换DashScope响应格式为兼容格式
                    # DashScope返回: {"output": {"results": [...]}, "usage": {...}, "request_id": "..."}
                    # 转换为: {"results": [...]}
                    if "output" in response_data and "results" in response_data["output"]:
                        return {"results": response_data["output"]["results"]}
                    else:
                        # 如果已经是兼容格式，直接返回
                        return response_data
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
            重排序后的VectorStoreQueryResult

        Raises:
            ValueError: 参数验证失败时
            RuntimeError: API返回错误时
        """
        # 参数验证
        if not query:
            raise ValueError("查询内容不能为空")
        if not result:
            raise ValueError("VectorStoreQueryResult列表不能为空")

        origin_nodes = result.nodes
        documents = [node.text for node in origin_nodes]
        response_data = await self.rerank(query, documents, model, top_n)

        try:
            return_nodes = []
            return_similarities = []
            for result_item in response_data["results"]:
                # DashScope返回格式: {"index": int, "relevance_score": float, "document": {"text": str}}
                index = result_item["index"]
                node = origin_nodes[index]
                node.metadata["rerank"] = True
                return_nodes.append(node)
                return_similarities.append(result_item["relevance_score"])
            return VectorStoreQueryResult(nodes=return_nodes, similarities=return_similarities)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"响应解析失败: 无法从响应中提取结果 ({str(e)})") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"响应解析失败: {str(e)}") from e


async def test_rerank():
    """测试DashScope reranker"""
    # 需要设置环境变量 DASHSCOPE_API_KEY 或直接传入api_key
    import os
    api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")

    reranker = DashscopeReranker(
        base_url="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        model="qwen3-rerank",  # 或 "gte-rerank-v2"
        timeout=60,
        api_key=api_key
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
