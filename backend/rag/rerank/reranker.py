import json
from typing import List, Optional
from dataclasses import dataclass
from llama_index.core.vector_stores.types import VectorStoreQueryResult
import aiohttp
from utils.http_session import HttpSessionShared
from extensions.trace.rag_wrapper import reranker_wrapper


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

    # Qwen3-Reranker 模型列表，这些模型需要使用特殊的格式
    QWEN3_RERANKER_MODELS = ["Qwen3-Reranker-8B", "Qwen3-Reranker-4B", "Qwen3-Reranker-0.6B"]

    # Qwen3-Reranker 特殊格式的 prefix 和 suffix
    QWEN3_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    QWEN3_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

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
        top_n: Optional[int] = None,
        similarity_threshold: float = 0,
    ) -> List[RerankResult]:
        """
        执行文档重排序

        Args:
            query: 查询语句
            documents: 需要排序的文档列表
            model: 覆盖默认模型
            top_n: 返回的最相关文档数量
            similarity_threshold: 相似度阈值
        Returns:
            排序好的结果列表，每个结果包含index, score, doc字段

        Raises:
            ValueError: 参数验证失败时
            requests.exceptions.RequestException: 网络请求相关异常
            RuntimeError: API返回错误时
        """
        # Parameter validation
        if not query:
            raise ValueError("Query content cannot be empty")
        if not documents:
            raise ValueError("Document list cannot be empty")

        model = model or self.model

        use_qwen3_format = model in self.QWEN3_RERANKER_MODELS

        # 根据模型类型格式化 query 和 documents
        if use_qwen3_format:
            # 格式化 query: {prefix}<Instruct>: {instruction}\n<Query>: {query}\n
            formatted_query = f"{self.QWEN3_PREFIX}<Instruct>: {self.QWEN3_INSTRUCTION}\n<Query>: {query}\n"
            # 格式化 documents: <Document>: {doc}{suffix}
            formatted_documents = [
                f"<Document>: {doc}{self.QWEN3_SUFFIX}" for doc in documents
            ]
        else:
            formatted_query = query
            formatted_documents = documents

        # 构造请求数据
        payload = {
            "model": model,
            "query": formatted_query,
            "documents": formatted_documents,
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

                # Parse response and return sorted results
                if "results" not in response_data:
                    raise RuntimeError("Response format error: results field not found")

                raw_results = response_data["results"]
                if not isinstance(raw_results, list):
                    raise RuntimeError("Response format error: results should be a list")

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

                    # 如果是 Qwen3-Reranker 模型，需要从返回的 text 中提取原始内容
                    # 去掉 <Document>: 前缀和 suffix
                    if use_qwen3_format and doc:
                        # 去掉 <Document>: 前缀
                        if doc.startswith("<Document>:"):
                            doc = doc[len("<Document>:"):].lstrip()
                        # 去掉 suffix
                        if doc.endswith(self.QWEN3_SUFFIX):
                            doc = doc[:-len(self.QWEN3_SUFFIX)].rstrip()
                        # 如果仍然没有找到原始内容，使用原始 documents 中的文本
                        if not doc:
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
        model: Optional[str] = None,
        similarity_threshold: float = 0,
    ) -> VectorStoreQueryResult:
        """
        执行vector store query result重排序

        Args:
            query: 查询语句
            vector_result: 需要排序的vector store query result
            top_n: 返回的最相关node数量
            model: 覆盖默认模型
            similarity_threshold: 相似度阈值
        Returns:
            API响应结果

        Raises:
            ValueError: 参数验证失败时
            requests.exceptions.RequestException: 网络请求相关异常
            RuntimeError: API返回错误时
        """
        # Parameter validation
        if not query:
            raise ValueError("Query content cannot be empty")
        if not vector_result:
            raise ValueError("VectorStoreQueryResult list cannot be empty")

        # Rerank APIs reject empty documents; drop empty-text nodes first while
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
        documents=[node.text for node in origin_nodes]
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
