import json
import requests
from typing import List, Dict, Any, Optional

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

    def rerank(
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

        # 发送请求
        try:
            response = requests.post(
                f"{self.base_url}/v1/rerank",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API请求失败: {str(e)}") from e

        # 处理响应
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"响应解析失败: {str(e)}") from e

# 示例用法
if __name__ == "__main__":
    # 创建客户端实例
    reranker = OpenAICompatibleReranker(
        base_url="http://xxx.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/qwen3_reranker",
        model="Qwen3-Reranker-4B",
        timeout=60,
        api_key="=="
    )

    try:
        # 执行重排序
        result = reranker.rerank(
            query="中国首都是哪儿?",
            documents=[
                "中国首都是北京。",
                "美国首都是华盛顿。",
                "今天是星期五。",
            ],
            top_n=3
        )

        # 格式化输出结果
        print("重排序结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"发生错误: {str(e)}")
