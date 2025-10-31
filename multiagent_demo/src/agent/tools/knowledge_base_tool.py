# tools/knowledge_base_tool.py
import httpx
from langchain_core.tools import tool
from agent.config import AgentConfig

_http_client = None


def get_http_client() -> httpx.AsyncClient:
    """Get or create HTTP client with connection pooling."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=60)
    return _http_client


@tool
async def knowledge_base_search(query: str) -> str:
    """
    通过向指定知识库 API 发送请求，检索与 query 相关的知识。
    
    Args:
        query (str): 用户的检索关键词或问题
        
    Returns:
        str: 知识库返回的文本结果，若失败则返回错误信息
    """
    if not AgentConfig.KB_API_URL or not AgentConfig.KB_AUTHORIZATION:
        return "知识库配置未设置，请配置 KB_API_URL 和 KB_AUTHORIZATION 环境变量。"
    
    url = AgentConfig.KB_API_URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": AgentConfig.KB_AUTHORIZATION
    }
    payload = {
        "knowledge_id": AgentConfig.KB_KNOWLEDGE_ID,
        "query": query
    }

    try:
        client = get_http_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        records = data.get("records", [])
        if not records:
            return "知识库中未找到相关信息。"
        return "\n\n".join([r.get("content", "") for r in records])
    except httpx.HTTPStatusError as e:
        return f"知识库检索失败: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"知识库检索失败: {str(e)}"