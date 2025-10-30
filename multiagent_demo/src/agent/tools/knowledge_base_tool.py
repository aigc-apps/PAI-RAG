# tools/knowledge_base_tool.py
import httpx
from langchain_core.tools import tool

@tool
async def knowledge_base_search(query: str) -> str:
    """
    通过向指定知识库 API 发送请求，检索与 query 相关的知识。
    
    Args:
        query (str): 用户的检索关键词或问题
        
    Returns:
        str: 知识库返回的文本结果，若失败则返回错误信息
    """
    url = "http://rag.xxxx.cn-hangzhou.pai-eas.aliyuncs.com/v1/retrieval"
    
    # Modify the authorization!!
    headers = {
        "Content-Type": "application/json",
        "Authorization": "xxxxxx=="
    }
    payload = {"knowledge_id": "kbc5f1a19c3fd0489fa48943850a59ced4","query": query}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])
            if not records:
                return "知识库中未找到相关信息。"
            return "\n\n".join([r.get("content", "") for r in records])
    except Exception as e:
        return f"知识库检索失败: {str(e)}"