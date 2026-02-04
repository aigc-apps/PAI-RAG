from typing import List
from db.models.mcp import McpServerEntity
from common.encrypt_utils import decrypt_key
from tools.utils.mcp_client import BasicMCPClient
from llama_index.tools.mcp.base import McpToolSpec
from llama_index.core.tools.function_tool import FunctionTool
from utils.lru_cache import LruCache
import traceback
from loguru import logger


mcp_cache = LruCache(max_size=100)

async def create_mcp_tools_async(config: McpServerEntity) -> List[FunctionTool]:
    mcp_key = f"mcp-{config.url}-{config.encrypted_auth_token or 'none'}"
    mcp_tools = mcp_cache.get(mcp_key)
    if mcp_tools:
        return mcp_tools

    mcp_headers = {}
    auth_token = decrypt_key(config.encrypted_auth_token)
    if auth_token:
        mcp_headers = {"Authorization": "Bearer " + auth_token}

    mcp = BasicMCPClient(
        name=config.name,
        command_or_url=config.url,
        headers=mcp_headers,
        connection_type=config.type if hasattr(config, 'type') and config.type else "streamable_http"
    )
    mcp_tool_spec = McpToolSpec(client=mcp)
    mcp_tools = []
    try:
        tools: List[FunctionTool] = await mcp_tool_spec.to_tool_list_async()
        for tool in tools:
            # transform tool name to server_name-tool_name
            # 尽管存在mcp server为a,tool name 为b-c和server为a-b, tool为c的小概率撞车情形
            # 考虑到大部分tool命名规则以及撞车概率极小，故忽略此情形（真撞车的话说明这俩mcp事实上重复了）
            # 若因为碰撞报错，则建议用户修改server name 或者不用某个tool
            tool.metadata.name = f"{config.name}-{tool.metadata.name}"
            mcp_tools.append(tool)
    except Exception:
        # it happens when mcp server is not reachable. just log it without throwing
        logger.error(f"Failed to create mcp tools for {config.name}: {traceback.format_exc()}")

    if mcp_tools:
        mcp_cache.put(mcp_key, mcp_tools)
    return mcp_tools
