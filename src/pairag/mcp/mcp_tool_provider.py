from typing import Dict, List

from sqlmodel import select
from pairag.db.encrypt_utils import decrypt_key
from pairag.db.models import McpServerCreate, McpServerEntity
from pairag.db.db_context import with_async_db_session
from pairag.mcp.mcp_client import BasicMCPClient
from llama_index.tools.mcp.base import McpToolSpec
from llama_index.core.tools.function_tool import FunctionTool
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger


@with_async_db_session
async def fetch_mcp_tools(session: AsyncSession):
    logger.info("[McpProvider] Start fetching mcp servers.")
    sql_results = await session.exec(select(McpServerEntity))
    mcp_results = sql_results.all()
    mcp_server_configs = [
        McpServerCreate(
            name=mcp.name,
            enabled=mcp.enabled,
            auth_token=decrypt_key(mcp.encrypted_auth_token),
            type=mcp.type,
            url=mcp.url,
        )
        for mcp in mcp_results
    ]
    logger.info(f"[McpProvider] fetched {len(mcp_server_configs)} mcp servers.")

    return await create_mcp_tools(mcp_server_configs)


async def create_mcp_tools(mcp_server_configs: List[McpServerCreate]):
    mcp_clients = []
    for mcp_server_config in mcp_server_configs:
        if mcp_server_config.enabled:
            mcp_headers = {}
            if mcp_server_config.auth_token:
                mcp_headers = {
                    "Authorization": "Bearer " + mcp_server_config.auth_token
                }
            mcp_client = BasicMCPClient(
                name=mcp_server_config.name,
                command_or_url=mcp_server_config.url,
                headers=mcp_headers,
            )
            mcp_clients.append(mcp_client)

    mcp_tools_map: Dict[str, List[FunctionTool]] = {}

    for mcp_client in mcp_clients:
        mcp_server_name = mcp_client.name
        mcp_tools_map[mcp_server_name] = []
        mcp_tool = McpToolSpec(client=mcp_client)

        # TODO: can retrieve in parallel with asyncio.gather
        tools: List[FunctionTool] = await mcp_tool.to_tool_list_async()
        for tool in tools:
            # transform tool name to server_name-tool_name
            # 尽管存在mcp server为a,tool name 为b-c和server为a-b, tool为c的小概率撞车情形
            # 考虑到大部分tool命名规则以及撞车概率极小，故忽略此情形（真撞车的话说明这俩mcp事实上重复了）
            # 若因为碰撞报错，则建议用户修改server name 或者不用某个tool
            tool.metadata.name = f"{mcp_server_name}-{tool.metadata.name}"
            mcp_tools_map[mcp_server_name].append(tool)

    logger.info("Created mcp tools from server config.")
    return mcp_tools_map


class McpToolProvider:
    def __init__(self):
        # empty map
        self.mcp_tools_map = {}

    async def refresh(self):
        self.mcp_tools_map = await fetch_mcp_tools()

    def get_mcp_tools(self, mcp_server_name_list: List[str]) -> List[FunctionTool]:
        tools = []
        for mcp_server_name in mcp_server_name_list:
            if mcp_server_name not in self.mcp_tools_map:
                logger.warning(f"MCP server {mcp_server_name} not found")
            else:
                tools.extend(self.mcp_tools_map[mcp_server_name])
        return tools


mcp_provider = McpToolProvider()
