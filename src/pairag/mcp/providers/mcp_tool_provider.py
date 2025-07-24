from typing import Dict, List
from sqlmodel import Field, select
from pairag.db.encrypt_utils import decrypt_key
from pairag.db.models.mcp import McpServerCreate, McpServerEntity
from pairag.db.db_context import with_async_db_session
from pairag.mcp.providers.base_provider import BaseConfigProvider
from pairag.mcp.providers.mcp_client import BasicMCPClient
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


async def create_mcp_tools(mcp_server_configs: List[McpServerEntity]):
    mcp_clients = []
    for mcp_server_config in mcp_server_configs:
        if mcp_server_config.enabled:
            mcp_headers = {}
            auth_token = decrypt_key(mcp_server_config.encrypted_auth_token)
            if auth_token:
                mcp_headers = {
                    "Authorization": "Bearer " + auth_token
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


    logger.info("Created mcp tools from server config.")
    return mcp_tools_map


class McpToolProvider(BaseConfigProvider):
    name_to_entry_id: Dict[str, str] = Field(default={})

    def _load_entries(self, entries):
        super()._load_entries(entries)
        for entry_id, entry in self.config_map.items():
            self.name_to_entry_id[entry.name] = entry_id

    async def get_mcp_tools_async(self, mcp_server_name_list: List[str]) -> List[FunctionTool]:
        all_tools = []
        for mcp_server_name in mcp_server_name_list:
            tools = self.instance_map.get(mcp_server_name)
            if tools is None:
                mcp_id = self.name_to_entry_id[mcp_server_name]
                tools = await self._create_instance_async(self.config_map[mcp_id])
                self.instance_map.put(mcp_server_name, tools)
            all_tools.extend(tools)
            logger.info(f"Get {len(tools)} for mcp {mcp_server_name}.")
        return all_tools

    async def _create_instance_async(self, config: McpServerEntity):
        if config.enabled:
            mcp_headers = {}
            auth_token = decrypt_key(config.encrypted_auth_token)
            if auth_token:
                mcp_headers = {
                    "Authorization": "Bearer " + auth_token
                }
            mcp_client = BasicMCPClient(
                name=config.name,
                command_or_url=config.url,
                headers=mcp_headers,
            )

            mcp_tool_spec = McpToolSpec(client=mcp_client)

            mcp_tools = []
            # TODO: can retrieve in parallel with asyncio.gather
            try:
                tools: List[FunctionTool] = await mcp_tool_spec.to_tool_list_async()
                for tool in tools:
                    # transform tool name to server_name-tool_name
                    # 尽管存在mcp server为a,tool name 为b-c和server为a-b, tool为c的小概率撞车情形
                    # 考虑到大部分tool命名规则以及撞车概率极小，故忽略此情形（真撞车的话说明这俩mcp事实上重复了）
                    # 若因为碰撞报错，则建议用户修改server name 或者不用某个tool
                    tool.metadata.name = f"{config.name}-{tool.metadata.name}"
                    mcp_tools.append(tool)
            except Exception as e:
                # it happens when mcp server is not reachable. just log it without throwing
                logger.error(f"Failed to create mcp tools for {config.name}: {e}")
            return mcp_tools
        else:
            return []


mcp_provider = McpToolProvider()
