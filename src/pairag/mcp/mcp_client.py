from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters
from loguru import logger
from pydantic import BaseModel
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from typing import List, Optional
import httpx


# MCP 客户端配置
class MCPServerConfig(BaseModel):
    id: str
    name: str
    url: str
    auth_token: Optional[str] = None
    type: str = "sse"
    active: Optional[bool] = False


class BasicMCPClient:
    """
    Basic MCP client that can be used to connect to an MCP server.
    This is useful for verifying that the MCP server which implements `FastMCP` is working.
    """

    def __init__(
        self,
        name: str,
        command_or_url: str,
        args: list[str] = [],
        env: dict[str, str] = {},
        headers: dict[str, str] = {},
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
    ):
        self.name = name
        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.headers = headers
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout

    @asynccontextmanager
    async def _run_session(self):
        if urlparse(self.command_or_url).scheme in ("http", "https"):
            async with sse_client(
                url=self.command_or_url,
                headers=self.headers,
                timeout=self.timeout,
                sse_read_timeout=self.sse_read_timeout,
            ) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    yield session
        else:
            server_parameters = StdioServerParameters(
                command=self.command_or_url, args=self.args, env=self.env
            )
            async with stdio_client(server_parameters) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    yield session

    async def call_tool(self, tool_name: str, arguments: dict):
        async with self._run_session() as session:
            return await session.call_tool(tool_name, arguments)

    async def list_tools(self):
        async with self._run_session() as session:
            return await session.list_tools()


async def fetch_mcp_servers():
    mcp_servers = []
    try:
        port = 8680
        logger.info(f"/api/chat BACKEND_PORT {port}")

        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:{port}/api/configs")
            response.raise_for_status()

            config_data = response.json()

            mcp_servers = [
                MCPServerConfig(**item)
                for item in config_data.get("mcp_config", [])
                if item.get("active")
            ]
    except httpx.HTTPError as fetch_error:
        logger.exception("Failed to fetch MCP server configurations")
        raise fetch_error

    return mcp_servers


async def resolve_mcp_clients() -> List[BasicMCPClient]:
    mcp_clients = []
    mcp_server_configs = await fetch_mcp_servers()
    for mcp_server_config in mcp_server_configs:
        if mcp_server_config.active:
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
    return mcp_clients
