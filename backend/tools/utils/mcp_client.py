from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from contextlib import asynccontextmanager
from datetime import timedelta
from loguru import logger


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
        connection_type: str = "streamable_http",
    ):
        self.name = name
        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.headers = headers
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.connection_type = connection_type

    @asynccontextmanager
    async def _run_session(self):
        read_timeout = timedelta(seconds=self.sse_read_timeout)

        if self.connection_type == "sse":
            logger.info(f"Connecting to MCP server '{self.name}' via SSE: {self.command_or_url}")
            try:
                async with sse_client(
                    url=self.command_or_url,
                    headers=self.headers,
                    timeout=self.timeout,
                    sse_read_timeout=self.sse_read_timeout,
                ) as streams:
                    async with ClientSession(
                        read_stream=streams[0],
                        write_stream=streams[1],
                        read_timeout_seconds=read_timeout,
                    ) as session:
                        logger.info(f"Initializing MCP session for '{self.name}'")
                        await session.initialize()
                        logger.info(f"MCP session initialized successfully for '{self.name}'")
                        yield session
            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{self.name}' via SSE: {e}")
                raise
        else:
            logger.info(f"Connecting to MCP server '{self.name}' via Streamable HTTP: {self.command_or_url}")
            try:
                async with streamablehttp_client(
                    url=self.command_or_url,
                    headers=self.headers,
                    timeout=self.timeout,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(
                        read_stream=read_stream,
                        write_stream=write_stream
                    ) as session:
                        logger.info(f"Initializing MCP session for '{self.name}'")
                        await session.initialize()
                        logger.info(f"MCP session initialized successfully for '{self.name}'")
                        yield session
            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{self.name}' via Streamable HTTP: {e}")
                raise

    async def call_tool(self, tool_name: str, arguments: dict):
        logger.debug(f"Calling tool '{tool_name}' on MCP server '{self.name}' with arguments: {arguments}")
        try:
            async with self._run_session() as session:
                result = await session.call_tool(tool_name, arguments)
                logger.debug(f"Tool '{tool_name}' returned successfully")
                return result
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}' on MCP server '{self.name}': {e}")
            raise

    async def list_tools(self):
        logger.debug(f"Listing tools from MCP server '{self.name}'")
        try:
            async with self._run_session() as session:
                tools = await session.list_tools()
                logger.info(f"Successfully listed {len(tools.tools) if hasattr(tools, 'tools') else 0} tools from MCP server '{self.name}'")
                return tools
        except Exception as e:
            logger.error(f"Failed to list tools from MCP server '{self.name}': {e}")
            raise
