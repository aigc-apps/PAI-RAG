from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

from urllib.parse import urlparse
from contextlib import asynccontextmanager
from llama_index.tools.mcp import BasicMCPClient


class PaiBasicMCPClient(BasicMCPClient):
    """
    Basic MCP client that can be used to connect to an MCP server.

    This is useful for verifying that the MCP server which implements `FastMCP` is working.
    """

    def __init__(
        self,
        command_or_url: str,
        args: list[str] = [],
        env: dict[str, str] = {},
        headers: dict[str, str] = {},
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
    ):
        super().__init__(
            command_or_url=command_or_url,
            args=args,
            env=env,
        )
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
