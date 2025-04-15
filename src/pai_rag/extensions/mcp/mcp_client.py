from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Literal, Optional, TypedDict, cast
from mcp import ClientSession


from mcp.client.sse import sse_client
from pai_rag.extensions.mcp.utils.load_mcp_tools import (
    load_mcp_tools,
    mcp_tools_server_info,
)
from pai_rag.extensions.mcp.utils.load_mcp_prompts import load_mcp_prompt
from mcp.types import PromptMessage


DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5


class SSEConnection(TypedDict):
    transport: Literal["sse"]

    url: str
    """The URL of the SSE endpoint to connect to."""

    headers: dict[str, Any] | None
    """HTTP headers to send to the SSE endpoint"""

    timeout: float
    """HTTP timeout"""

    sse_read_timeout: float
    """SSE read timeout"""

    session_kwargs: dict[str, Any] | None
    """Additional keyword arguments to pass to the ClientSession"""


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers and loading LangChain-compatible tools from them."""

    def __init__(self, connections: dict[str, SSEConnection] | None = None) -> None:
        """Initialize a MultiServerMCPClient with MCP servers connections.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                Each configuration can be SSEConnection.
                If None, no initial connections are established.

        """
        self.connections: dict[str, SSEConnection] = connections or {}
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.server_name_to_tools: dict[str, list[dict]] = {}
        self.tools_to_server_name: dict[str, str] = {}

    async def _initialize_session_and_load_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Initialize a session and load tools from it.

        Args:
            server_name: Name to identify this server connection
            session: The ClientSession to initialize
        """
        # Initialize the session
        await session.initialize()
        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(server_name, session)
        self.server_name_to_tools[server_name] = server_tools
        self.tools_to_server_name.update(
            await mcp_tools_server_info(server_name, session)
        )

    async def connect_to_server(
        self,
        server_name: str,
        *,
        transport: Literal["stdio", "sse"] = "sse",
        **kwargs,
    ) -> None:
        """Connect to an MCP server using SSE.

        This is a generic method that calls connect_to_server_via_sse
        based on the provided transport parameter.

        Args:
            server_name: Name to identify this server connection
            transport: Type of transport to use ("stdio" or "sse"), defaults to "sse"
            **kwargs: Additional arguments to pass to the specific connection method

        Raises:
            ValueError: If transport is not recognized
            ValueError: If required parameters for the specified transport are missing
        """
        if transport == "sse":
            if "url" not in kwargs:
                raise ValueError("'url' parameter is required for SSE connection")
            await self.connect_to_server_via_sse(
                server_name,
                url=kwargs["url"],
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout", DEFAULT_HTTP_TIMEOUT),
                sse_read_timeout=kwargs.get(
                    "sse_read_timeout", DEFAULT_SSE_READ_TIMEOUT
                ),
                session_kwargs=kwargs.get("session_kwargs"),
            )
        else:
            raise ValueError(f"Unsupported transport: {transport}. Must 'sse'")

    async def connect_to_server_via_sse(
        self,
        server_name: str,
        *,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
        session_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Connect to a specific MCP server using SSE

        Args:
            server_name: Name to identify this server connection
            url: URL of the SSE server
            headers: HTTP headers to send to the SSE endpoint
            timeout: HTTP timeout
            sse_read_timeout: SSE read timeout
            session_kwargs: Additional keyword arguments to pass to the ClientSession
        """
        # Create and store the connection
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url, headers, timeout, sse_read_timeout)
        )
        read, write = sse_transport
        session_kwargs = session_kwargs or {}
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(
                ClientSession(read, write, **session_kwargs)
            ),
        )

        await self._initialize_session_and_load_tools(server_name, session)

    def get_tools(self) -> list[dict]:
        """Get a list of all tools from all connected servers."""
        all_tools: list[dict] = []
        for server_tools in self.server_name_to_tools.values():
            all_tools.extend(server_tools)
        return all_tools

    async def get_prompt(
        self, server_name: str, prompt_name: str, arguments: Optional[dict[str, Any]]
    ) -> list[PromptMessage]:
        """Get a prompt from a given MCP server."""
        session = self.sessions[server_name]
        return await load_mcp_prompt(session, prompt_name, arguments)

    async def __aenter__(self) -> "MultiServerMCPClient":
        try:
            connections = self.connections or {}
            for server_name, connection in connections.items():
                await self.connect_to_server(server_name, **connection)

            return self
        except Exception:
            await self.exit_stack.aclose()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.exit_stack.aclose()
