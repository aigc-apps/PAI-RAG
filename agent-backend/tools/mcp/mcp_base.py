from typing import Any, Callable, List, Optional
from mcp.client.session import ClientSession

from llama_index.tools.mcp.base import McpToolSpec
import mcp.types as types


class McpToolUtils(McpToolSpec):
    """
    McpToolUtils will get the tools from MCP Client (only need to implement ClientSession) and convert them to LlamaIndex's FunctionTool objects.
    Args:
        client: An MCP client instance implementing ClientSession, and it should support the following methods in ClientSession:
            - list_tools: List all tools.
            - call_tool: Call a tool.
        allowed_tools: If set, only return tools with the specified names.
    """

    def __init__(
        self,
        mcp_server_name: str,
        client: ClientSession,
        allowed_tools: Optional[List[str]] = None,
    ) -> None:
        self.client = client
        self.allowed_tools = allowed_tools if allowed_tools is not None else []
        self.mcp_server_name = mcp_server_name

    async def fetch_tools(self) -> List[Any]:
        """
        An asynchronous method to get the tools list from MCP Client. If allowed_tools is set, it will filter the tools.
        Returns:
            A list of tools, each tool object needs to contain name, description, inputSchema properties.
        """
        response = await self.client.list_tools()
        tools = response.tools if hasattr(response, "tools") else []
        if self.allowed_tools:
            tools = [
                types.Tool(
                    name=self.mcp_server_name + ":" + tool.name,
                    description=tool.description,
                    inputSchema=tool.inputSchema,
                )
                for tool in tools
                if tool.name in self.allowed_tools
            ]
        else:
            tools = [
                types.Tool(
                    name=self.mcp_server_name + ":" + tool.name,
                    description=tool.description,
                    inputSchema=tool.inputSchema,
                )
                for tool in tools
            ]
        return tools

    def _create_tool_fn(self, tool_name: str) -> Callable:
        """
        Create a tool call function for a specified MCP tool name. The function internally wraps the call_tool call to the MCP Client.
        """

        async def async_tool_fn(**kwargs):
            mcp_tool_name = tool_name.removeprefix(f"{self.mcp_server_name}:")
            return await self.client.call_tool(mcp_tool_name, kwargs)

        return async_tool_fn
