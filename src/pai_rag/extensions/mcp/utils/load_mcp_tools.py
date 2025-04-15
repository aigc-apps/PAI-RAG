from mcp import ClientSession


async def load_mcp_tools(server_name: str, session: ClientSession) -> list[dict]:
    """Load all available MCP tools and convert them to LangChain tools."""
    tools = await session.list_tools()
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": server_name + "[pai_rag]" + tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            },
        }
        for tool in tools.tools
    ]
    return available_tools


async def mcp_tools_server_info(server_name: str, session: ClientSession) -> dict:
    """Load all available MCP tools and convert them to LangChain tools."""
    tools_to_server_info = {}
    tools = await session.list_tools()
    for tool in tools.tools:
        server_tool_name = server_name + "[pai_rag]" + tool.name
        tools_to_server_info[server_tool_name] = server_name

    return tools_to_server_info
