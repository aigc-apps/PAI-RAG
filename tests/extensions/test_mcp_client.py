from pai_rag.extensions.mcp.mcp_client import MultiServerMCPClient
from pai_rag.core.rag_module import resolve
import pytest
from pathlib import Path
import os
from openai import OpenAI
import json

from pai_rag.core.models.config import McpServerConfig


if "OPENAI_API_KEY" not in os.environ or os.getenv("SKIP_GPU_TESTS", "false") == "true":
    pytest.skip(
        allow_module_level=True,
        reason='Environment variable "OPENAI_API_KEY" not set.',
    )

llm = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60,
    max_retries=5,
)


BASE_DIR = Path(__file__).parent.parent.parent


config_instance = McpServerConfig(
    name="amaps",
    url="https://mcp-server-amap-jitptfyoyw.cn-hangzhou.fcapp.run/sse",
    transport="sse",
)


mcp_servers_connections = [config_instance]


@pytest.mark.asyncio
async def test_mcp_client() -> str:
    messages = [{"role": "user", "content": "杭州东到西湖怎么走"}]
    connections = {}
    for mcp_server_config in mcp_servers_connections:
        connection = {
            "transport": mcp_server_config.transport or "sse",
            "url": mcp_server_config.url,
        }
        connections[mcp_server_config.name] = connection

    mcp_client = resolve(cls=MultiServerMCPClient, connections=connections)
    mcp_client = await mcp_client.__aenter__()
    available_tools = mcp_client.get_tools()
    response = llm.chat.completions.create(
        model="qwen-max", messages=messages, tools=available_tools, stream=False
    )

    assert response.choices[0].finish_reason == "tool_calls"
    final_text = []
    choice = response.choices[0]
    while choice.finish_reason != "stop":
        choice = response.choices[0]
        messages.append(choice.message.model_dump())
        if choice.finish_reason == "tool_calls":
            for tool in choice.message.tool_calls:
                tool_name = tool.function.name
                tool_args = json.loads(tool.function.arguments)
                server_name = mcp_client.tools_to_server_name[tool_name]
                session = mcp_client.sessions[server_name]
                assert tool_name in [
                    tool["function"]["name"] for tool in available_tools
                ]
                # 还原原始tool_name名称
                tool_name = tool_name.split("[pai_rag]")[1]
                result = await session.call_tool(tool_name, tool_args)
                print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                messages.append(
                    {
                        "role": "tool",
                        "content": result.content[0].text,
                        "tool_call_id": tool.id,
                    }
                )
            response = llm.chat.completions.create(
                model="qwen-max", messages=messages, tools=available_tools, stream=False
            )
            choice = response.choices[0]
            final_text.append(choice.message.content)
        else:
            final_text.append(choice.message.content)

    await mcp_client.__aexit__(None, None, None)
