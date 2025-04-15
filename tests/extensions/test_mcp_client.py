from pai_rag.extensions.mcp.mcp_client import MultiServerMCPClient
from pai_rag.core.rag_module import resolve
import pytest
from pathlib import Path
import os
from openai import OpenAI

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


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_GPU_TESTS", "false") == "true",
    reason="Need to execute in a CUDA environment.",
)


config_instance = McpServerConfig(
    name="amaps",
    url="https://mcp-server-amap-jitptfyoyw.cn-hangzhou.fcapp.run/sse",
    transport="sse",
)


mcp_servers_connections = [config_instance]


@pytest.fixture(scope="module", autouse=True)
async def process_query() -> str:
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
    await mcp_client.__aexit__(None, None, None)
