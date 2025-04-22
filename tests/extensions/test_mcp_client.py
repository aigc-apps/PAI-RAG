import pytest
import os
from pai_rag.extensions.mcp.mcp_client import PaiBasicMCPClient
from pai_rag.extensions.mcp.mcp_base import McpToolSpec

from pai_rag.core.models.config import McpServerConfig


if "OPENAI_API_KEY" not in os.environ or os.getenv("SKIP_GPU_TESTS", "false") == "true":
    pytest.skip(
        allow_module_level=True,
        reason='Environment variable "OPENAI_API_KEY" not set.',
    )


config_instance = McpServerConfig(
    name="amaps",
    url="https://mcp-server-amap-jitptfyoyw.cn-hangzhou.fcapp.run/sse",
    transport="sse",
)


mcp_servers_connections = [config_instance]


@pytest.mark.asyncio
async def test_mcp_client() -> str:
    for mcp_server_config in mcp_servers_connections:
        if mcp_server_config.auth_token:
            mcp_client = PaiBasicMCPClient(
                command_or_url=mcp_server_config.url,
                headers={"Authorization": "Bearer " + mcp_server_config.auth_token},
            )
        else:
            mcp_client = PaiBasicMCPClient(command_or_url=mcp_server_config.url)
        mcp_tool = McpToolSpec(
            mcp_server_name=mcp_server_config.name, client=mcp_client
        )
        tools = mcp_tool.to_tool_list()

        assert len(tools) > 0
