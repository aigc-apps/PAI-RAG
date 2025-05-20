from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
import json
import os
from bs4 import BeautifulSoup
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn

from mcp.server.fastmcp.prompts import Prompt  
from mcp.server.fastmcp.prompts.base import PromptArgument  

from textwrap import dedent  
from aliyun_search_tool import AliyunSearchTool
        
load_dotenv()

mcp = FastMCP("AliyunSearchMcp")

tool = AliyunSearchTool(access_key_id=os.getenv("ACCESS_KEY_ID"), access_key_secret=os.getenv("ACCESS_KEY_SECRET"))
@mcp.tool()
async def search_web(query: str):
    """
    从网络上搜索给定查询的最新内容。

    参数:
    query: 要搜索的查询 (例如 "React Agent")

    返回:
    网页内容
    """
    res = await tool.aquery(query)

    return res



## sse传输
def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

if __name__ == "__main__":
    mcp_server = mcp._mcp_server

    import argparse

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8020, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)