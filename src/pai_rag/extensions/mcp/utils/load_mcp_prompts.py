from typing import Any, Optional
from mcp import ClientSession
from mcp.types import PromptMessage


async def load_mcp_prompt(
    session: ClientSession, name: str, arguments: Optional[dict[str, Any]] = None
) -> list[PromptMessage]:
    """Load MCP prompt and convert to LangChain messages."""
    response = await session.get_prompt(name, arguments)
    return response.messages
