"""Utility functions for handling tool calls and results."""

import json
from typing import Optional
from llama_index.core.tools.function_tool import FunctionTool
from common.llm.models import TextChunk
from loguru import logger


def check_and_handle_return_direct(
    tool_obj: FunctionTool,
    tool_name: str,
    tool_content: Optional[str],
    tool_error: Optional[str],
) -> Optional[TextChunk]:
    """
    Check if a tool has return_direct=True and format the result accordingly.

    Args:
        tool_obj: The FunctionTool object
        tool_name: Name of the tool
        tool_content: Content returned by the tool (None if error)
        tool_error: Error message if tool call failed (None if success)
        agent_name: Name of the agent (for logging)

    Returns:
        TextChunk if return_direct=True, None otherwise
    """
    return_direct = getattr(tool_obj.metadata, 'return_direct', False)

    if not return_direct:
        return None

    logger.info(f"Tool {tool_name} has return_direct=True, returning tool result directly.")

    if tool_error:
        return TextChunk(delta="Tool call failed: {tool_error}")

    if not tool_content:
        return TextChunk(delta="Tool call successful, but no content returned.")

    try:
        result_data = json.loads(tool_content)
        if isinstance(result_data, dict) and "result" in result_data:
            # Format FAQ results or similar structured results
            formatted_result = ""
            for item in result_data.get("result", []):
                if isinstance(item, dict):
                    content = item.get("content", "")
                    if content:
                        formatted_result += content + "\n\n"
            if formatted_result:
                return TextChunk(delta=formatted_result.strip())
            else:
                return TextChunk(delta=tool_content)
        else:
            return TextChunk(delta=tool_content)
    except (json.JSONDecodeError, Exception):
        return TextChunk(delta=tool_content)
