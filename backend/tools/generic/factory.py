"""Factory for creating generic tools used by skills.

Provides a single entry point to get all generic tools that serve as
the "hands" for declarative skills.
"""

from typing import List, Set
from llama_index.core.tools import FunctionTool
from loguru import logger

from tools.generic.run_command import create_run_command_tool
from tools.generic.check_env import create_check_env_tool, create_check_prerequisites_tool
from tools.generic.http_request import create_http_request_tool
from tools.generic.file_ops import create_read_file_tool, create_write_file_tool, create_list_directory_tool


# All available generic tool creators, keyed by tool name
_GENERIC_TOOL_REGISTRY = {
    "run_command": create_run_command_tool,
    "check_env": create_check_env_tool,
    "check_prerequisites": create_check_prerequisites_tool,
    "http_request": create_http_request_tool,
    "read_file": create_read_file_tool,
    "write_file": create_write_file_tool,
    "list_directory": create_list_directory_tool,
}


def create_generic_tools(
    tool_names: List[str] = None,
    exclude: Set[str] = None,
) -> List[FunctionTool]:
    """Create generic tools, optionally filtered by name.

    Args:
        tool_names: If provided, only create these tools. If None, create all.
        exclude: Set of tool names to exclude (applied after tool_names filter).

    Returns:
        List of FunctionTool instances.
    """
    exclude = exclude or set()

    if tool_names:
        names_to_create = [n for n in tool_names if n in _GENERIC_TOOL_REGISTRY and n not in exclude]
    else:
        names_to_create = [n for n in _GENERIC_TOOL_REGISTRY if n not in exclude]

    tools = []
    for name in names_to_create:
        try:
            tool = _GENERIC_TOOL_REGISTRY[name]()
            tools.append(tool)
            logger.debug(f"Created generic tool: {name}")
        except Exception as e:
            logger.warning(f"Failed to create generic tool '{name}': {e}")

    return tools


def get_tools_required_by_skills(skill_tool_names: List[str]) -> List[FunctionTool]:
    """Create only the generic tools that are required by the given skill declarations.

    This is used to selectively add tools based on what the enabled skills need.

    Args:
        skill_tool_names: Combined list of tool names from all enabled skills.

    Returns:
        List of FunctionTool instances (deduplicated).
    """
    # Deduplicate and filter to only known generic tools
    unique_names = list(dict.fromkeys(
        name for name in skill_tool_names if name in _GENERIC_TOOL_REGISTRY
    ))
    return create_generic_tools(tool_names=unique_names)


def get_all_generic_tool_names() -> List[str]:
    """Get all available generic tool names."""
    return list(_GENERIC_TOOL_REGISTRY.keys())
