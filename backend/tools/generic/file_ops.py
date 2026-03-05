"""File Operations Tool - Read and write files for skill use.

Allows skills to read configuration files, write outputs, etc.
"""

import os
import json
from pathlib import Path
from typing import Optional
from loguru import logger
from llama_index.core.tools import FunctionTool

# Maximum file read size
MAX_READ_SIZE = 200000
# Allowed base directories (for security)
# Empty means allow all paths — can be restricted per deployment
ALLOWED_BASE_DIRS = []


def _is_path_allowed(file_path: str) -> bool:
    """Check if a file path is within allowed directories."""
    if not ALLOWED_BASE_DIRS:
        return True
    abs_path = os.path.abspath(file_path)
    return any(abs_path.startswith(os.path.abspath(d)) for d in ALLOWED_BASE_DIRS)


async def _read_file(
    file_path: str,
    encoding: str = "utf-8",
) -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read.
        encoding: File encoding (default: utf-8).

    Returns:
        The file contents as a string, or an error message.
    """
    if not file_path:
        return "Error: file_path is required."

    if not _is_path_allowed(file_path):
        return f"Error: Access denied for path: {file_path}"

    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    if not path.is_file():
        return f"Error: Not a file: {file_path}"

    try:
        content = path.read_text(encoding=encoding)
        if len(content) > MAX_READ_SIZE:
            content = content[:MAX_READ_SIZE] + "\n...(file truncated)"
        logger.info(f"Read file: {file_path} ({len(content)} chars)")
        return content
    except Exception as e:
        return f"Error reading file: {e}"


async def _write_file(
    file_path: str,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True,
) -> str:
    """Write content to a file.

    Args:
        file_path: Path to the file to write.
        content: The content to write.
        encoding: File encoding (default: utf-8).
        create_dirs: Whether to create parent directories if they don't exist (default: True).

    Returns:
        Success message or error message.
    """
    if not file_path:
        return "Error: file_path is required."
    if content is None:
        return "Error: content is required."

    if not _is_path_allowed(file_path):
        return f"Error: Access denied for path: {file_path}"

    path = Path(file_path)

    try:
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding=encoding)
        logger.info(f"Wrote file: {file_path} ({len(content)} chars)")
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def _list_directory(
    dir_path: str,
) -> str:
    """List the contents of a directory.

    Args:
        dir_path: Path to the directory.

    Returns:
        A formatted listing of directory contents, or an error message.
    """
    if not dir_path:
        return "Error: dir_path is required."

    if not _is_path_allowed(dir_path):
        return f"Error: Access denied for path: {dir_path}"

    path = Path(dir_path)
    if not path.exists():
        return f"Error: Directory not found: {dir_path}"
    if not path.is_dir():
        return f"Error: Not a directory: {dir_path}"

    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries[:200]:  # Limit to 200 entries
            entry_type = "DIR" if entry.is_dir() else "FILE"
            size = entry.stat().st_size if entry.is_file() else "-"
            lines.append(f"[{entry_type}] {entry.name}  ({size})")

        if len(entries) > 200:
            lines.append(f"... and {len(entries) - 200} more entries")

        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {e}"


def create_read_file_tool() -> FunctionTool:
    """Create the read_file FunctionTool."""
    return FunctionTool.from_defaults(
        async_fn=_read_file,
        name="read_file",
        description=(
            "Read the contents of a file. "
            "Parameters: file_path (str, required) - path to the file; "
            "encoding (str, optional) - file encoding (default: utf-8). "
            "Returns the file contents as text."
        ),
        return_direct=False,
    )


def create_write_file_tool() -> FunctionTool:
    """Create the write_file FunctionTool."""
    return FunctionTool.from_defaults(
        async_fn=_write_file,
        name="write_file",
        description=(
            "Write content to a file. Creates parent directories if needed. "
            "Parameters: file_path (str, required) - path to the file; "
            "content (str, required) - text content to write; "
            "encoding (str, optional) - file encoding (default: utf-8); "
            "create_dirs (bool, optional) - create parent directories (default: true). "
            "Returns success or error message."
        ),
        return_direct=False,
    )


def create_list_directory_tool() -> FunctionTool:
    """Create the list_directory FunctionTool."""
    return FunctionTool.from_defaults(
        async_fn=_list_directory,
        name="list_directory",
        description=(
            "List the contents of a directory. "
            "Parameters: dir_path (str, required) - path to the directory. "
            "Returns a formatted listing with file/directory types and sizes."
        ),
        return_direct=False,
    )
