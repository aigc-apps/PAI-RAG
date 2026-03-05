"""Run Command Tool - Execute shell commands in a controlled manner.

This tool allows skills to run terminal commands, install dependencies,
execute scripts, etc. It includes safety measures like timeout and
command filtering.
"""

import asyncio
import os
import shlex
from typing import Optional
from loguru import logger
from llama_index.core.tools import FunctionTool


# Commands that are completely blocked for safety
BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero",
    ":(){:|:&};:",
}

# Maximum output size in characters
MAX_OUTPUT_SIZE = 50000

# Default timeout in seconds
DEFAULT_TIMEOUT = 60


def _is_safe_command(command: str) -> bool:
    """Basic safety check for shell commands."""
    normalized = command.strip().lower()
    for blocked in BLOCKED_COMMANDS:
        if blocked in normalized:
            return False
    return True


async def _run_command(
    command: str,
    working_directory: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
        working_directory: Optional working directory for the command.
            Defaults to the current working directory.
        timeout: Maximum execution time in seconds (default: 60).

    Returns:
        A string containing the command output (stdout + stderr),
        exit code, and execution status.
    """
    if not command or not command.strip():
        return "Error: Empty command provided."

    if not _is_safe_command(command):
        return f"Error: Command blocked for safety reasons: {command}"

    cwd = working_directory or os.getcwd()

    logger.info(f"Executing command: {command} (cwd: {cwd}, timeout: {timeout}s)")

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"Error: Command timed out after {timeout} seconds.\nCommand: {command}"

        stdout_str = stdout.decode("utf-8", errors="replace").strip()
        stderr_str = stderr.decode("utf-8", errors="replace").strip()
        exit_code = process.returncode

        # Build result
        parts = []
        parts.append(f"Exit Code: {exit_code}")

        if stdout_str:
            if len(stdout_str) > MAX_OUTPUT_SIZE:
                stdout_str = stdout_str[:MAX_OUTPUT_SIZE] + "\n...(output truncated)"
            parts.append(f"STDOUT:\n{stdout_str}")

        if stderr_str:
            if len(stderr_str) > MAX_OUTPUT_SIZE:
                stderr_str = stderr_str[:MAX_OUTPUT_SIZE] + "\n...(output truncated)"
            parts.append(f"STDERR:\n{stderr_str}")

        if not stdout_str and not stderr_str:
            parts.append("(no output)")

        result = "\n\n".join(parts)
        logger.info(f"Command completed with exit code {exit_code}")
        return result

    except Exception as e:
        error_msg = f"Error executing command: {e}"
        logger.error(error_msg)
        return error_msg


def create_run_command_tool() -> FunctionTool:
    """Create the run_command FunctionTool."""
    return FunctionTool.from_defaults(
        async_fn=_run_command,
        name="run_command",
        description=(
            "Execute a shell command in the terminal and return the output. "
            "Use this tool to run system commands, install packages, execute scripts, "
            "check system status, etc. "
            "Parameters: command (str, required) - the shell command to execute; "
            "working_directory (str, optional) - the directory to run the command in; "
            "timeout (int, optional) - maximum execution time in seconds (default: 60)."
        ),
        return_direct=False,
    )
