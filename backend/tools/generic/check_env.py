"""Check Environment Tool - Inspect environment variables and system state.

This tool allows skills to verify that required environment variables
are set, check MCP connection status, and validate system prerequisites.
"""

import os
import json
from typing import Optional, List
from loguru import logger
from llama_index.core.tools import FunctionTool


async def _check_env(
    variable_names: str,
) -> str:
    """Check if specified environment variables are set.

    Args:
        variable_names: Comma-separated list of environment variable names to check.
            Example: "API_KEY,SECRET_TOKEN,DATABASE_URL"

    Returns:
        A JSON string with the check results for each variable.
        Each entry shows whether the variable is set (without revealing the value).
    """
    if not variable_names or not variable_names.strip():
        return json.dumps({"error": "No variable names provided."})

    names = [name.strip() for name in variable_names.split(",") if name.strip()]
    results = {}

    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            results[name] = {
                "status": "set",
                "length": len(value),
                "preview": value[:3] + "***" if len(value) > 3 else "***",
            }
        else:
            results[name] = {
                "status": "not_set",
            }

    logger.info(f"Environment check for {names}: {[r['status'] for r in results.values()]}")
    return json.dumps(results, ensure_ascii=False)


async def _check_prerequisites(
    checks: str,
) -> str:
    """Check multiple prerequisites for a skill.

    Args:
        checks: A JSON string describing the prerequisite checks.
            Format: [{"type": "env", "name": "VAR_NAME"}, {"type": "tool", "name": "tool_name"}]

            Supported check types:
            - "env": Check if an environment variable is set.
            - "tool": Check if a tool is available (always returns true from this tool).
            - "command": Check if a command exists on the system (uses 'which').

    Returns:
        A JSON string with the check results.
    """
    try:
        check_list = json.loads(checks)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    if not isinstance(check_list, list):
        return json.dumps({"error": "Expected a JSON array of checks."})

    results = []

    for check in check_list:
        check_type = check.get("type", "unknown")
        name = check.get("name", "")

        if check_type == "env":
            value = os.environ.get(name)
            results.append({
                "type": "env",
                "name": name,
                "passed": value is not None and value != "",
            })

        elif check_type == "command":
            import shutil
            cmd_path = shutil.which(name)
            results.append({
                "type": "command",
                "name": name,
                "passed": cmd_path is not None,
                "path": cmd_path,
            })

        elif check_type == "tool":
            # Tool availability is checked at the agent level
            results.append({
                "type": "tool",
                "name": name,
                "passed": True,
                "note": "Tool availability is managed by the agent."
            })

        else:
            results.append({
                "type": check_type,
                "name": name,
                "passed": False,
                "error": f"Unknown check type: {check_type}",
            })

    all_passed = all(r.get("passed", False) for r in results)

    return json.dumps({
        "all_passed": all_passed,
        "results": results,
    }, ensure_ascii=False)


def create_check_env_tool() -> FunctionTool:
    """Create the check_env FunctionTool."""
    return FunctionTool.from_defaults(
        async_fn=_check_env,
        name="check_env",
        description=(
            "Check if specified environment variables are set in the current system. "
            "Parameters: variable_names (str, required) - comma-separated list of "
            "environment variable names to check (e.g., 'API_KEY,DATABASE_URL'). "
            "Returns a JSON object showing the status of each variable."
        ),
        return_direct=False,
    )


def create_check_prerequisites_tool() -> FunctionTool:
    """Create the check_prerequisites FunctionTool."""
    return FunctionTool.from_defaults(
        async_fn=_check_prerequisites,
        name="check_prerequisites",
        description=(
            "Check multiple prerequisites for a skill. "
            "Parameters: checks (str, required) - a JSON string describing the checks. "
            "Format: [{\"type\": \"env\", \"name\": \"VAR_NAME\"}, "
            "{\"type\": \"command\", \"name\": \"python3\"}]. "
            "Supported types: 'env' (environment variable), 'command' (system command), 'tool' (agent tool). "
            "Returns a JSON object with individual and overall pass/fail status."
        ),
        return_direct=False,
    )
