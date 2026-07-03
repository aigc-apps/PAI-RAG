from __future__ import annotations

import json
from typing import Any, Dict, Optional, Protocol

from agent.tools.base import Tool


class ShellSandboxProvider(Protocol):
    """Subset of the sandbox provider surface: a one-shot shell command runner.

    The sandbox is treated like an MCP that exposes several tools —
    ``code_interpreter`` (run_code) and ``shell`` (run_command) share the same
    warm sandbox instance, mounts, and env contract.
    """

    async def run_command(
        self,
        *,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]: ...


def _format_result(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"shell failed: {result['error']}"
    payload = {
        "exit_code": result.get("exit_code", result.get("exitCode", 0)),
        "stdout": str(result.get("stdout") or ""),
        "stderr": str(result.get("stderr") or ""),
    }
    if result.get("cwd"):
        payload["cwd"] = result["cwd"]
    return json.dumps(payload, ensure_ascii=False)


def make_shell_tool(provider: ShellSandboxProvider, *, default_timeout: int = 30) -> Tool:
    async def fn(
        command: str,
        cwd: Optional[str] = None,
        timeout: int = default_timeout,
    ) -> str:
        try:
            result = await provider.run_command(
                command=command,
                cwd=cwd,
                timeout=timeout,
            )
            return _format_result(result)
        except Exception as ex:
            return f"shell failed: {ex}"

    return Tool(
        name="shell",
        description=(
            "Execute a shell command in the isolated cloud sandbox and return "
            "stdout, stderr, and exit_code. Use it for system inspection, file "
            "operations, running CLIs, git, or anything that is naturally a shell "
            "command rather than a Python/JavaScript program. Prefer code_interpreter "
            "for data analysis or multi-line code. The REST gateway enforces a "
            "hard 30s execution cap. Mounted volumes are accessible: /mnt/user is "
            "writable, /mnt/system and /mnt/skills are read-only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute inside the sandbox.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory inside the sandbox.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (hard 30s cap on the REST gateway).",
                },
            },
            "required": ["command"],
        },
        fn=fn,
    )
