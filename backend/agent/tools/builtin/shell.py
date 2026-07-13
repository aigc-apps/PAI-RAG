from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Protocol

from agent.tools.base import Tool
from agent.tools.builtin._aliyun_notice import maybe_emit_aliyun_notice


# The agent's own tools are function calls, not shell programs. Once the sandbox
# `shell` is in the toolbox the model sometimes conflates the two and types a tool
# name as a command (e.g. `load_skill skill.foo`) — the skill system's dual nature
# (load_skill is a tool, but skills also bundle scripts you run in the sandbox) makes
# this an easy slip. These names are unmistakably ours, not real binaries, so we
# intercept them before spending a sandbox round-trip and redirect the model to call
# the tool directly. A real binary with the same name is still runnable by an
# explicit path (./x, /usr/bin/x), which doesn't start with a bare identifier.
_TOOL_NOT_SHELL = frozenset({
    "load_skill", "read_skill_resource",
    "knowledge_search", "knowledge_read", "knowledge_find", "knowledge_list",
    "publish_artifact", "code_interpreter",
    "web_search", "web_fetch", "current_datetime",
})
_LEADING_IDENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")


def _leading_tool_name(command: str) -> Optional[str]:
    """The command's leading bare identifier if it names one of the agent's tools."""
    m = _LEADING_IDENT_RE.match((command or "").strip())
    return m.group(1) if m and m.group(1) in _TOOL_NOT_SHELL else None


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


def _format_result(result: Dict[str, Any], hint: Optional[str] = None) -> str:
    if "error" in result:
        return f"shell failed: {result['error']}"
    payload = {
        "exit_code": result.get("exit_code", result.get("exitCode", 0)),
        "stdout": str(result.get("stdout") or ""),
        "stderr": str(result.get("stderr") or ""),
    }
    if result.get("cwd"):
        payload["cwd"] = result["cwd"]
    if hint:
        # Reactive guidance (e.g. aliyun authorization surfaced to the user).
        payload["hint"] = hint
    return json.dumps(payload, ensure_ascii=False)


def make_shell_tool(provider: ShellSandboxProvider, *, default_timeout: int = 30) -> Tool:
    async def fn(
        command: str,
        cwd: Optional[str] = None,
        timeout: int = default_timeout,
    ) -> str:
        tool = _leading_tool_name(command)
        if tool is not None:
            return (
                f"'{tool}' is one of your own tools, not a shell program — call it "
                "directly as a tool/function call. The shell is only for "
                "operating-system and CLI commands (ls, cat, git, python, …). "
                "Nothing was executed."
            )
        try:
            result = await provider.run_command(
                command=command,
                cwd=cwd,
                timeout=timeout,
            )
            hint = maybe_emit_aliyun_notice(command, result)
            return _format_result(result, hint)
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
