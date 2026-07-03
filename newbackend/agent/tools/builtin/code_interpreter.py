from __future__ import annotations

import json
from typing import Any, Dict, Optional, Protocol

from agent.tools.base import Tool


class SandboxProvider(Protocol):
    async def run_code(
        self,
        *,
        code: str,
        language: str,
        timeout: int,
        context_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]: ...


def _format_result(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"code_interpreter failed: {result['error']}"
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    exit_code = result.get("exit_code", result.get("exitCode", 0))
    payload = {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }
    if result.get("artifacts"):
        payload["artifacts"] = result["artifacts"]
    return json.dumps(payload, ensure_ascii=False)


def make_code_interpreter_tool(provider: SandboxProvider, *, default_timeout: int = 60) -> Tool:
    async def fn(
        code: str,
        language: str = "python",
        timeout: int = default_timeout,
        context_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> str:
        try:
            result = await provider.run_code(
                code=code,
                language=language,
                timeout=timeout,
                context_id=context_id,
                cwd=cwd,
            )
            return _format_result(result)
        except Exception as ex:
            return f"code_interpreter failed: {ex}"

    return Tool(
        name="code_interpreter",
        description=(
            "Execute Python or JavaScript code in an isolated cloud sandbox. "
            "Use it for data analysis, file processing, calculations, and code "
            "experiments. User files and custom skills may be mounted into the "
            "sandbox by the configured provider."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to execute inside the sandbox.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript"],
                    "description": "Execution language. Defaults to python.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds.",
                },
                "context_id": {
                    "type": "string",
                    "description": "Optional stateful execution context id.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory inside the sandbox.",
                },
            },
            "required": ["code"],
        },
        fn=fn,
    )
