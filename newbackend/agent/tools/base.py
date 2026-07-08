from __future__ import annotations
import traceback
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Literal, Optional
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
from loguru import logger
from agent.message import Message, ToolCall
from agent.tools.artifacts import (
    begin_artifact_capture,
    begin_notice_capture,
    drain_artifacts,
    drain_tool_notice,
    end_artifact_capture,
    end_notice_capture,
)
from agent.tools.scope import ToolScope, reset_current_tool_scope, set_current_tool_scope
from utils.json_utils import parse_tool_arguments


@dataclass
class Tool:
    """A callable the agent can invoke. `fn` takes the parsed JSON arguments as
    kwargs and returns a string (the tool output)."""

    name: str
    description: str
    parameters: Dict  # JSON Schema for the arguments object
    fn: Callable[..., Awaitable[str]]
    return_direct: bool = False
    permission: Literal["auto", "ask", "admin"] = "auto"

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    message: Message
    content: Optional[str]
    error: Optional[str]
    name: str
    tool_call: ToolCall
    # Structured file artifacts the tool surfaced (e.g. publish_artifact). These
    # ride the event/SSE separately from `content` — the model only sees the
    # short string in `content`, never the bytes.
    files: Optional[List[dict]] = None
    # A single structured UI notice the tool surfaced (e.g. an "aliyun
    # authorization required" card). Stream-only, never persisted.
    notice: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def _call_with_retry(tool: "Tool", args: dict) -> str:
    """Invoke the tool fn inside a tracing span, with retry.
    The tracing span is optional: in lean mode (no trace extension) the call
    runs inside a nullcontext instead."""
    try:
        from extensions.trace.tracer import get_tracer
        span_cm = get_tracer().start_as_current_span(f"tool {tool.name}")
    except Exception:  # lean mode: no trace extension
        from contextlib import nullcontext
        span_cm = nullcontext()

    with span_cm as span:
        if span is not None:
            try:
                span.set_attribute("tool.name", tool.name)
            except Exception:
                pass
        result = await tool.fn(**args)
        return result if isinstance(result, str) else str(result)


class ToolBox:
    def __init__(self, tools: List[Tool]):
        self.tools = tools
        self._by_name = {t.name: t for t in tools}

    def __bool__(self) -> bool:
        return bool(self.tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._by_name.get(name)

    def is_return_direct(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return bool(tool and tool.return_direct)

    def openai_schema(self) -> List[dict]:
        return [t.openai_schema() for t in self.tools]

    async def dispatch(self, tc: ToolCall, scope: Optional[ToolScope] = None) -> ToolResult:
        tool = self._by_name.get(tc.name)
        if tool is None:
            err = f"Unknown tool: {tc.name}. Available: {list(self._by_name)}"
            logger.warning(err)
            return ToolResult(
                message=Message("tool", content=err, tool_call_id=tc.id),
                content=None,
                error=err,
                name=tc.name,
                tool_call=tc,
            )
        token = set_current_tool_scope(scope or ToolScope())
        artifact_token = begin_artifact_capture()
        notice_token = begin_notice_capture()
        try:
            if tool.permission == "admin" and not get_current_scope_admin():
                raise PermissionError(f"{tc.name} requires admin permission")
            args = parse_tool_arguments(tc.arguments)
            logger.info(f"Calling tool {tc.name} with args: {args}")
            content = await _call_with_retry(tool, args)
            files = [a.model_dump() for a in drain_artifacts()] or None
            return ToolResult(
                message=Message("tool", content=content, tool_call_id=tc.id),
                content=content,
                error=None,
                name=tc.name,
                tool_call=tc,
                files=files,
                notice=drain_tool_notice(),
            )
        except RetryError as re:
            logger.error(f"Tool call failed after retries: {traceback.format_exc()}")
            err = f"Tool call failed: {re.last_attempt.exception()}"
        except Exception as ex:
            logger.error(f"Tool call failed: {traceback.format_exc()}")
            err = f"Tool call failed: {ex}"
        finally:
            notice = drain_tool_notice()
            end_notice_capture(notice_token)
            end_artifact_capture(artifact_token)
            reset_current_tool_scope(token)
        return ToolResult(
            message=Message("tool", content=err, tool_call_id=tc.id),
            content=None,
            error=err,
            name=tc.name,
            tool_call=tc,
            notice=notice,
        )


def get_current_scope_admin() -> bool:
    from agent.tools.scope import get_current_tool_scope

    return get_current_tool_scope().is_admin
