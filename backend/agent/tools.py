from __future__ import annotations
import traceback
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core.tools.function_tool import FunctionTool
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
from loguru import logger
from agent.message import Message, ToolCall
from utils.json_utils import parse_tool_arguments


@dataclass
class ToolResult:
    message: Message              # the "tool" role message to append to history
    content: Optional[str]        # raw tool output (None on error)
    error: Optional[str]          # error text (None on success)
    name: str
    tool_call: ToolCall

    @property
    def ok(self) -> bool:
        return self.error is None


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def _call_with_retry(tool: FunctionTool, fn_args: dict):
    """Wrap a tool call with tracing and retry.

    NOTE: ``instrument_async_call`` expects the tool object (not a bare
    coroutine function) because it accesses ``async_fn.metadata.name`` and
    calls ``async_fn.acall(**fn_args)``.  This matches how react_agent.py's
    ``call_tool_with_retry`` uses it: the ``tool_fn_map`` stores the full
    ``FunctionTool`` object and that object is passed directly.
    """
    from extensions.trace.pai_agent_wrapper import instrument_async_call
    return await instrument_async_call(tool, fn_args)


class ToolBox:
    def __init__(self, tools: List[FunctionTool]):
        self.tools = tools
        self._by_name = {t.metadata.name: t for t in tools}

    def __bool__(self) -> bool:
        return bool(self.tools)

    def get(self, name: str) -> Optional[FunctionTool]:
        return self._by_name.get(name)

    def is_return_direct(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return bool(tool and getattr(tool.metadata, "return_direct", False))

    def openai_schema(self) -> List[dict]:
        return [t.metadata.to_openai_tool(skip_length_check=True) for t in self.tools]

    async def dispatch(self, tc: ToolCall) -> ToolResult:
        tool = self._by_name.get(tc.name)
        if tool is None:
            err = f"Unknown tool: {tc.name}. Available: {list(self._by_name)}"
            logger.warning(err)
            return ToolResult(
                message=Message("tool", content=err, tool_call_id=tc.id),
                content=None, error=err, name=tc.name, tool_call=tc,
            )
        args = parse_tool_arguments(tc.arguments)
        logger.info(f"Calling tool {tc.name} with args: {args}")
        try:
            out = await _call_with_retry(tool, args)
            content = out.content
            return ToolResult(
                message=Message("tool", content=content, tool_call_id=tc.id),
                content=content, error=None, name=tc.name, tool_call=tc,
            )
        except RetryError as re:
            logger.error(f"Tool call failed after retries: {traceback.format_exc()}")
            err = f"Tool call failed: {re.last_attempt.exception()}"
        except Exception as ex:
            logger.error(f"Tool call failed: {traceback.format_exc()}")
            err = f"Tool call failed: {ex}"
        return ToolResult(
            message=Message("tool", content=err, tool_call_id=tc.id),
            content=None, error=err, name=tc.name, tool_call=tc,
        )
