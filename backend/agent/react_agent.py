import asyncio
import traceback
from typing import Dict, Optional, Tuple
from common.llm.models import ErrorChunk, ReasoningChunk, ToolResultChunk, TextChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from utils.time_utils import get_current_time_str
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from loguru import logger
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
from utils.constants import try_get_int_env
from agent.state import AgentState
from llama_index.core.tools.function_tool import FunctionTool, ToolOutput
from common.llm.llm_model import PaiLlm, ChatResponseGenerator
from extensions.trace.base import use_current_span
from opentelemetry import trace
from utils.json_utils import parse_tool_arguments
from agent.tool_utils import check_and_handle_return_direct
from agent.message_manager import AgentMessageManager

MAX_RECURSION_STEPS = try_get_int_env("MAX_RECURSION_STEPS", 20) # 最大循环步数
# 流式调用的"空闲超时":超过该秒数没有收到任何分片(package)即超时(非总时长)
LLM_STREAM_IDLE_TIMEOUT = try_get_int_env("LLM_STREAM_IDLE_TIMEOUT_SECONDS", 30)


async def _iter_with_idle_timeout(stream, timeout: int):
    """Yield chunks from a streaming response, raising ``asyncio.TimeoutError`` if
    no chunk arrives within ``timeout`` seconds. This is an idle/inter-chunk
    timeout (resets on every chunk), NOT a total-duration budget."""
    iterator = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield chunk


# 当模型只"预告"下一步动作却没产生 tool_call 时,最多纠正(轻推)几次
MAX_INTENT_NUDGES = try_get_int_env("MAX_INTENT_NUDGES", 1)
_INTENT_NUDGE_MSG = (
    "你刚才只描述了下一步,但没有真正执行。请在本次响应中**直接调用合适的工具**,"
    "或者直接给出最终答案。不要再预告或描述将要调用的工具。"
)
# 行动预告的常见措辞(中英),用于识别"只说不做"的悬空消息。
# 刻意只保留"明显要去用工具"的短语,避免误伤正常答案(去掉了"接下来/下一步/我将"等宽泛词)。
_INTENT_PHRASES = (
    "让我搜索", "让我查", "让我继续", "让我再", "让我先",
    "我需要查找", "我需要搜索", "我来搜", "我来查", "继续搜索",
    "let me search", "let me look", "let me find",
    "i'll search", "i will search", "i need to search", "i need to find", "i'll look",
)


# 一个"行动预告"必然是短消息;超过这个长度就当作正常正文,不再扣留/纠正。
_INTENT_MAX_LEN = 200


def _looks_like_unfinished_intent(text: str) -> bool:
    """A short assistant message that only announces a next action (no tool call)."""
    if not text:
        return False
    t = text.strip().lower()
    return len(text) < _INTENT_MAX_LEN and any(p in t for p in _INTENT_PHRASES)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    from extensions.trace.pai_agent_wrapper import instrument_async_call
    return await instrument_async_call(async_fn, fn_args)


async def execute_single_tool_call(
    tool_call: ChoiceDeltaToolCall,
    tool_fn_map: Dict[str, FunctionTool],
) -> Tuple[ChoiceDeltaToolCall, Optional[str], Optional[str], str]:
    """Execute a single tool call and return results.

    Returns:
        Tuple of (tool_call, tool_content, tool_error, message_content)
    """
    function_name = tool_call.function.name

    if not function_name or function_name not in tool_fn_map:
        logger.warning(f"Unknown tool: {function_name}, skipping.")
        return (tool_call, None, f"Unknown tool: {function_name}", f"Unknown tool: {function_name}")

    # Parse tool arguments
    function_args = parse_tool_arguments(
        tool_call.function.arguments,
    )

    # Execute tool with retry
    async_fn = tool_fn_map[function_name]
    logger.info(f"Calling tool {function_name} with args: {function_args}")

    try:
        tool_result = await call_tool_with_retry(async_fn, function_args)
        tool_content = tool_result.content
        tool_error = None
        message_content = tool_content
    except RetryError as retry_err:
        logger.error(f"Tool call failed after retries: {traceback.format_exc()}")
        inner_exception = retry_err.last_attempt.exception()
        tool_content = None
        tool_error = f"Tool call failed: {inner_exception}"
        message_content = tool_error
    except Exception as ex:
        logger.error(f"Tool call failed: {traceback.format_exc()}")
        tool_content = None
        tool_error = f"Tool call failed: {ex}"
        message_content = tool_error

    return (tool_call, tool_content, tool_error, message_content)


class ReactAgent:
    """A simplified ReAct agent that manages its own message state and tool execution loop."""

    def __init__(
        self,
        llm: PaiLlm,
        system_prompt: str,
        tools: list[FunctionTool],
        max_steps: int = MAX_RECURSION_STEPS,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_steps = max_steps

        self.tools = tools
        self.tool_fn_map = {tool.metadata.name: tool for tool in self.tools}
        self.tool_metadata = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in self.tools
        ]
        self.msg_manager = AgentMessageManager(
            context_window=llm.context_window,
            max_output_tokens=llm.max_tokens,
        )


    @pai_agent_wrapper
    async def run_async(self, state: AgentState) -> ChatResponseGenerator:
        """Execute the ReAct loop with tool calling until completion or max_steps."""
        logger.info("Starting ReAct agent run.")

        @use_current_span(trace.get_current_span())
        async def gen():
            # Initialize messages with system prompt and conversation history
            messages = state.messages.copy()
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    time_prefix = f"[System Time: {get_current_time_str()}]\n"
                    content = messages[i].get("content", "")
                    if isinstance(content, str):
                        messages[i]["content"] = time_prefix + content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                block["text"] = time_prefix + (block.get("text") or "")
                                break
                        else:
                            content.insert(0, {"type": "text", "text": time_prefix})
                    break
            messages = [{"role": "system", "content": self.system_prompt}] + messages

            # Build tool metadata, adding plan tool if enable_agent is True
            tools_to_use = self.tool_metadata.copy()
            react_step = 1
            nudge_count = 0  # times we've corrected an "announce-but-don't-act" turn

            # ReAct loop: continue calling tools until completion or max_steps
            while react_step <= self.max_steps:
                logger.info(f"ReAct step {react_step} / {self.max_steps}")
                react_step += 1

                tool_calls = []
                step_content = ""
                # If we can still nudge this turn, withhold streamed text until we
                # know it isn't a bare action-preview — otherwise the user sees the
                # dangling "let me search…" before we silently correct it. We only
                # hold back the short intent window; past _INTENT_MAX_LEN (or once a
                # tool_call appears) the text can't be a preview, so flush and stream
                # live. `pending` is the held-back, not-yet-yielded text.
                buffering = nudge_count < MAX_INTENT_NUDGES
                pending = ""
                # Usage normally rides only the final chunk, so while we withhold
                # text we keep the latest usage seen and re-attach it on flush —
                # otherwise a buffered short answer would drop its token counts.
                pending_usage = None

                # Compress messages to fit within token budget
                messages = self.msg_manager.fit_to_budget(messages)

                # Call LLM with current messages and available tools.
                # Wrap the stream with an idle timeout: if no package arrives within
                # LLM_STREAM_IDLE_TIMEOUT seconds, abort with a clear error.
                _llm_stream = await self.llm.astream(
                    messages=messages,
                    tools=tools_to_use,
                )
                try:
                    async for chunk in _iter_with_idle_timeout(_llm_stream, LLM_STREAM_IDLE_TIMEOUT):
                        if isinstance(chunk, ErrorChunk):
                            logger.error(f"LLM call failed: {chunk.error_message}")
                            yield chunk
                            return

                        if chunk.tool_calls:
                            tool_calls = chunk.tool_calls
                            # A tool call means any text so far is legit pre-call
                            # narration, not a dangling preview — release it.
                            if buffering:
                                if pending:
                                    yield TextChunk(delta=pending, usage=chunk.usage or pending_usage)
                                    pending = ""
                                    pending_usage = None
                                buffering = False

                        if isinstance(chunk, ReasoningChunk):
                            yield chunk
                        elif buffering:
                            step_content += chunk.delta
                            pending += chunk.delta
                            if chunk.usage:
                                pending_usage = chunk.usage
                            # Past the intent window it can't be a short preview:
                            # flush what we held and stream the rest live.
                            if len(step_content) >= _INTENT_MAX_LEN:
                                yield TextChunk(delta=pending, usage=chunk.usage or pending_usage)
                                pending = ""
                                pending_usage = None
                                buffering = False
                        else:
                            step_content += chunk.delta
                            yield TextChunk(delta=chunk.delta, usage=chunk.usage)
                except asyncio.TimeoutError:
                    logger.error(
                        f"LLM stream idle for >{LLM_STREAM_IDLE_TIMEOUT}s (no package received); aborting."
                    )
                    yield ErrorChunk(
                        error_message=f"模型调用超时：{LLM_STREAM_IDLE_TIMEOUT}s 内未收到任何响应分片。",
                        error_type="llm_stream_timeout",
                    )
                    return

                # No tool calls: either a genuine final answer, OR the model only
                # narrated an intended next action (finish_reason=stop, no tool_call).
                # In the latter case, nudge once to act instead of ending the run.
                if not tool_calls:
                    if (
                        step_content
                        and nudge_count < MAX_INTENT_NUDGES
                        and _looks_like_unfinished_intent(step_content)
                    ):
                        # Dangling preview: it was withheld (still in `pending`), so
                        # the user never saw it — drop the TEXT and nudge to actually
                        # act, but still surface the usage so token accounting is kept.
                        nudge_count += 1
                        if pending_usage:
                            yield TextChunk(delta="", usage=pending_usage)
                        messages.append({"role": "assistant", "content": step_content})
                        messages.append({"role": "user", "content": _INTENT_NUDGE_MSG})
                        logger.info(
                            f"Action-preview without tool call; nudging ({nudge_count}/{MAX_INTENT_NUDGES})."
                        )
                        step_content = ""
                        pending = ""
                        pending_usage = None
                        continue
                    # Real final answer (or nudges exhausted): release any held text
                    # (with its usage), then persist + finish.
                    if pending:
                        yield TextChunk(delta=pending, usage=pending_usage)
                        pending = ""
                        pending_usage = None
                    if step_content:
                        messages.append({"role": "assistant", "content": step_content})
                        step_content = ""
                    logger.info("No tool calls. ReAct loop complete.")
                    break

                # Tool calls present: carry any pre-call narration ON the tool-call
                # assistant message (below), NOT as a separate assistant turn — the
                # "narration then tool_call" history pattern teaches the model to
                # emit bare previews.
                narration_content = step_content or None
                step_content = ""

                # Filter valid tool calls
                valid_tool_calls = [
                    tc for tc in tool_calls
                    if tc.type == "function" and tc.function.name in self.tool_fn_map
                ]

                if not valid_tool_calls:
                    # Log invalid tool calls for debugging
                    logger.warning(f"No valid function tool calls. Invalid tools: {tool_calls}")

                    # Add assistant message with invalid tool calls to maintain conversation state
                    if tool_calls:
                        invalid_tc = tool_calls[0]
                        messages.append({
                            "role": "assistant",
                            "content": narration_content,
                            "tool_calls": [invalid_tc]
                        })

                        # Add error messages for invalid tool calls
                        error_msg = f"Error: Tool '{invalid_tc.function.name}' is not available. Available tools: {list(self.tool_fn_map.keys())}"
                        messages.append({
                            "role": "tool",
                            "content": error_msg,
                            "tool_call_id": invalid_tc.id
                        })

                        # Continue the loop to let LLM correct itself
                        logger.info("Continuing ReAct loop to allow LLM to correct invalid tool calls.")
                        continue
                    else:
                        # No tool calls at all but also no content - this shouldn't happen
                        logger.info("No valid function tool calls and no content. ReAct loop complete.")
                        break

                # Yield tool calls before execution
                for tool_call in valid_tool_calls:
                    yield TextChunk(tool_calls=[tool_call])

                # Execute all tool calls in parallel
                logger.info(f"Executing {len(valid_tool_calls)} tool calls in parallel")
                tool_execution_tasks = [
                    execute_single_tool_call(tc, self.tool_fn_map)
                    for tc in valid_tool_calls
                ]
                tool_results = await asyncio.gather(*tool_execution_tasks)

                # Process results and check for return_direct
                should_return = False
                for idx, (tool_call, tool_content, tool_error, message_content) in enumerate(tool_results):
                    # Add assistant message with tool call; attach any narration to
                    # the first one so it stays bound to an actual tool call.
                    messages.append({
                        "role": "assistant",
                        "content": narration_content if idx == 0 else None,
                        "tool_calls": [tool_call]
                    })

                    # Add tool result message (cap large results)
                    capped_content = self.msg_manager.cap_tool_result(message_content) if message_content else message_content
                    messages.append({
                        "role": "tool",
                        "content": capped_content,
                        "tool_call_id": tool_call.id
                    })

                    # Yield tool result chunk
                    yield ToolResultChunk(
                        tool=tool_call,
                        result=tool_content,
                        error=tool_error
                    )

                    # Check if tool has return_direct=True
                    function_name = tool_call.function.name
                    if function_name in self.tool_fn_map:
                        tool_obj = self.tool_fn_map[function_name]
                        return_chunk = check_and_handle_return_direct(
                            tool_obj=tool_obj,
                            tool_name=function_name,
                            tool_content=tool_content,
                            tool_error=tool_error,
                        )
                        if return_chunk:
                            yield return_chunk
                            should_return = True
                            break

                if should_return:
                    return

            # Max steps exceeded warning
            if react_step > self.max_steps:
                warning_msg = f"Reached max steps: {self.max_steps}. Stopping."
                logger.warning(warning_msg)
                yield TextChunk(delta=f"\n\nReached maximum iteration count ({self.max_steps}), task ended.")

        return gen()
