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

            # ReAct loop: continue calling tools until completion or max_steps
            while react_step <= self.max_steps:
                logger.info(f"ReAct step {react_step} / {self.max_steps}")
                react_step += 1

                tool_calls = []
                step_content = ""

                # Compress messages to fit within token budget
                messages = self.msg_manager.fit_to_budget(messages)

                # Call LLM with current messages and available tools
                async for chunk in await self.llm.astream(
                    messages=messages,
                    tools=tools_to_use,
                ):
                    if isinstance(chunk, ErrorChunk):
                        logger.error(f"LLM call failed: {chunk.error_message}")
                        yield chunk
                        return

                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls

                    if isinstance(chunk, ReasoningChunk):
                        yield chunk
                    else:
                        step_content += chunk.delta
                        yield TextChunk(delta=chunk.delta, usage=chunk.usage)

                # If LLM generated text response, add it to messages
                if step_content:
                    messages.append({
                        "role": "assistant",
                        "content": step_content,
                    })
                    step_content = ""

                # If no tool calls, agent has finished
                if not tool_calls:
                    logger.info("No tool calls. ReAct loop complete.")
                    break

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
                            "content": None,
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
                for tool_call, tool_content, tool_error, message_content in tool_results:
                    # Add assistant message with tool call
                    messages.append({
                        "role": "assistant",
                        "content": None,
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
