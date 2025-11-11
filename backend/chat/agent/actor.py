# chat/agent/actor.py

import json
import traceback
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from loguru import logger
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
from chat.agent.base import BaseAgent
from chat.agent.state import AgentState
from llama_index.core.tools.function_tool import FunctionTool, ToolOutput
from chat.llm.llm_model import PaiLlm
from chat.llm.models import TextChunk, ChatResponseGenerator, ToolResultChunk
from extensions.trace.base import use_current_span
from chat.agent.prompts import SUMMARY_PROMPT
from common.chat.constants import MessageRole
from opentelemetry import trace


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    from extensions.trace.pai_agent_wrapper import instrument_async_call
    return await instrument_async_call(async_fn, fn_args)


MAX_RECURSION_STEPS = 20



class Actor(BaseAgent):
    def __init__(
        self,
        prompt: str,
        llm: PaiLlm,
        tools: list[FunctionTool],
        name: str = "actor",
        max_steps: int = MAX_RECURSION_STEPS
    ):
        super().__init__(prompt, llm, tools, name)
        self.max_steps = max_steps
        self.tool_fn_map = {tool.metadata.name: tool for tool in self.tools}
        self.tool_metadata = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in self.tools
        ]


    def build_prompt(self, state: AgentState) -> str:
        return self.prompt.format(
            context_variables=state.format_context_str(),
        )




    @pai_agent_wrapper
    async def run_async(self, state: AgentState) -> ChatResponseGenerator:
        logger.info("Running actor agent.")
        @use_current_span(trace.get_current_span())
        async def gen():
            react_step = 1
            messages = state.messages.copy()
            observations = ""
            if state.current_tool_call:
                selected_tool = state.current_tool_call
                tool_name = selected_tool.function.name

                if selected_tool.function.arguments:
                    try:
                        function_args = json.loads(selected_tool.function.arguments)
                    except json.JSONDecodeError:
                        logger.error(f"[{self.name}] Invalid JSON args: {selected_tool.function.arguments}")
                        function_args = {}
                else:
                    function_args = {}

                yield TextChunk(
                    tool_calls=[selected_tool],
                )
                async_fn = self.tool_fn_map[tool_name]
                logger.info(f"Calling tool {tool_name} with args {function_args}.")
                try:
                    tool_result = await call_tool_with_retry(async_fn, function_args)
                    tool_content = tool_result.content
                    tool_error = None
                    message_content = tool_content
                except RetryError as retry_err:
                    logger.error(f"Call tool failed: {traceback.format_exc()}")
                    inner_exception = retry_err.last_attempt.exception()
                    tool_content = None
                    tool_error = f"工具调用失败: {inner_exception}"
                    message_content = tool_error
                except Exception as ex:
                    logger.error(f"Call tool failed: {traceback.format_exc()}")
                    tool_content = None
                    tool_error = f"工具调用失败: {ex}"
                    message_content = tool_error

                #logger.info(f"Get tool result {tool_result}.")
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            selected_tool
                        ]
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": message_content,
                        "tool_call_id": selected_tool.id
                    }
                )
                yield ToolResultChunk(
                    tool=selected_tool,
                    result=tool_content,
                    error=tool_error
                )

                state.current_tool_call = None
                observations += message_content + "\n\n"
            act_prompt = self.build_prompt(state)
            messages = [{"role": "system", "content": act_prompt}] + messages

            while react_step <= self.max_steps:
                logger.info(f"[{self.name}] ReAct step {react_step} / {self.max_steps}")
                react_step += 1

                tool_calls = []
                step_content = ""


                async for chunk in await self.invoke_llm_async(
                    messages=messages,
                    tools=self.tool_metadata,
                ):
                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                    if chunk.delta:
                        step_content += chunk.delta
                        yield TextChunk(delta=chunk.delta)

                if step_content:
                    messages.append({
                        "role": "assistant",
                        "content": step_content,
                    })
                    step_content = ""

                if not tool_calls:
                    logger.info(f"[{self.name}] No more tool calls. Exiting ReAct loop.")
                    break

                # 处理工具调用
                for tool in tool_calls:
                    if tool.type != "function":
                        continue

                    function_name = tool.function.name
                    if not function_name or function_name not in self.tool_fn_map:
                        logger.warning(f"[{self.name}] Unknown tool: {function_name}, skipping.")
                        continue

                    try:
                        function_args = json.loads(tool.function.arguments) if tool.function.arguments else {}
                    except json.JSONDecodeError:
                        logger.error(f"[{self.name}] Invalid JSON args: {tool.function.arguments}")
                        function_args = {}

                    yield TextChunk(tool_calls=[tool])

                    # 调用工具
                    async_fn = self.tool_fn_map[function_name]
                    logger.info(f"[{self.name}] Calling {function_name} with args: {function_args}")
                    try:
                        tool_result = await call_tool_with_retry(async_fn, function_args)
                        tool_content = tool_result.content
                        tool_error = None
                        message_content = tool_content
                    except RetryError as retry_err:
                        logger.error(f"Call tool failed: {traceback.format_exc()}")
                        inner_exception = retry_err.last_attempt.exception()
                        tool_content = None
                        tool_error = f"工具调用失败: {inner_exception}"
                        message_content = tool_error
                    except Exception as ex:
                        logger.error(f"Call tool failed: {traceback.format_exc()}")
                        tool_content = None
                        tool_error = f"工具调用失败: {ex}"
                        message_content = tool_error

                    #logger.info(f"Get tool result {tool_result}.")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                tool
                            ]
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "content": message_content,
                            "tool_call_id": tool.id
                        }
                    )
                    yield ToolResultChunk(
                        tool=tool,
                        result=tool_content,
                        error=tool_error
                    )
                    observations += message_content + "\n\n"

            # 超出步数保护
            if react_step > self.max_steps:
                logger.warning(f"Reached max recursion steps: {self.max_steps}")
                current_datetime = state.context_variables.get("current_datetime", "")
                prompt = SUMMARY_PROMPT.format(
                    tool_results=observations,
                    chat_history=state.chat_history,
                    current_datetime=current_datetime,
                    user_query=state.user_query,
                )
                response_gen = await self.invoke_llm_async(messages=[
                {"role": MessageRole.USER, "content": prompt},
            ])
                async for chunk in response_gen:
                    yield chunk
                return

        return gen()
