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
from opentelemetry import trace


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    from extensions.trace.pai_agent_wrapper import instrument_async_call
    return await instrument_async_call(async_fn, fn_args)


MAX_RECURSION_STEPS = 20


class ActorWithPlan(BaseAgent):
    def __init__(
        self,
        prompt: str,
        llm: PaiLlm,
        tools: list[FunctionTool],
        name: str,
        max_steps: int = MAX_RECURSION_STEPS
    ):
        super().__init__(prompt, llm, tools, name)

        self.max_steps = max_steps
        self.tool_fn_map = {tool.metadata.name: tool for tool in self.tools}
        self.tool_metadata = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in self.tools
        ]

    def build_prompt(self, state: AgentState) -> str:
        plan_list = ""
        if len(state.plan["steps"]) > 0:
            for i, task in enumerate(state.plan["steps"]):
                plan_list += f"Step {i+1}. {task}\n"

        return self.prompt.format(
            task_results=state.observations,
            plan_list=plan_list,
            step=state.step,
            context_variables=state.format_context_str(),
            task_name=state.plan['steps'][state.step-1],
            **state.context_variables,
        )



    @pai_agent_wrapper
    async def run_async(self, state: AgentState) -> ChatResponseGenerator:
        logger.info("Running actor_with_plan agent.")
        act_with_plan_prompt = self.build_prompt(state)
        messages = [{"role": "user", "content": act_with_plan_prompt}]

        @use_current_span(trace.get_current_span())
        async def gen():
            action_step = 1
            while action_step <= self.max_steps:
                logger.info(f"Acting at step {action_step} with messages: {messages}")
                action_step += 1

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
                        yield TextChunk(
                            delta=chunk.delta
                        )

                if step_content:
                    messages.append({
                        "role": "assistant",
                        "content": step_content,
                    })
                    step_content = ""

                if tool_calls:
                    for tool in tool_calls:
                        if tool.type == "function":
                            function_name = tool.function.name
                            if not function_name or function_name not in self.tool_fn_map:
                                logger.warning(f"Unknown tool_call: {tool}, skip it.")
                                continue


                            if function_name == "respond-tool":
                                logger.info("Actor finished with respond-tool.")
                                yield TextChunk(tool_calls=[tool])
                                return


                            if tool.function.arguments:
                                function_args = json.loads(tool.function.arguments)
                            else:
                                function_args = {}

                            yield TextChunk(
                                tool_calls=[tool],
                            )
                            async_fn = self.tool_fn_map[function_name]
                            logger.info(f"Calling tool {function_name} with args {function_args}.")
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
                else:
                    break

            if action_step > self.max_steps:
                yield TextChunk(delta="任务失败: 超出最大迭代次数，任务已结束。")


        return gen()
