import json
from typing import Any, Dict
from backend.chat.agent.actor_with_plan import ActorWithPlan
from chat.agent.prompts import ACT_PROMPT, PLAN_PROMPT, SUMMARY_PROMPT
from chat.agent.summarizer import Summarizer
from backend.chat.agent.actor import Actor
from chat.llm.models import ChunkStage, ToolResultChunk
from chat.llm.utils import parse_llm_json
from chat.tools.plan_tool import aget_plan_tool, aget_respond_tool
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
from utils.constants import try_get_int_env
from chat.agent.base import BaseAgent
from chat.agent.state import AgentState
from llama_index.core.tools.function_tool import FunctionTool, ToolOutput
from chat.llm.llm_model import PaiLlm, TextChunk, ReasoningChunk, ChatResponseGenerator
from extensions.trace.base import use_current_span
from opentelemetry import trace


MAX_RECURSION_STEPS = try_get_int_env("MAX_RECURSION_STEPS", 20) # 最大循环步数


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    return await async_fn.acall(**fn_args)



class PlanAgentPromptSet(BaseModel):
    plan_prompt: str = PLAN_PROMPT
    act_prompt: str = ACT_PROMPT
    summary_prompt: str = SUMMARY_PROMPT


class Planner(BaseAgent):
    def __init__(
        self,
        llm: PaiLlm,
        tools: list[FunctionTool],
        name: str,
        prompt_set: PlanAgentPromptSet,
        max_steps: int = MAX_RECURSION_STEPS,
    ):
        super().__init__(prompt_set.plan_prompt, llm, tools, name)

        self.prompt_set = prompt_set
        self.max_steps = max_steps
        self.tool_fn_map = {tool.metadata.name: tool for tool in self.tools}
        self.tool_metadata = [
            tool.metadata.to_openai_tool() for tool in self.tools
        ]

    # internal tool for planning agent
    async def get_plan_tool_meta(self) -> Dict[str, Any]:
        plan_tool = await aget_plan_tool()
        return plan_tool.metadata.to_openai_tool()

    @pai_agent_wrapper
    async def run_async(self, state: AgentState) -> ChatResponseGenerator:
        logger.info("Start agentic run.")
        plan_prompt = self.prompt.format(context_variables=state.format_context_str())
        tools_to_plan = self.tool_metadata
        if tools_to_plan and state.enable_agent:
            tools_to_plan.append(await self.get_plan_tool_meta())

        @use_current_span(trace.get_current_span())
        async def gen():
            selected_tool = None
            plan_delta = ""
            messages = [{"role": "system", "content": plan_prompt}] + state.messages

            async for chunk in await self.invoke_llm_async(
                messages=messages,
                tools=tools_to_plan,
            ):
                if chunk.tool_calls:
                    selected_tool = chunk.tool_calls[0]

                plan_delta += chunk.delta or ""
                if chunk.delta and selected_tool:
                    yield ReasoningChunk(
                        reasoning_delta=chunk.delta,
                        tool_calls=chunk.tool_calls,
                        stage=ChunkStage.ACTING,
                    )
                else:
                    chunk.stage = ChunkStage.ACTING
                    yield chunk
                # yield chunk

            # TODO: Fallback for empty plan
            if selected_tool is None:
                if not plan_delta:
                    logger.warning("Planner execute error, no direct response and no tool.")
                    yield TextChunk(delta="抱歉，出现错误，请重试。")
                return


            tool_name = selected_tool.function.name
            if tool_name != "planning-tool":
                logger.info(f"Single tool execution detected, tool: {tool_name}.")

                if not tool_name or tool_name not in self.tool_fn_map:
                    logger.warning(f"Unknown tool_call: {tool_name}, ignore it.")
                else:
                    if selected_tool.function.arguments:
                        function_args = json.loads(selected_tool.function.arguments)
                    else:
                        function_args = {}

                    yield TextChunk(
                        tool_calls=[selected_tool],
                    )
                    async_fn = self.tool_fn_map[tool_name]
                    logger.info(f"Calling tool {tool_name} with args {function_args}.")
                    tool_result = await call_tool_with_retry(async_fn, function_args)
                    logger.info(f"Get tool result {tool_result}.")


                    state.messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [selected_tool]
                        }
                    )
                    state.messages.append(
                        {
                            "role": "tool",
                            "content": tool_result.content,
                            "tool_call_id": selected_tool.id
                        }
                    )
                    state.observations += tool_result.content + "\n\n"

                yield ToolResultChunk(
                    tool=selected_tool,
                    result=tool_result.content,
                )

                actor = Actor(
                    prompt=plan_prompt,
                    llm=self.llm,
                    tools=self.tools,
                    name="actor",
                    max_steps=self.max_steps
                )

                summarizer = Summarizer(
                    self.prompt_set.summary_prompt,
                    llm=self.llm,
                    name="summarizer",
                )
                response_gen = await actor.run_async(state)
                async for chunk in response_gen:
                    if chunk.tool_calls:
                        if chunk.tool_calls[0].function.name == "respond-tool":
                            logger.info("Actor finished with respond-tool.")
                            break
                        elif isinstance(chunk, ToolResultChunk):
                            state.messages.append(
                                {
                                    "role": "tool",
                                    "content": tool_result.content,
                                    "tool_call_id": chunk.tool.id
                                }
                            )
                            state.observations += chunk.result + "\n\n"

                    if chunk.delta:
                        yield ReasoningChunk(
                            reasoning_delta=chunk.delta,
                            tool_calls=chunk.tool_calls,
                            stage=ChunkStage.ACTING,
                        )
                    else:
                        chunk.stage = ChunkStage.ACTING
                        yield chunk


                answer_gen = await summarizer.run_async(state)
                is_first_chunk = True
                async for chunk in answer_gen:
                    chunk.stage = ChunkStage.RESPONSE
                    if is_first_chunk:
                        chunk.delta = "\n" + chunk.delta # Summary 换行
                        is_first_chunk = False

                    yield chunk
            else:
                logger.info(f"Planning tool execution detected, plan: {selected_tool.function.arguments}.")

                yield ToolResultChunk(
                    tool=selected_tool,
                    result=selected_tool.function.arguments,
                )

                try:
                    state.plan = parse_llm_json(selected_tool.function.arguments)
                except Exception as ex:
                    logger.error(f"Parse plan json error: {ex}, plan content: {selected_tool.function.arguments}")
                    raise ex

                if len(state.plan["steps"]) == 0:
                    logger.error("Empty plan steps.")
                    raise Exception("Empty plan steps.")


                actor_with_plan = ActorWithPlan(
                    prompt=self.prompt_set.act_prompt,
                    llm=self.llm,
                    tools=self.tools + [await aget_respond_tool()],
                    name="actor_with_plan",
                    max_steps=10,
                )
                summarizer = Summarizer(
                    self.prompt_set.summary_prompt,
                    llm=self.llm,
                    name="summarizer",
                )

                response_gen = await actor_with_plan.run_async(state)

                async for chunk in response_gen:
                    if chunk.tool_calls:
                        if chunk.tool_calls[0].function.name == "respond-tool":
                            logger.info("Actor finished with respond-tool.")
                            break
                        elif isinstance(chunk, ToolResultChunk):
                            state.messages.append(
                                {
                                    "role": "tool",
                                    "content": tool_result.content,
                                    "tool_call_id": chunk.tool.id
                                }
                            )
                            state.observations += chunk.result + "\n\n"

                    if chunk.delta:
                        yield ReasoningChunk(
                            reasoning_delta=chunk.delta,
                            tool_calls=chunk.tool_calls,
                            stage=ChunkStage.ACTING,
                        )
                    else:
                        chunk.stage = ChunkStage.ACTING
                        yield chunk


                answer_gen = await summarizer.run_async(state)
                is_first_chunk = True
                async for chunk in answer_gen:
                    chunk.stage = ChunkStage.RESPONSE
                    if is_first_chunk:
                        chunk.delta = "\n" + chunk.delta # Summary 换行
                        is_first_chunk = False

                    yield chunk

        return gen()
