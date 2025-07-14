import json
import traceback
from typing import Dict, List, cast
from loguru import logger
from pairag.mcp.models import ChatAgentRequest
from pairag.mcp.prompts import (
    PROMPT_WITH_DEEP_RESEARCH,
    PROMPT_WITHOUT_DEEP_RESEARCH,
    PROMPT_WITHOUT_TOOLS,
)
from pairag.mcp.trace.pai_agent_wrapper import pai_agent_wrapper
from pairag.mcp.utils.message_utils import convert_to_chat_messages
from pairag.mcp.utils.time_utils import get_prompt_current_time_str
from pairag.mcp.tools.think.think_and_planning_tool import aget_simple_think_tool
from pairag.mcp.providers.mcp_tool_provider import mcp_provider
from pairag.mcp.providers.llm_provider import llm_provider
from pairag.mcp.providers.websearch_provider import websearch_provider
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from llama_index.core.llms import LLM
from pairag.mcp.constants import MAX_CHAT_STEPS
from pairag.memory.base_memory import BaseMemory
from llama_index.core.tools import FunctionTool, ToolOutput
from tenacity import retry, stop_after_attempt, wait_fixed
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ChatResponseAsyncGen,
    ChatResponse,
)
from pairag.integrations.trace.base import use_current_span
from opentelemetry import trace


def get_system_prompt(enable_search, enable_mcp, enable_thinking):
    if enable_search or enable_mcp:
        if enable_thinking:
            system_prompt = PROMPT_WITH_DEEP_RESEARCH.format(
                current_datetime=get_prompt_current_time_str()
            )
        else:
            system_prompt = PROMPT_WITHOUT_DEEP_RESEARCH.format(
                current_datetime=get_prompt_current_time_str()
            )
    else:
        system_prompt = PROMPT_WITHOUT_TOOLS.format(
            current_datetime=get_prompt_current_time_str()
        )
    return system_prompt


async def aget_mcp_tools(chat_request: ChatAgentRequest) -> List[FunctionTool]:
    mcp_tools = []

    # 获取思考工具
    think_cache = []
    think_tool = await aget_simple_think_tool(think_cache=think_cache)
    mcp_tools.append(think_tool)
    if chat_request.enable_search:
        websearch_tools = websearch_provider.get_search_tools()
        mcp_tools.extend(websearch_tools)
    if chat_request.enable_mcp:
        logger.info(f"[Model] selected mcp servers: {chat_request.mcp_servers}")
        mcp_tools.extend(mcp_provider.get_mcp_tools(chat_request.mcp_servers))
        logger.info(f"[Model] mcp_tools: {mcp_tools}")

    return mcp_tools


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    return await async_fn.acall(**fn_args)


async def astep_gen(
    llm: LLM,
    tools: List[FunctionTool],
    tool_name_map: Dict[str, FunctionTool],
    memory: BaseMemory = None,
):
    messages = memory.get_context()
    if tools:
        response_gen: ChatResponseAsyncGen = await llm.astream_chat(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream_options={"include_usage": True},
        )
    else:
        response_gen: ChatResponseAsyncGen = await llm.astream_chat(
            messages=messages,
            stream_options={"include_usage": True},
        )

    tool_calls = []
    response_context = ""
    async for response in response_gen:
        tool_calls = response.message.additional_kwargs.get("tool_calls")
        if response.delta:
            response.message.additional_kwargs.pop("tool_calls", None)
            response_context += response.delta
            yield response

    if response_context:
        memory.add(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_context,
            )
        )
    if tool_calls:
        tool_calls = cast(List[ChoiceDeltaToolCall], tool_calls)
        for tool_call in tool_calls:
            async_tool_fn = tool_name_map[tool_call.function.name]
            if tool_call.function.arguments:
                fn_args = json.loads(tool_call.function.arguments)
            else:
                fn_args = {}
            tool_result = await call_tool_with_retry(async_tool_fn, fn_args)

            tool_call_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={"tool_calls": [tool_call]},
            )
            tool_result_message = ChatMessage(
                role=MessageRole.TOOL,
                content=tool_result.content,
                additional_kwargs={
                    "tool_call_id": tool_call.id,
                },
            )
            memory.add(tool_call_message)
            memory.add(tool_result_message)

            yield ChatResponse(
                message=tool_call_message,
                delta="",
            )
            yield ChatResponse(
                message=tool_result_message,
                delta=tool_result.content,
            )
    else:
        yield ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={"STOP_FLAG": True},
            ),
            delta="",
        )


class AgentLoop:
    def __init__(self, max_steps: int = MAX_CHAT_STEPS):
        self.max_steps = max_steps

    @pai_agent_wrapper
    async def arun(self, chat_request: ChatAgentRequest) -> ChatResponseAsyncGen:
        mcp_tools = await aget_mcp_tools(chat_request)
        tools = []
        tool_name_map = {}

        for tool in mcp_tools:
            tools.append(
                {"type": "function", "function": tool.metadata.to_openai_function()}
            )
            tool_name_map[tool.metadata.name] = tool

        llm: LLM = llm_provider.get_llm_model(model_id=chat_request.model)

        system_prompt = get_system_prompt(
            enable_search=chat_request.enable_search,
            enable_mcp=chat_request.enable_mcp,
            enable_thinking=chat_request.enable_thinking,
        )

        input_messages = [
            {"role": "system", "content": system_prompt}
        ] + chat_request.messages
        messages = convert_to_chat_messages(input_messages)
        memory = BaseMemory()
        memory.from_messages(messages)

        max_steps = chat_request.max_steps or self.max_steps

        @use_current_span(trace.get_current_span())
        async def gen():
            cur_step = 0
            stop_flag = False

            while cur_step <= max_steps:
                cur_step += 1
                logger.info(f"Running step {cur_step}/{max_steps}.")
                try:
                    step_gen = astep_gen(
                        llm=llm,
                        tools=tools,
                        tool_name_map=tool_name_map,
                        memory=memory,
                    )
                    async for chunk in step_gen:
                        chunk.message.additional_kwargs["step"] = cur_step
                        yield chunk
                        if chunk.message.additional_kwargs.get("STOP_FLAG"):
                            stop_flag = True
                            break
                    if stop_flag:
                        logger.info("Reached stop flag, ending agent loop.")
                        break

                except (ValueError, TypeError, KeyError):
                    # 情况1: 参数错误
                    logger.exception("ValueError: 工具调用参数异常")
                    logger.error(traceback.format_exc())
                    continue

                except Exception as e:
                    # 情况2: 其他错误
                    logger.exception("UnhandledError: 工具调用失败")
                    if hasattr(e, "last_attempt") and hasattr(
                        e.last_attempt, "_exception"
                    ):
                        error_detail = e.last_attempt._exception
                        logger.error(f"Retry failed: {error_detail}")
                    logger.error(traceback.format_exc())

                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=f"工具调用失败，请检查你的工具配置是否正确。\n{e}",
                        ),
                        delta=f"工具调用失败，请检查你的工具配置是否正确。\n{e}",
                        additional_kwargs={"failed": True, "step": cur_step},
                    )
                    break

        return gen()
