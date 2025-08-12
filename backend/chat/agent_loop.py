import json
import traceback
from typing import Dict, List, cast, AsyncGenerator
from loguru import logger
from common.chat.models import ChatAgentRequest
from utils.message_utils import convert_to_chat_messages
from utils.time_utils import get_prompt_current_time_str
from tools.think.think_and_planning_tool import aget_simple_think_tool
from tools.attachments.file_reader import aget_file_reader
from tools.attachments.file_searcher import aget_file_searcher
from config.providers.mcp_tool_provider import mcp_provider
from config.providers.llm_provider import llm_provider
from config.providers.websearch_provider import websearch_provider
from config.providers.prompt_provider import prompt_provider
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from rag.knowledgebase_tool import aget_knowledgebase_tool
from llama_index.core.llms import LLM
from common.chat.constants import MAX_CHAT_STEPS
from memory.base_memory import BaseMemory
from llama_index.core.tools import FunctionTool, ToolOutput
from tenacity import retry, stop_after_attempt, wait_fixed
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ChatResponseAsyncGen,
    ChatResponse,
)
from llama_index.core.bridge.pydantic import Field, BaseModel, ConfigDict
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from extensions.trace.base import use_current_span
from opentelemetry import trace

class AgentState(BaseModel):
    llm: LLM = Field(description="llm")
    step: int = Field(description="step", default=0)
    stop_flag: bool = Field(description="whether stop", default=False)
    memory: BaseMemory = Field(description="memory", default=None)
    tools: List[Dict] = Field(description="tools", default=None)
    tool_name_map: Dict[str, FunctionTool] = Field(description="tool_name_map", default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

def get_system_prompt(enable_search: bool = False, enable_agent: bool = False, enable_attachments: bool = False, kb_ids: List[str] = []):
    tools_prompt = []
    prompt = prompt_provider.get_prompts()
    if enable_agent:
        tools_prompt.append(prompt.prompts["planning_tool_prompt"])
    if enable_search:
        tools_prompt.append(prompt.prompts["search_web_tool_prompt"].format(
        current_datetime=get_prompt_current_time_str()))
    if enable_attachments:
        tools_prompt.append(prompt.prompts["attachments_tool_prompt"])
    if kb_ids:
        tools_prompt.append(prompt.prompts["knowledgebase_tool_prompt"])
    if not tools_prompt:
        tools_prompt.append(prompt.prompts["without_tools_prompt"])
    system_prompt = prompt.prompts["system_prompt"].format(
        tools_prompt="\n\n".join(tools_prompt), current_datetime=get_prompt_current_time_str()
            )
    return system_prompt


async def aget_mcp_tools(chat_request: ChatAgentRequest) -> List[FunctionTool]:
    mcp_tools = []

    if chat_request.enable_attachments:
        # 获取文件搜索工具
        attachments = []
        for message in chat_request.messages:
            if message.get("role") == "user" and len(message.get("attachments", [])) > 0:
                attachments.extend(message.get("attachments", []))
        file_searcher_tool = await aget_file_searcher(attachments=attachments)
        mcp_tools.append(file_searcher_tool)
    # 获取思考工具
    think_cache = []
    think_tool = await aget_simple_think_tool(think_cache=think_cache)
    if chat_request.enable_agent:
        mcp_tools.append(think_tool)
    if chat_request.enable_search:
        websearch_tools = websearch_provider.get_search_tools()
        mcp_tools.extend(websearch_tools)
    if len(chat_request.mcp_ids) > 0:
        logger.info(f"[Model] selected mcp servers: {chat_request.mcp_ids}")
        mcp_tools.extend(await mcp_provider.get_mcp_tools_async(chat_request.mcp_ids))
        logger.info(f"[Model] mcp_tools: {mcp_tools}")

    mcp_tools.extend(await aget_kb_tools(chat_request))
    return mcp_tools


async def aget_kb_tools(chat_request: ChatAgentRequest) -> List[FunctionTool]:
    kb_tools = []

    kb_ids = chat_request.kb_ids or []
    for kb_id in kb_ids:
        kb_tools.append(await aget_knowledgebase_tool(kb_id))

    logger.info(f"Resolved {len(kb_tools)} knowledgebase tools.")
    return kb_tools


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    return await async_fn.acall(**fn_args)

async def synthesize_agent(state: AgentState) -> AsyncGenerator[ChatResponse, None]:
    history_memory_without_sys_prompt =  state.memory.get_context()[1:]
    synthesize_prompt = get_system_prompt()
    new_messages = []
    new_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=synthesize_prompt))
    new_messages.extend(history_memory_without_sys_prompt)
    sythesize_memory = BaseMemory()
    sythesize_memory.from_messages(new_messages)
    state.memory = sythesize_memory

    async for chunk in astep_gen(
            llm=state.llm,
            memory=state.memory,
        ):
        chunk.message.additional_kwargs["step"] = state.step
        yield chunk
        if chunk.message.additional_kwargs.get("STOP_FLAG"):
            state.stop_flag = True
            break

async def step_agent(state: AgentState, attachments: List[dict]) -> AsyncGenerator[ChatResponse, None]:
    async for chunk in astep_gen(
        llm=state.llm,
        tools=state.tools,
        tool_name_map=state.tool_name_map,
        memory=state.memory,
        attachments=attachments
    ):
        chunk.message.additional_kwargs["step"] = state.step
        yield chunk
        if chunk.message.additional_kwargs.get("STOP_FLAG"):
            state.stop_flag = True
            break

async def astep_gen(
    llm: LLM,
    tools: List[FunctionTool]= None,
    tool_name_map: Dict[str, FunctionTool]= None,
    memory: BaseMemory = None,
    attachments: List[dict] = None
):
    image_urls = []
    if attachments and len(attachments) > 0:
        for attachment in attachments:
            file_reader = await aget_file_reader()
            file_reader_fn_args = {
                "file_id": attachment.get("id"),
                "file_name": attachment.get("name", "未知附件"),
            }
            file_reader_tool_call = ChoiceDeltaToolCall(
                index=0,
                id=f"call_file_reader_{attachment.get('id')}",
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=file_reader.metadata.name,
                    arguments=json.dumps(
                        file_reader_fn_args, ensure_ascii=False
                    ),
                ),
            )

            tool_result = await call_tool_with_retry(
                file_reader, file_reader_fn_args
            )
            if str(attachment.get("contentType")).startswith("image/"):
                image_urls.append(json.loads(tool_result.content).get("data", ""))
            else:
                tool_call_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    additional_kwargs={"tool_calls": [file_reader_tool_call]},
                )
                tool_result_message = ChatMessage(
                    role=MessageRole.TOOL,
                    content=tool_result.content,
                    additional_kwargs={
                        "tool_call_id": file_reader_tool_call.id,
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
    messages = memory.get_context(image_urls)
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
        if tool_calls:
            tool_calls = cast(List[ChoiceDeltaToolCall], tool_calls)
            for tool_call in tool_calls:
                async_tool_fn = tool_name_map[tool_call.function.name]
                tool_call_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    additional_kwargs={"tool_calls": [tool_call]},
                )
                yield ChatResponse(
                    message=tool_call_message,
                    delta="",
                )
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
            enable_agent=chat_request.enable_agent,
            enable_attachments=chat_request.enable_attachments,
            kb_ids=chat_request.kb_ids,

        )


        input_messages = [
            {"role": "system", "content": system_prompt}
        ] + chat_request.messages
        messages = convert_to_chat_messages(input_messages)
        memory = BaseMemory()
        memory.from_messages(messages)
        if not chat_request.enable_agent:
            chat_request.max_steps = 1

        max_steps = chat_request.max_steps or self.max_steps

        @use_current_span(trace.get_current_span())
        async def gen():
            state = AgentState(
                llm=llm,
                step=0,
                stop_flag=False,
                memory=memory,
                tools=tools,
                tool_name_map=tool_name_map,
            )
            while state.step < max_steps:
                state.step += 1
                attachments = chat_request.messages[-1].get("attachments", [])
                logger.info(f"Running step {state.step}/{max_steps}.")
                try:
                    # 执行单步并更新状态
                    async for chunk in step_agent(state, attachments):
                        yield chunk
                        if state.stop_flag:
                            break
                    if state.stop_flag:
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
                        additional_kwargs={"failed": True, "step": state.step},
                    )
                    break

            if not state.stop_flag:
                logger.info(f"Running step {state.step}/{max_steps}.")
                logger.info("Reached maximum steps, ending conversation.")
                async for chunk in synthesize_agent(state):
                    yield chunk

        return gen()
