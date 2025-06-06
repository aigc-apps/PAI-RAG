from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json
from utils.messages import convert_to_openai_messages
from tools.mcp.mcp_client import resolve_mcp_clients
from utils.prompts import (
    PROMPT_WITH_DEEP_RESEARCH,
    PROMPT_WITHOUT_DEEP_RESEARCH,
    PROMPT_WITHOUT_TOOLS,
)
from utils.time_utils import get_prompt_current_time_str
from llama_index.tools.mcp.base import McpToolSpec
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from openai.types.chat import (
    ChatCompletionToolMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from search.aliyun_search_tool import aget_aliyun_search_tool
from utils.models import fetch_llm
from utils.constants import MAX_CHAT_STEPS

app = FastAPI()


async def get_model_instance(model_id: str):
    model = await fetch_llm(model_id)
    if model:
        model_source = model.get("source", "unknown")
        model_name = model.get("model_name", "unknown")
        if model_source == "openai":
            return (
                model_name,
                AsyncOpenAI(
                    api_key=model["api_key"], base_url="https://api.openai.com/v1"
                ).chat.completions,
            )
        elif model_source == "qwen":
            return (
                model_name,
                AsyncOpenAI(
                    api_key=model["api_key"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                ).chat.completions,
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_source}")
    else:
        raise ValueError(f"Model id {model_id} not exists.")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
async def gen_stream_response(model, model_name, messages, openai_tools):
    logger.info(f"messages {messages}")
    if openai_tools:
        return await model.create(
            model=model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            tools=openai_tools,
            tool_choice="auto",
        )
    else:
        return await model.create(
            model=model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )


async def process_mcp_tools():
    """
    process_mcp_tools will get the tools from MCP Client (only need to implement ClientSession) and convert them to LlamaIndex's FunctionTool objects and transformed tool name to tool Dict.
    Args:
    Returns:
        openai_tools: List[Dict]
        tools_name_to_fn: Dict[str, FunctionTool]

    """
    # TODO: 不用每个request都list_tools
    mcp_clients = await resolve_mcp_clients()
    openai_tools = []
    tools_name_to_fn = {}
    for mcp_client in mcp_clients:
        mcp_tool = McpToolSpec(client=mcp_client)
        mcp_server_name = mcp_client.name
        tools = await mcp_tool.to_tool_list_async()
        for tool in tools:
            # transform tool name to server_name--tool_name
            tool_name = mcp_server_name + "--" + tool.metadata.name
            tools_name_to_fn[tool_name] = tool
            tool_metadata = tool.metadata
            tool_metadata.name = tool_name
            openai_tools.append(tool_metadata.to_openai_tool())

    return openai_tools, tools_name_to_fn


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(tool_name, tool_args, tools_name_to_fn):
    return await tools_name_to_fn[tool_name].acall(**tool_args)


# 流式生成文本
async def generate_stream(model, model_name, messages, openai_tools, tools_name_to_fn):
    try:
        max_steps = MAX_CHAT_STEPS  # 防止无限循环的最大步骤数
        step_count = 0
        stop_flag = False
        while step_count < max_steps:
            response = await gen_stream_response(
                model, model_name, messages, openai_tools
            )
            draft_tool_calls = []
            draft_tool_calls_index = -1
            async for chunk in response:
                for choice in chunk.choices:
                    # 调用工具,收集工具参数
                    if choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            id = tool_call.id
                            name = tool_call.function.name
                            arguments = tool_call.function.arguments or ""

                            if id:
                                draft_tool_calls_index += 1
                                draft_tool_calls.append(
                                    {"id": id, "name": name, "arguments": arguments}
                                )

                            else:
                                draft_tool_calls[draft_tool_calls_index][
                                    "arguments"
                                ] += arguments
                    # 普通内容
                    if choice.delta.content:
                        yield "0:{text}\n".format(
                            text=json.dumps(choice.delta.content, ensure_ascii=False)
                        )
                        if (
                            isinstance(messages[-1], ChatCompletionMessage)
                            and messages[-1].role == "assistant"
                        ):
                            messages[-1].content += str(choice.delta.content)
                        else:
                            messages.append(
                                ChatCompletionMessage(
                                    role="assistant", content=str(choice.delta.content)
                                )
                            )  # 更新历史

                    # 模型生成已结束
                    # 1. 因需要调用工具而结束,根据参数调用工具
                    if choice.finish_reason == "tool_calls":
                        for tool_call in draft_tool_calls:
                            if tool_call and tool_call["arguments"].strip():
                                args = json.loads(tool_call["arguments"])
                            else:
                                args = {}
                            try:
                                result = await call_tool_with_retry(
                                    tool_call["name"], args, tools_name_to_fn
                                )

                                tool_result = result.content

                                # 返回工具调用和结果（标记9和a）
                                yield f'9:{json.dumps({"toolCallId": tool_call["id"], "toolName": tool_call["name"], "args": args}, ensure_ascii=False)}\n'
                                if tool_call["name"] == "search_web":
                                    yield f'a:{json.dumps({"toolCallId": tool_call["id"], "result": json.loads(tool_result)}, ensure_ascii=False)}\n'
                                else:
                                    yield f'a:{json.dumps({"toolCallId": tool_call["id"], "result": tool_result}, ensure_ascii=False)}\n'

                                # 将工具调用和结果加入消息历史,供模型继续推理
                                messages.append(
                                    ChatCompletionMessage(
                                        role="assistant",
                                        content="",
                                        tool_calls=[
                                            ChatCompletionMessageToolCall(
                                                id=tool_call["id"],
                                                type="function",
                                                function={
                                                    "name": tool_call["name"],
                                                    "arguments": tool_call["arguments"],
                                                },
                                            )
                                        ],
                                    )
                                )

                                messages.append(
                                    ChatCompletionToolMessageParam(
                                        role="tool",
                                        content=json.dumps(
                                            tool_result, ensure_ascii=False
                                        ),
                                        tool_call_id=tool_call["id"],
                                    )
                                )

                            except (ValueError, TypeError, KeyError) as e:
                                # 情况1: 参数错误
                                logger.exception("工具调用参数异常")
                                yield 'd:{"finishReason":"error", "error": "%s"}\n' % str(
                                    e
                                )
                                continue  # 继续调用 LLM
                            except Exception as e:
                                # 情况3: 其他错误
                                logger.exception("工具调用异常")
                                yield 'd:{"finishReason":"error", "error": "%s"}\n' % str(
                                    e
                                )
                                stop_flag = True

                    # 2.自然停止输出or因生成长度过长而结束
                    elif (
                        choice.finish_reason == "stop"
                        or choice.finish_reason == "length"
                    ):
                        stop_flag = True
                        if choice.delta.content:
                            yield "0:{text}\n".format(
                                text=json.dumps(
                                    choice.delta.content, ensure_ascii=False
                                )
                            )
                        yield 'd:{"finishReason":"{choice.finish_reason}"}\n'
                        break
                # 在include_usage为true时，最后一个chunk为空，本次chat请求使用的Token信息在最后一个chunk显示。
                if chunk.choices == []:
                    usage = chunk.usage
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens

                    yield 'd:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}}}}\n'.format(
                        reason="tool-calls" if len(draft_tool_calls) > 0 else "stop",
                        prompt=prompt_tokens,
                        completion=completion_tokens,
                    )
            if stop_flag:
                break
            step_count += 1
        if not stop_flag:
            yield "0:Agent stopped due to iteration limit\n"
            yield 'd:{"finishReason":"Agent stopped due to iteration limit"}\n'
    except Exception as e:
        yield 'd:{"finishReason":"error", "error": "%s"}\n' % str(e)
        raise


def x_options_to_prompt_mode(x_options):
    if "search" in x_options or "mcp" in x_options:
        if "thinking" in x_options:
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


async def handle_chat(request: Request):
    try:
        # 解析请求体
        data = await request.json()
        messages = data.get("messages", [])

        # 从 headers 中获取模型参数
        model_id = request.headers.get("X-Model-Id")
        x_options = (
            request.headers.get("X-Options").split(",")
            if request.headers.get("X-Options")
            else []
        )

        openai_tools = []
        tools_name_to_fn = {}
        system = data.get("system", x_options_to_prompt_mode(x_options))

        if "search" in x_options:
            search_openai_tools, search_tools_name_to_fn = (
                await aget_aliyun_search_tool()
            )
            openai_tools.extend(search_openai_tools)
            tools_name_to_fn.update(search_tools_name_to_fn)
        if "mcp" in x_options:
            mcp_openai_tools, mcp_tools_name_to_fn = await process_mcp_tools()
            openai_tools.extend(mcp_openai_tools)
            tools_name_to_fn.update(mcp_tools_name_to_fn)

        logger.info(f"[Tool] openai_tools: {openai_tools}")
        logger.info(f"[Tool] tools_name_to_fn: {tools_name_to_fn}")
        model_name, model = await get_model_instance(model_id)
        logger.info(f"[Model] model_name: {model_name}")

        # 构建openai_messages
        full_messages = [
            ChatCompletionSystemMessageParam(role="system", content=system)
        ] + messages
        full_messages = convert_to_openai_messages(full_messages)
        # 返回流式响应
        return StreamingResponse(
            generate_stream(
                model, model_name, full_messages, openai_tools, tools_name_to_fn
            ),
            media_type="text/event-stream",
            headers={"x-vercel-ai-data-stream": "v1"},
        )

    except Exception:
        logger.exception("Error in /api/chat")
        return Response(content="Internal Server Error", status_code=500)
