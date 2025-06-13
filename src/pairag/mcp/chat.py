from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from pairag.mcp.constants import MAX_CHAT_STEPS
from pairag.mcp.mcp_client import resolve_mcp_clients
from llama_index.tools.mcp.base import McpToolSpec
from loguru import logger
from openai.types.chat import (
    ChatCompletionToolMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from opentelemetry import trace
from pairag.mcp.trace.pai_agent_wrapper import pai_agent_wrapper, with_current_context
from tenacity import retry, stop_after_attempt, wait_fixed

app = FastAPI()


async def gen_stream_response(model, model_name, messages, openai_tools):
    logger.info(f"messages {messages}")
    if openai_tools:
        return await model.create(
            model=model_name,
            messages=messages,
            stream=True,
            tools=openai_tools,
            tool_choice="auto",
            stream_options={"include_usage": True},
        )
    else:
        return await model.create(
            model=model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )


async def process_mcp_tools(mcp_server_configs):
    """
    process_mcp_tools will get the tools from MCP Client (only need to implement ClientSession) and convert them to LlamaIndex's FunctionTool objects and transformed tool name to tool Dict.
    Args:
    Returns:
        openai_tools: List[Dict]
        tools_name_to_fn: Dict[str, FunctionTool]

    """
    # TODO: 不用每个request都list_tools
    mcp_clients = await resolve_mcp_clients(mcp_server_configs)
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
@with_current_context
async def generate_stream(
    model, model_name, messages, openai_tools, tools_name_to_fn, current_context
):
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
            last_chunk = None
            async for chunk in response:
                last_chunk = chunk
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
                                if tool_call["name"] in [
                                    "search_web",
                                    "think_and_planning",
                                ]:
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
                                break

                    # 2.自然停止输出or因生成长度过长而结束
                    elif (
                        choice.finish_reason == "stop"
                        or choice.finish_reason == "length"
                    ):
                        stop_flag = True
                        yield 'd:{"finishReason":"{choice.finish_reason}"}\n'
                        break
            # 在include_usage为true时，最后一个chunk为空，本次chat请求使用的Token信息在最后一个chunk显示。
            if last_chunk and last_chunk.choices == []:
                usage = last_chunk.usage
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


@pai_agent_wrapper
async def handle_chat(model, model_name, messages, tools, tools_name_to_fn):
    current_span = trace.get_current_span()
    current_context = trace.set_span_in_context(current_span)

    # 返回流式响应
    return StreamingResponse(
        generate_stream(
            model,
            model_name,
            messages,
            tools,
            tools_name_to_fn,
            current_context,
        ),
        media_type="text/event-stream",
        headers={"x-vercel-ai-data-stream": "v1"},
    )
