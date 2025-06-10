from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from pairag.mcp.mcp_client import resolve_mcp_clients
from llama_index.tools.mcp.base import McpToolSpec
from loguru import logger
from openai.types.chat import (
    ChatCompletionToolMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from opentelemetry import trace
from pairag.mcp.trace.pai_query_wrapper import pai_query_wrapper, with_current_context

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


# 流式生成文本
@with_current_context
async def generate_stream(
    model, model_name, messages, openai_tools, tools_name_to_fn, current_context
):
    max_steps = 5  # 防止无限循环的最大步骤数
    step_count = 0
    while step_count < max_steps:
        stop_flag = False
        response = await gen_stream_response(model, model_name, messages, openai_tools)
        draft_tool_calls = []
        draft_tool_calls_index = -1
        async for chunk in response:
            for choice in chunk.choices:
                # 模型生成已结束
                if choice.finish_reason == "stop":
                    stop_flag = True
                    if choice.delta.content:
                        yield "0:{text}\n".format(
                            text=json.dumps(choice.delta.content, ensure_ascii=False)
                        )
                    yield 'd:{"finishReason":"stop"}\n'
                    break
                # 调用工具,收集工具参数
                elif choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        id = tool_call.id
                        name = tool_call.function.name
                        arguments = tool_call.function.arguments or ""

                        if id is not None and id != "":
                            draft_tool_calls_index += 1
                            draft_tool_calls.append(
                                {"id": id, "name": name, "arguments": arguments}
                            )

                        else:
                            draft_tool_calls[draft_tool_calls_index][
                                "arguments"
                            ] += arguments
                # 普通内容
                elif choice.delta.content:
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

                # 根据参数调用工具
                if choice.finish_reason == "tool_calls":
                    for tool_call in draft_tool_calls:
                        if tool_call and tool_call["arguments"].strip():
                            args = json.loads(tool_call["arguments"])
                        else:
                            args = {}
                        try:
                            result = await tools_name_to_fn[tool_call["name"]].acall(
                                **args
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
                                    content=json.dumps(tool_result, ensure_ascii=False),
                                    tool_call_id=tool_call["id"],
                                )
                            )

                        except Exception as e:
                            logger.error(f"工具调用异常: {str(e)}")
                            error_message = {"finishReason": "工具调用发生未知错误，请检查输入或重试"}
                            yield "d:{text}\n".format(
                                text=json.dumps(error_message, ensure_ascii=False)
                            )

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


@pai_query_wrapper
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
