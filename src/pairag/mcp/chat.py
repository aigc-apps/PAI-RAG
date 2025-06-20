from typing import List, AsyncGenerator
from fastapi.responses import StreamingResponse
import json
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
from loguru import logger
from opentelemetry import trace
from pairag.mcp.trace.pai_agent_wrapper import pai_agent_wrapper
from pairag.integrations.trace.base import use_current_span
from tenacity import retry, stop_after_attempt, wait_fixed
from pairag.mcp.constants import MAX_CHAT_STEPS


async def response_to_raw(
    astream_chat_response: AsyncGenerator,
) -> AsyncGenerator:
    async for chat_response in astream_chat_response:
        yield chat_response.raw


async def gen_stream_response(llm, messages, openai_tools):
    logger.info(f"messages {messages}, tools {openai_tools}")
    chat_messages = []
    for m in messages:
        if isinstance(m, dict):
            chat_messages.append(ChatMessage.parse_obj(m))
        elif isinstance(m, ChatMessage):
            chat_messages.append(m)
        else:
            logger.error(f"wrong message type, {type(m)}: {m}")

    if openai_tools:
        response = await llm.astream_chat(
            messages=chat_messages,
            tools=openai_tools,
            tool_choice="auto",
            stream_options={"include_usage": True},
        )
    else:
        response = await llm.astream_chat(
            messages=chat_messages,
            stream_options={"include_usage": True},
        )

    return response_to_raw(response)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args):
    return await async_fn.acall(**fn_args)


# 流式生成文本
async def generate_stream(llm, messages, tools: List[FunctionTool]):
    try:
        openai_tools = []
        tool_name_map = {}
        for tool in tools:
            openai_tools.append(
                {"type": "function", "function": tool.metadata.to_openai_function()}
            )
            tool_name_map[tool.metadata.name] = tool

        max_steps = MAX_CHAT_STEPS  # 防止无限循环的最大步骤数
        step_count = 0
        stop_flag = False
        while step_count < max_steps:
            response = await gen_stream_response(llm, messages, openai_tools)
            draft_tool_calls = []
            draft_tool_calls_index = -1
            last_chunk = None
            async for chunk in response:
                if stop_flag:
                    logger.info("Stop early due to errors.")
                    break
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
                            isinstance(messages[-1], ChatMessage)
                            and messages[-1].role == "assistant"
                        ):
                            messages[-1].content += str(choice.delta.content)
                        else:
                            messages.append(
                                ChatMessage(
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
                                async_fn = tool_name_map[tool_call["name"]]
                                result = await call_tool_with_retry(async_fn, args)

                                tool_result = result.content

                                # 返回工具调用和结果（标记9和a）
                                yield f'9:{json.dumps({"toolCallId": tool_call["id"], "toolName": tool_call["name"], "args": args}, ensure_ascii=False)}\n'
                                if tool_call["name"] in [
                                    "search-web",
                                    "think-and-planning",
                                ]:
                                    yield f'a:{json.dumps({"toolCallId": tool_call["id"], "result": json.loads(tool_result)}, ensure_ascii=False)}\n'
                                else:
                                    yield f'a:{json.dumps({"toolCallId": tool_call["id"], "result": tool_result}, ensure_ascii=False)}\n'

                                # 将工具调用和结果加入消息历史,供模型继续推理
                                messages.append(
                                    ChatMessage(
                                        role="assistant",
                                        content="",
                                        additional_kwargs={
                                            "tool_calls": [
                                                {
                                                    "id": tool_call["id"],
                                                    "type": "function",
                                                    "function": {
                                                        "name": tool_call["name"],
                                                        "arguments": tool_call[
                                                            "arguments"
                                                        ],
                                                    },
                                                },
                                            ]
                                        },
                                    )
                                )

                                messages.append(
                                    ChatMessage(
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
                logger.info(f"Agent loop break at step {step_count}")
                break
            step_count += 1
        if not stop_flag:
            finish_reason = "Agent stopped due to iteration limit"
            yield "0:{reason}\n".format(
                reason=json.dumps(finish_reason, ensure_ascii=False)
            )
            yield 'd:{{"finishReason":"{reason}"}}\n'.format(reason=finish_reason)
    except Exception as e:
        yield 'd:{"finishReason":"error", "error": "%s"}\n' % str(e)
        raise


@pai_agent_wrapper
async def handle_chat(llm, messages, tools: List[FunctionTool]):
    # wrap to inherit current context
    @use_current_span(trace.get_current_span())
    def _gen_streaming_response():
        return generate_stream(llm, messages, tools)

    # 返回流式响应
    return StreamingResponse(
        _gen_streaming_response(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-data-stream": "v1"},
    )
