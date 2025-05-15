from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json
import datetime
from utils.messages import convert_to_openai_messages
from tools.mcp.mcp_base import McpToolSpec
from tools.mcp.mcp_client import resolve_mcp_clients
from tenacity import retry, stop_after_attempt, wait_exponential


app = FastAPI()

# 系统时间
today = datetime.datetime.now().strftime("%Y-%m-%d")

# 系统提示
SYSTEM_PROMPT = f"""
当前系统时间：{today}

1. 你是一个 agent，请持续调用工具直至完美完成用户的任务，停止调用工具后，系统会自动交还控制权给用户。
2. 请善加利用你的工具收集相关信息，绝对不要猜测或编造答案。
3. 在每次调用任务工具之前，
  - 你必须**首先思考和规划**：针对用户的任务进行详细思考，并给出你对拆解后任务的规划，同时需要对之前工具调用的结果进行深入反思并继续规划（如有）。
  - 思考完成之后不需要等待工具返回，你可以继续调用其他任务工具，你一次可以调用多个任务工具。
  - 任务工具调用完成之后，你可以停止输出，系统会把工具调用结果给你，你必须再次思考和规划，然后继续调用任务工具，如此循环，直到完美地完成用户的任务或者达到循环的最大步骤数。
"""


def get_model_instance(model_name: str, model_source: str, api_key: str):
    if model_source == "openai":
        return AsyncOpenAI(
            api_key=api_key, base_url="https://api.openai.com/v1"
        ).chat.completions
    elif model_source == "qwen":
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).chat.completions
    else:
        raise ValueError(f"Unsupported model provider: {model_source}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
async def gen_stream_response(model, model_name, messages, openai_tools):
    return await model.create(
        model=model_name,
        messages=messages,
        stream=True,
        tools=openai_tools,
        tool_choice="auto",
    )


# 流式生成文本
async def generate_stream(model, model_name, messages, mcp_tools):
    openai_tools = []
    tools_name_to_fn = {}
    for tool in mcp_tools:
        openai_tools.append(tool.metadata.to_openai_tool())
        tools_name_to_fn[tool.metadata.name] = tool

    max_steps = 10  # 防止无限循环的最大步骤数
    step_count = 0
    while step_count < max_steps:
        response = await gen_stream_response(model, model_name, messages, openai_tools)
        draft_tool_calls = []
        draft_tool_calls_index = -1
        async for chunk in response:
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    continue
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
                elif choice.delta.content:
                    yield "0:{text}\n".format(text=json.dumps(choice.delta.content))
                    messages.append(
                        {"role": "assistant", "content": choice.delta.content}
                    )  # 更新历史

                if choice.finish_reason == "tool_calls":
                    for tool_call in draft_tool_calls:
                        args = json.loads(tool_call["arguments"])
                        try:
                            result = await tools_name_to_fn[tool_call["name"]].acall(
                                **args
                            )
                            tool_result = result.content

                            # 返回工具调用和结果（标记9和a）
                            yield f'9:{json.dumps({"toolCallId": tool_call["id"], "toolName": tool_call["name"], "args": json.loads(tool_call["arguments"])})}\n'
                            yield f'a:{json.dumps({"toolCallId": tool_call["id"], "result": tool_result})}\n'

                            # 将工具调用和结果加入消息历史,供模型继续推理
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": tool_call["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tool_call["name"],
                                                "arguments": tool_call["arguments"],
                                            },
                                        }
                                    ],
                                }
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "content": tool_result,
                                    "tool_call_id": tool_call["id"],
                                }
                            )

                        except Exception as e:
                            print(f"工具调用异常: {str(e)}")
                            error_message = {
                                "finishReason": "工具调用发生未知错误，请检查输入或重试"
                            }
                            yield "d:{text}\n".format(text=json.dumps(error_message))

            if chunk.choices == []:
                usage = chunk.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens

                yield 'd:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}}}}\n'.format(
                    reason="tool-calls" if len(draft_tool_calls) > 0 else "stop",
                    prompt=prompt_tokens,
                    completion=completion_tokens,
                )
        step_count += 1


async def handle_chat(request: Request):
    try:
        # 解析请求体
        data = await request.json()
        messages = data.get("messages", [])
        system = data.get("system", SYSTEM_PROMPT)

        # 从 headers 中获取模型参数
        model_name = request.headers.get("X-Model-Name")
        api_key = request.headers.get("X-Api-Key")
        model_source = request.headers.get("X-Model-Source")

        model = get_model_instance(model_name, model_source, api_key)

        # 构建openai_messages
        full_messages = [{"role": "system", "content": system}] + messages
        full_messages = convert_to_openai_messages(full_messages)

        mcp_clients = await resolve_mcp_clients()
        mcp_tools = []
        for mcp_server_name, mcp_client in mcp_clients:
            mcp_tool = McpToolSpec(mcp_server_name=mcp_server_name, client=mcp_client)
            tools = await mcp_tool.to_tool_list_async()
            mcp_tools.extend(tools)
        # 返回流式响应
        return StreamingResponse(
            generate_stream(model, model_name, full_messages, mcp_tools),
            media_type="text/event-stream",
            headers={"x-vercel-ai-data-stream": "v1"},
        )

    except Exception as e:
        print("Error in /api/chat:", e)
        return Response(content="Internal Server Error", status_code=500)
