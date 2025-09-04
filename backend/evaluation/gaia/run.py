import os
from common.chat.models import ChatAgentRequest
from loguru import logger
import asyncio
import aiohttp
import json
from typing import Dict, AsyncGenerator

BACKEND_PORT = os.environ.get("BACKEND_PORT", "8682")
CHAT_API = f"http://127.0.0.1:{BACKEND_PORT}/v1/chat/completions"

def parse_function_call(json_line:dict, observation:str):
    actions = json_line.get("actions", [])[0] if json_line else {}
    actions["observation"] = observation
    return actions

async def parse_sse_events(response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
    """解析SSE格式的事件流"""
    buffer = ""

    async for line in response.content.iter_any():
        if not line:
            continue

        try:
            decoded_line = line.decode('utf-8')
            buffer += decoded_line

            # SSE事件以双换行符分隔 (HTTP 协议统一使用 \r\n 作为换行符)
            while '\r\n\r\n' in buffer:
                event_data, buffer = buffer.split('\r\n\r\n', 1)
                event_lines = event_data.strip().split('\n')

                # 提取data字段
                data_lines = [line[6:] for line in event_lines
                             if line.startswith('data: ')]

                if not data_lines:
                    continue

                # 拼接所有data行
                json_str = ''.join(data_lines).strip()


                # 尝试解析JSON
                try:
                    event = json.loads(json_str)
                    yield event
                except json.JSONDecodeError:
                    # 可能是不完整的JSON，保留到下一次
                    buffer = json_str + buffer
                    continue

        except Exception as e:
            logger.debug(f"Error processing SSE event: {e}")
            continue

async def run_gaia_agent(chat_request: ChatAgentRequest):
    logger.info(f"Chat agent body: {chat_request} via backend api {CHAT_API}")
    chat_request_dict = chat_request.model_dump()
    async with aiohttp.ClientSession() as session:
        async with session.post(CHAT_API, json=chat_request_dict) as response:
            if response.status == 200:
                result = ""
                execution_metadata = []
                last_function_call_dict = None
                async for sse_event in parse_sse_events(response):
                    if sse_event.get("choices", [])[0].get("finish_reason", "") == "stop":
                        break
                    content = sse_event.get("choices", [])[0].get("delta", {}).get("content", "")
                    if content:
                        result += content
                    observation = sse_event.get("observation", "")
                    if observation:
                        execution_metadata.append(parse_function_call(last_function_call_dict, observation))
                    last_function_call_dict = sse_event

                logger.info(f"Chat agent final response: {result}")
                return result, execution_metadata, True
            else:
                error_text = await response.text()
                logger.error(f"Request failed with status {response.status}, body: {error_text}")
                return f"Request failed: {error_text}", [], False

if __name__ == '__main__':
    chat_request = ChatAgentRequest(
        model="qwen-max-xw5",
        messages=[{"role": "user", "content": "In Audre Lorde’s poem “Father Son and Holy Ghost”, what is the number of the stanza in which some lines are indented?"}],
        stream=True,
        mcp_ids=[],
        enable_search=True,
        enable_agent=False,
        chatbot_id='',
    )
    res = asyncio.run(run_gaia_agent(chat_request))
    print("res", res)
