from loguru import logger
import aiohttp
import json
from typing import Dict, AsyncGenerator


def parse_function_call(json_line:dict, observation:str):
    result = {}
    if json_line and json_line.get("actions"):
        result = json_line["actions"][0]

    result["observation"] = observation
    return result

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
