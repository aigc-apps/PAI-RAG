import os
import aiohttp
from loguru import logger
from common.chat.models import ChatAgentRequest
from evaluation.utils import parse_function_call, parse_sse_events
from evaluation.evaluator.utils import create_evaluator
from llama_index.core.llms import LLM
import asyncio


BACKEND_PORT = os.environ.get("BACKEND_PORT", "8682")
CHAT_API = f"http://127.0.0.1:{BACKEND_PORT}/v1/chat/completions"

async def run_agent(chat_request: ChatAgentRequest):
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

async def run_evaluator(input: str, prediction: str, reference: str, eval_config: dict, eval_llm: LLM = None):
    evaluator = create_evaluator(eval_config, eval_llm)
    result = await evaluator.evaluate_async(input, prediction, reference)
    return result

if __name__ == '__main__':
    chat_request = ChatAgentRequest(
        model="qwen-max-xw5",
        messages=[{"role": "user", "content": "1+1=?"}],
        stream=True,
        mcp_ids=[],
        enable_search=True,
        enable_agent=False,
        chatbot_id='',
    )
    res, _, _ = asyncio.run(run_agent(chat_request))
    print("res", res)
