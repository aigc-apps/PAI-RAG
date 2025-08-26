import asyncio
import functools
from typing import cast
from common.chat.chat_utils import response_from_text, response_gen_from_text
from common.chat.models import ChatAgentRequest
from extensions.guardrail.config import CHECK_OUTPUT_CHUNK_SIZE
from extensions.guardrail.guardrail_check import TextCheckResult
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from config.providers.guardrail_provider import guardrail_provider
from llama_index.core.base.llms.types import ChatResponseAsyncGen


def extract_user_message(raw_msg: ChatCompletionMessageParam) -> str:
    content = raw_msg.get("content", "")
    logger.info(f"Extracted {content} from {raw_msg}.")
    if isinstance(content, str):
        return content
    else:
        user_content = ""
        for block in content:
            user_content += block.get("text", "")
        return user_content


def ai_guardrail(func):
    """
    装饰器：异步检查chat_request。
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        chat_request = kwargs.get("chat_request", None)
        if not chat_request:
            raise ValueError("Chat request is not valid")

        chat_request = cast(ChatAgentRequest, chat_request)
        # 创建审核器
        checker = guardrail_provider.get_checker()

        if checker is None or (not chat_request.enable_input_guardrail and not chat_request.enable_output_guardrail):
            # 无须护栏审核，直接返回
            return func(*args, **kwargs)

        guardrail_violation_kwargs = {
            "safety_violation": True,
            "STOP_FLAG": True,
        }

        # 输入护栏
        if chat_request.enable_input_guardrail:
            user_message = extract_user_message(chat_request.messages[-1])
            check_result = await checker.acheck_input(text=user_message)
            if check_result.reject:
                return response_gen_from_text(check_result.advice or chat_request.guardrail_hint, additional_kwargs=guardrail_violation_kwargs)

        if not chat_request.enable_output_guardrail:
            return func(*args, **kwargs)
        else:
            async def wrapped_generator():
                current_content = ""
                output_check_result = TextCheckResult()

                check_tasks = []
                check_failed = False

                response_gen: ChatResponseAsyncGen = func(*args, **kwargs)

                async for chunk in response_gen:
                    if output_check_result.reject:
                        check_failed=True
                        yield response_from_text(output_check_result.advice or chat_request.guardrail_hint, additional_kwargs=guardrail_violation_kwargs)
                        break

                    chunk.additional_kwargs["STOP_FLAG"] = False

                    yield chunk
                    current_content += chunk.delta
                    if len(current_content) >= CHECK_OUTPUT_CHUNK_SIZE:
                        check_tasks.append(asyncio.create_task(checker.acheck_output(text=current_content, current_result=output_check_result)))
                        current_content = ""

                if current_content:
                    check_tasks.append(asyncio.create_task(checker.acheck_output(text=current_content, current_result=output_check_result)))

                if not check_failed:
                    # 检查所有未完成的任务
                    await asyncio.gather(*check_tasks)
                    if output_check_result.reject:
                        yield response_from_text(output_check_result.advice or chat_request.guardrail_hint, additional_kwargs=guardrail_violation_kwargs)
                    else:
                        yield response_from_text(text="", additional_kwargs={"STOP_FLAG": True})


            return wrapped_generator()
    return wrapper
