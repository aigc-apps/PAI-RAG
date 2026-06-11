import asyncio
import json
import time
import uuid
from common.llm.models import ChatResponseGenerator, ErrorChunk, ReasoningChunk, ToolResultChunk, TextChunk
from extensions.guardrail.guardrail_check import TextCheckResult
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_chunk import ChoiceDelta, Choice as ChunkChoice
from openai.types.chat.chat_completion import Choice
from extensions.guardrail.config import CHECK_OUTPUT_CHUNK_SIZE, CHECK_OUTPUT_CHUNK_OVERLAP
from extensions.guardrail.guardrail_check import GuardrailChecker
from sqlmodel.ext.asyncio.session import AsyncSession

from loguru import logger
from extensions.trace.context import get_request_id
from typing import List


def parse_llm_json(json_str: str) -> dict:
    start_pos = json_str.find("{")
    end_pos = json_str.rfind("}")

    if start_pos == -1 or end_pos == -1 or start_pos >= end_pos:
        logger.warning("Invalid JSON string: {json_str}")
        return {}

    return json.loads(json_str[start_pos:end_pos+1])



def get_citation_source(tool_name: str) -> str:
    return "knowledgebase" if tool_name.startswith("search-knowledgebase") else "web"


def extract_citations(tool_chunk: ToolResultChunk):
    citations, citation_details = [], []
    tool_name = tool_chunk.tool.function.name or "dummy"
    seen_files = set()
    if tool_name == "aliyun-websearch" or tool_name == "tavily-websearch" or tool_name.startswith("search-knowledgebase"):
        if not tool_chunk.result:
            return citations, citation_details

        tool_call_results = json.loads(tool_chunk.result).get("result", []) or []

        for result in tool_call_results:
            file_name = result.get("title", "")
            if not file_name or file_name in seen_files:
                continue

            citations.append(file_name)
            citation_details.append({
                "source": get_citation_source(tool_name),
                "text": result.get("content", ""),
                "name": file_name,
                "url": result.get("url", ""),
                "score": result.get("score", 0),
            })

            seen_files.add(file_name)

    return citations, citation_details



MAX_TOOL_HISTORY_CHARS = 20000
TOOL_HISTORY_TRUNCATED_MARKER = "\n...[content truncated]"


def _collect_tool_history(chunk: ToolResultChunk, tool_history_messages: List[dict]):
    tool_call = chunk.tool
    tool_history_messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
        }]
    })
    raw_content = chunk.result or chunk.error or ""
    if isinstance(raw_content, str) and len(raw_content) > MAX_TOOL_HISTORY_CHARS:
        raw_content = raw_content[:MAX_TOOL_HISTORY_CHARS] + TOOL_HISTORY_TRUNCATED_MARKER
    tool_history_messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": raw_content,
    })


async def error_chunk_gen(message: str, exception: Exception | None = None) -> ChatResponseGenerator:
    yield ErrorChunk(
        delta=message,
        exception=str(exception) if exception else exception,
    )


async def convert_gen_to_stream_chat_completions(
    model: str,
    response_generator: ChatResponseGenerator,
    enable_output_check: bool = False,
    guardrail_hint: str | None = None,
    checker: GuardrailChecker | None = None,
    session: AsyncSession = None,
    user_id: str = None,
    session_id: str = None,
    user_message: dict = None,
):
    logger.info(f"convert_gen_to_stream_chat_completions: model={model}, enable_output_check={enable_output_check}, guardrail_hint={guardrail_hint}")
    if enable_output_check and not checker:
        logger.warning("convert_gen_to_stream_chat_completions: checker is None, set enable_output_check to False")
        enable_output_check = False

    chunk_index = 0
    chat_id = get_request_id() or uuid.uuid4().hex
    total_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    citations, citation_details = [], []

    current_content = ""
    check_tasks = []
    output_check_result = TextCheckResult()
    fail_fast = False
    final_content = ""  # 累积完整的助手回复内容
    tool_history_messages = []  # 收集 tool 交互消息用于保存历史

    try:
        async for chunk in response_generator:
            # 出错直接返回
            if output_check_result.reject:
                logger.info("convert_gen_to_stream_chat_completions: output_check_result.reject=True, break")
                fail_fast = True
                break

            if chunk.usage:
                total_usage.prompt_tokens += chunk.usage.prompt_tokens
                total_usage.completion_tokens += chunk.usage.completion_tokens
                total_usage.total_tokens += chunk.usage.total_tokens
                continue

            if isinstance(chunk, ToolResultChunk):
                citations, citation_details = extract_citations(chunk)
                _collect_tool_history(chunk, tool_history_messages)

            current_content += chunk.delta
            final_content += chunk.delta
            if enable_output_check and checker and len(current_content) >= CHECK_OUTPUT_CHUNK_SIZE:
                check_tasks.append(asyncio.create_task(checker.acheck_output(text=current_content, current_result=output_check_result)))
                current_content = current_content[-CHECK_OUTPUT_CHUNK_OVERLAP:]

            completion_chunk = ChatCompletionChunk(
                id=chat_id,
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(
                            role="assistant",
                            content=chunk.delta,
                            reasoning_content=chunk.reasoning_delta if isinstance(chunk, ReasoningChunk) else None,
                        ),
                        index=chunk_index,
                        finish_reason=None,
                    )
                ],
                actions=[action.model_dump(mode="json") for action in chunk.tool_calls] if chunk.tool_calls else None,
                observation=chunk.model_dump(mode="json") if isinstance(chunk, ToolResultChunk) else None,
                trace_id=chunk.trace_id if isinstance(chunk, TextChunk) else None,
                model=model,
                created=int(time.time()),
                citations=citations,
                citation_details=citation_details,
                object="chat.completion.chunk",
            )
            chunk_index += 1

            yield json.dumps(completion_chunk.model_dump(mode="json"), ensure_ascii=False)

            if isinstance(chunk, ErrorChunk):
                fail_fast = True
                break
    finally:
        if response_generator and hasattr(response_generator, "aclose"):
            await response_generator.aclose()
            logger.info("convert_gen_to_stream_chat_completions: response_generator closed.")
            await session.close()
            logger.info("convert_gen_to_stream_chat_completions: session closed.")

        # 保存会话历史
        if final_content and user_id and session_id and user_message:
            try:
                from service.cache.session_history_manager import session_history_manager
                assistant_message = {
                    "role": "assistant",
                    "content": final_content,
                }
                await session_history_manager.save_messages(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    tool_messages=tool_history_messages if tool_history_messages else None,
                )
                logger.info(f"Session history saved in stream mode for user={user_id}, session={session_id}")
            except Exception as e:
                logger.error(f"Failed to save session history in stream mode: {e}", exc_info=True)

    if not fail_fast and len(current_content) > CHECK_OUTPUT_CHUNK_OVERLAP and enable_output_check and checker:
        check_tasks.append(asyncio.create_task(checker.acheck_output(text=current_content, current_result=output_check_result)))

    if not fail_fast and len(check_tasks) > 0:
        await asyncio.gather(*check_tasks)

    if output_check_result.reject:
        error_chunk = ChatCompletionChunk(
            id=chat_id,
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        role="assistant",
                        content=output_check_result.advice or guardrail_hint,
                    ),
                    index=chunk_index,
                    finish_reason=None,
                )
            ],
            safety_violation=True,
            model=model,
            created=int(time.time()),
            citations=citations,
            citation_details=citation_details,
            object="chat.completion.chunk",
        )
        yield json.dumps(error_chunk.model_dump(mode="json"), ensure_ascii=False)

    stop_chunk = ChatCompletionChunk(
            id=chat_id,
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        role="assistant",
                        content="",
                    ),
                    index=chunk_index,
                    finish_reason="stop",
                )
            ],
            model=model,
            created=int(time.time()),
            object="chat.completion.chunk",
            citation_details=citation_details,
            citations=citations,
            usage=total_usage
        )
    yield json.dumps(stop_chunk.model_dump(mode="json"), ensure_ascii=False)



async def convert_gen_to_chat_completions(
    model: str,
    response_generator: ChatResponseGenerator,
    enable_output_check: bool = False,
    guardrail_hint: str | None = None,
    checker: GuardrailChecker | None = None,
    user_id: str = None,
    session_id: str = None,
    user_message: dict = None,
):
    chat_id = get_request_id() or uuid.uuid4().hex

    total_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    reasoning_content = ""
    content = ""
    steps = []
    citations, citation_details = [], []


    checked = False
    tool_history_messages = []

    async for chunk in response_generator:
        if isinstance(chunk, ErrorChunk):
            logger.info(f"Input guardrail failed: {chunk.delta}, directly return.")
            content = chunk.delta
            checked = True
            break

        if chunk.usage:
            total_usage.prompt_tokens += chunk.usage.prompt_tokens
            total_usage.completion_tokens += chunk.usage.completion_tokens
            total_usage.total_tokens += chunk.usage.total_tokens

        if isinstance(chunk, ReasoningChunk) and chunk.reasoning_delta:
            reasoning_content += chunk.reasoning_delta

        content += chunk.delta

        if isinstance(chunk, ToolResultChunk):
            steps.append(chunk)
            citations, citation_details = extract_citations(chunk)
            _collect_tool_history(chunk, tool_history_messages)

    if not checked and enable_output_check and checker:
        current_result = TextCheckResult()
        await checker.acheck_output(text=content, current_result=current_result)
        if current_result.reject:
            logger.warning(f"Check output text failed: {content}")
            content = current_result.advice or guardrail_hint

    message = ChatCompletion(
            id=chat_id,
            model=model,
            created=int(time.time()),
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content,
                        reasoning_content=reasoning_content if reasoning_content else None,
                    ),
                    finish_reason="stop",
                )
            ],
            steps=steps,
            citations=citations,
            citation_details=citation_details,
            usage=total_usage
        )

    # 保存会话历史
    if content and user_id and session_id and user_message:
        try:
            from service.cache.session_history_manager import session_history_manager
            assistant_message = {
                "role": "assistant",
                "content": content,
            }
            await session_history_manager.save_messages(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                tool_messages=tool_history_messages if tool_history_messages else None,
            )
            logger.info(f"Session history saved in non-stream mode for user={user_id}, session={session_id}")
        except Exception as e:
            logger.error(f"Failed to save session history in non-stream mode: {e}", exc_info=True)

    return message.model_dump(mode="json")
