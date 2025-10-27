import asyncio
import json
import time
import uuid
from chat.llm.models import ChatResponseGenerator, ErrorChunk, ReasoningChunk, ToolResultChunk, TextChunk
from extensions.guardrail.guardrail_check import TextCheckResult
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_chunk import ChoiceDelta, Choice as ChunkChoice
from openai.types.chat.chat_completion import Choice
from config.providers.guardrail_provider import guardrail_provider
from extensions.guardrail.config import CHECK_OUTPUT_CHUNK_SIZE

from loguru import logger


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
):
    chunk_index = 0
    chat_id = uuid.uuid4().hex
    total_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    citations, citation_details = [], []

    current_content = ""
    check_tasks = []
    output_check_result = TextCheckResult()
    checker = guardrail_provider.get_checker() if enable_output_check else None

    async for chunk in response_generator:
        # 出错直接返回
        if output_check_result.reject:
            break

        if chunk.usage:
            total_usage.prompt_tokens += chunk.usage.prompt_tokens
            total_usage.completion_tokens += chunk.usage.completion_tokens
            total_usage.total_tokens += chunk.usage.total_tokens
            continue

        if isinstance(chunk, ToolResultChunk):
            citations, citation_details = extract_citations(chunk)
        else:
            citations, citation_details = [], []

        current_content += chunk.delta
        if checker and len(current_content) >= CHECK_OUTPUT_CHUNK_SIZE:
            check_tasks.append(asyncio.create_task(checker.acheck_output(text=current_content, current_result=output_check_result)))
            current_content = ""


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
            break


    if not output_check_result.reject and len(check_tasks) > 0:
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
):
    chat_id = uuid.uuid4().hex
    total_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    reasoning_content = ""
    content = ""
    steps = []
    citations, citation_details = [], []


    checked = False

    async for chunk in response_generator:
        # 出错直接返回
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

    if not checked and enable_output_check:
        checker = guardrail_provider.get_checker()
        if checker:
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

    return message.model_dump(mode="json")
