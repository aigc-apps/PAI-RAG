import json
import time
import uuid
from chat.llm.models import ChatResponseGenerator, ReasoningChunk, ToolResultChunk
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_chunk import ChoiceDelta, Choice as ChunkChoice
from openai.types.chat.chat_completion import Choice

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
    if tool_name == "aliyun-websearch" or tool_name == "tavily-websearch" or tool_name.startswith("search-knowledgebase"):
        tool_call_results = json.loads(tool_chunk.result).get("result", [])
        citations = [r.get("url", "") for r in tool_call_results]
        citation_details = [
            {
                "source": get_citation_source(tool_name),
                "text": r["content"],
                "name": r["title"],
                "url": r["url"],
                "score": r["score"],
            }
            for r in tool_call_results
        ]

    return citations, citation_details


async def convert_gen_to_stream_chat_completions(
    model: str,
    response_generator: ChatResponseGenerator
):
    chunk_index = 0
    chat_id = uuid.uuid4().hex
    total_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    citations, citation_details = [], []

    async for chunk in response_generator:
        if chunk.usage:
            total_usage.prompt_tokens += chunk.usage.prompt_tokens
            total_usage.completion_tokens += chunk.usage.completion_tokens
            total_usage.total_tokens += chunk.usage.total_tokens

        if isinstance(chunk, ToolResultChunk):
            citations, citation_details = extract_citations(chunk)
        chunk = ChatCompletionChunk(
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
            observation=chunk.result if isinstance(chunk, ToolResultChunk) else None,
            model=model,
            created=int(time.time()),
            citations=citations,
            citation_details=citation_details,
            object="chat.completion.chunk",
        )
        chunk_index += 1

        yield json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False)

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
        )
    yield json.dumps(stop_chunk.model_dump(mode="json"), ensure_ascii=False)



async def convert_gen_to_chat_completions(
    model: str,
    response_generator: ChatResponseGenerator
):
    chat_id = uuid.uuid4().hex
    total_usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    reasoning_content = ""
    content = ""
    steps = []
    citations, citation_details = [], []

    async for chunk in response_generator:
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
        )

    return message.model_dump(mode="json")
