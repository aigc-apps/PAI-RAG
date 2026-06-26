import json
from common.llm.models import ChatResponseGenerator, ErrorChunk, ToolResultChunk

from loguru import logger
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
