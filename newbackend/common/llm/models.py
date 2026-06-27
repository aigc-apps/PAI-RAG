from typing import List, Optional
from enum import Enum
from pydantic import BaseModel
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, CompletionUsage
from typing import AsyncGenerator


def update_tool_calls(
    tool_calls: List[ChoiceDeltaToolCall],
    tool_calls_delta: Optional[List[ChoiceDeltaToolCall]],
) -> List[ChoiceDeltaToolCall]:
    """
    Use the tool_calls_delta objects received from openai stream chunks
    to update the running tool_calls object.

    Handles parallel tool calls by matching on the ``index`` field.
    Each distinct index represents a separate tool call.

    Args:
        tool_calls: the accumulated list of tool calls so far.
        tool_calls_delta: new delta(s) from the current chunk.

    Returns:
        The updated tool calls list.
    """
    if tool_calls_delta is None or len(tool_calls_delta) == 0:
        return tool_calls

    for tc_delta in tool_calls_delta:
        # Find existing tool_call with the same index
        existing = None
        for tc in tool_calls:
            if tc.index == tc_delta.index:
                existing = tc
                break

        if existing is None:
            # First chunk for this index — start a new tool call entry
            tool_calls.append(tc_delta)
        else:
            # Continuation of an existing tool call — accumulate deltas
            assert existing.function is not None
            assert tc_delta.function is not None

            if existing.function.arguments is None:
                existing.function.arguments = ""
            if existing.function.name is None:
                existing.function.name = ""
            if existing.id is None:
                existing.id = ""

            existing.function.arguments += tc_delta.function.arguments or ""
            existing.function.name += tc_delta.function.name or ""
            # Only set id from delta if existing id is still empty;
            # avoids concatenating the same id across repeated chunks.
            if tc_delta.id and not existing.id:
                existing.id = tc_delta.id

    return tool_calls


DEFAULT_TEMPERATURE = 0.1
DEFAULT_CONTEXT_WINDOW = 110000  # tokens
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"


class ModelProvider(BaseModel):
    id: str
    name: str
    label: str


model_provider_map = {
    "openai_like": ModelProvider(id="openai_like", name="OpenAILike", label="OpenAI-Compatible"),
    "dashscope": ModelProvider(id="dashscope", name="Dashscope", label="DashScope"),
}


llm_url_to_model_provider_id_map = {
    "https://dashscope.aliyuncs.com/compatible-mode/v1": "dashscope",
}


class ChunkStage(str, Enum):
    PLANNING = "planning"
    ACTING = "acting"
    RESPONSE = "response"


class TextChunk(BaseModel):
    delta: str = ""
    tool_calls: List[ChoiceDeltaToolCall] = []
    usage: Optional[CompletionUsage] = None
    stage: str = ""  # planning/acting/response
    trace_id: str = ""


class ReasoningChunk(TextChunk):
    reasoning_delta: str = ""


class ToolResultChunk(TextChunk):
    result: str | None
    error: str | None = None # Tool出现错误，不影响主Loop
    tool: ChoiceDeltaToolCall


class ErrorChunk(TextChunk):
    error_message: str = ""
    exception: str | None = None
    error_type: str = ""


ChatResponseGenerator = AsyncGenerator[TextChunk, None]
