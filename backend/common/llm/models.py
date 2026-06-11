from typing import List, Optional
from enum import Enum
from pydantic import BaseModel
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, CompletionUsage
from typing import AsyncGenerator


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
