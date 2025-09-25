from typing import List, Optional
from enum import Enum
from pydantic import BaseModel
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, CompletionUsage
from typing import AsyncGenerator


DEFAULT_TEMPERATURE = 0.1
DEFAULT_CONTEXT_WINDOW = 30000  # tokens
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"


class ChunkStage(str, Enum):
    PLANNING = "planning"
    ACTING = "acting"
    RESPONSE = "response"


class TextChunk(BaseModel):
    delta: str = ""
    tool_calls: List[ChoiceDeltaToolCall] = []
    usage: Optional[CompletionUsage] = None
    stage: str = ""  # planning/acting/response


class ReasoningChunk(TextChunk):
    reasoning_delta: str = ""


class ToolResultChunk(TextChunk):
    result: str
    tool: ChoiceDeltaToolCall


class ErrorChunk(TextChunk):
    error_message: str = ""
    exception: str | None = None
    error_type: str = ""


ChatResponseGenerator = AsyncGenerator[TextChunk, None]
