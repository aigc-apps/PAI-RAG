from __future__ import annotations
from typing import List, Literal, Optional, Union
from pydantic import BaseModel


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class RunStarted(BaseModel):
    type: Literal["run.started"] = "run.started"
    response_id: str
    conversation_id: Optional[str] = None


class TextDelta(BaseModel):
    type: Literal["text.delta"] = "text.delta"
    text: str


class ReasoningDelta(BaseModel):
    type: Literal["reasoning.delta"] = "reasoning.delta"
    text: str


class ToolStarted(BaseModel):
    type: Literal["tool.started"] = "tool.started"
    call_id: str
    name: str


class ToolArgumentsDelta(BaseModel):
    type: Literal["tool.arguments.delta"] = "tool.arguments.delta"
    call_id: str
    name: str
    delta: str


class ToolCompleted(BaseModel):
    type: Literal["tool.completed"] = "tool.completed"
    call_id: str
    name: str
    arguments: str  # raw JSON string


class ToolResult(BaseModel):
    type: Literal["tool.result"] = "tool.result"
    call_id: str
    name: str
    ok: bool
    output: Optional[str] = None
    error: Optional[str] = None
    # Structured file artifacts, kept separate from `output` (the LLM-facing str).
    files: Optional[List[dict]] = None


class RunCompleted(BaseModel):
    type: Literal["run.completed"] = "run.completed"
    usage: Usage = Usage()
    finish_reason: str = "stop"


class RunFailed(BaseModel):
    type: Literal["run.failed"] = "run.failed"
    message: str
    error_type: str = "error"


AgentEvent = Union[
    RunStarted, TextDelta, ReasoningDelta, ToolStarted, ToolArgumentsDelta,
    ToolCompleted, ToolResult, RunCompleted, RunFailed,
]
