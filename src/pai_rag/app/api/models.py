from pydantic import BaseModel
from typing import List, Dict


class RagQuery(BaseModel):
    question: str
    temperature: float | None = 0.1
    vector_topk: int | None = 3
    score_threshold: float | None = 0.5
    chat_history: List[Dict[str, str]] | None = None
    stream: bool | None = False
    session_id: str | None = None


class LlmQuery(BaseModel):
    question: str
    temperature: float | None = 0.1
    chat_history: List[Dict[str, str]] | None = None
    stream: bool | None = False
    session_id: str | None = None


class RetrievalQuery(BaseModel):
    question: str
    topk: int | None = 3
    score_threshold: float | None = 0.5


class RagResponse(BaseModel):
    answer: str
    session_id: str | None = None
    # TODO
    # context: List[str] | None = None


class LlmResponse(BaseModel):
    answer: str
    session_id: str | None = None


class ContextDoc(BaseModel):
    text: str
    score: float
    metadata: Dict


class RetrievalResponse(BaseModel):
    docs: List[ContextDoc]


class DataInput(BaseModel):
    file_path: str
    enable_qa_extraction: bool = False
