from pydantic import BaseModel
from typing import List, Dict


class RagQuery(BaseModel):
    question: str
    topk: int | None = 3
    topp: float | None = 0.8
    temperature: float | None = 0.7
    vector_topk: int | None = 3
    score_threshold: float | None = 0.5
    chat_history: List[Dict[str, str]] | None = None


class LlmQuery(BaseModel):
    question: str
    topk: int | None = 3
    topp: float | None = 0.8
    temperature: float | None = 0.7
    chat_history: List[Dict[str, str]] | None = None


class RetrievalQuery(BaseModel):
    question: str
    topk: int | None = 3
    score_threshold: float | None = 0.5


class RagResponse(BaseModel):
    answer: str
    # TODO
    # context: List[str] | None = None


class LlmResponse(BaseModel):
    answer: str

class ContextDoc(BaseModel):
    text: str
    score: float
    metadata: Dict

class RetrievalResponse(BaseModel):
    docs: List[ContextDoc]


class KnowledgeInput(BaseModel):
    file_path: str
    enable_qa_extraction: bool = False
