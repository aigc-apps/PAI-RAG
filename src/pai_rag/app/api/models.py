from pydantic import BaseModel
from typing import List, Dict


class VectorDbConfig(BaseModel):
    faiss_path: str | None = None


class RagQuery(BaseModel):
    question: str
    temperature: float | None = 0.1
    chat_history: List[Dict[str, str]] | None = None
    session_id: str | None = None
    vector_db: VectorDbConfig | None = None
    stream: bool | None = False
    with_intent: bool | None = False


class RetrievalQuery(BaseModel):
    question: str
    vector_db: VectorDbConfig | None = None


class LlmResponse(BaseModel):
    answer: str
    session_id: str | None = None


class ContextDoc(BaseModel):
    text: str
    score: float
    metadata: Dict
    image_url: str | None = None


class RetrievalResponse(BaseModel):
    docs: List[ContextDoc]


class RagResponse(BaseModel):
    answer: str
    session_id: str | None = None
    docs: List[ContextDoc] | None = None
    new_query: str | None = None


class DataInput(BaseModel):
    file_path: str
    enable_qa_extraction: bool = False
    enable_raptor: bool = False
