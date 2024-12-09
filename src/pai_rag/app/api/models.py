from pydantic import BaseModel
from typing import List, Dict


# To Do: remove vector db config
class VectorDbConfig(BaseModel):
    faiss_path: str | None = None


class RagQuery(BaseModel):
    question: str
    temperature: float | None = 0.1
    chat_history: List[Dict[str, str]] | None = None
    session_id: str | None = None
    vector_db: VectorDbConfig | None = None
    stream: bool | None = False
    citation: bool | None = False
    with_intent: bool | None = False
    index_name: str | None = None


class RetrievalQuery(BaseModel):
    question: str
    index_name: str | None = None
    vector_db: VectorDbConfig | None = None


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
