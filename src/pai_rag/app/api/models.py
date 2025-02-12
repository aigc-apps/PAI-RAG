from pydantic import BaseModel
from typing import List, Dict, Optional
from llama_index.core.base.llms.types import ChatMessage


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
    search_web: bool | None = False
    prompt_template: str | None = None
    return_reference: bool | None = False


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


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False
    index_name: Optional[str] = None
    search_web: Optional[bool] = False
    nl2sql: Optional[bool] = False
    citation: Optional[bool] = False
    prompt_template: Optional[str] = None
