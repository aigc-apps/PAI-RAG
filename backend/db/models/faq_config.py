from sqlmodel import Field, SQLModel
from typing import Optional
from common.knowledgebase.constants import DEFAULT_FAQ_SIMILARITY_THRESHOLD


class FAQConfigCreate(SQLModel):
    active: bool = Field(
        default=True
    )
    # FAQ configuration fields
    kb_id: Optional[str] = Field(default=None, description="FAQ知识库ID")
    similarity_threshold: Optional[float] = Field(default=DEFAULT_FAQ_SIMILARITY_THRESHOLD, description="相似度阈值，范围0.8-1.0")
    embedding_model: Optional[str] = Field(default="BAAI/bge-m3", description="Embedding模型ID")
    enable_question_in_retrieval: Optional[bool] = Field(default=True, description="问题是否参与检索")
    enable_question_in_response: Optional[bool] = Field(default=True, description="问题是否参与回答")
    enable_answer_in_retrieval: Optional[bool] = Field(default=False, description="答案是否参与检索")
    enable_answer_in_response: Optional[bool] = Field(default=True, description="答案是否参与回答")
    return_direct: Optional[bool] = Field(default=False, description="是否直接返回工具结果，不经过LLM加工")
