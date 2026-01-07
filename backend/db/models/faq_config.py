from sqlmodel import Field, SQLModel
from typing import Optional


class FAQConfigCreate(SQLModel):
    active: bool = Field(
        default=True
    )
    # FAQ configuration fields
    similarity_threshold: Optional[float] = Field(default=0.9, description="相似度阈值，范围0.8-1.0")
    embedding_model: Optional[str] = Field(default="BAAI/bge-m3", description="Embedding模型ID")
    question_in_retrieval: Optional[bool] = Field(default=True, description="问题是否参与检索")
    question_in_response: Optional[bool] = Field(default=False, description="问题是否参与回答")
    answer_in_retrieval: Optional[bool] = Field(default=False, description="答案是否参与检索")
    answer_in_response: Optional[bool] = Field(default=True, description="答案是否参与回答")
