import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime
import sqlalchemy as sa
from datetime import datetime, timezone
from typing import Optional
from common.system_constants import DEFAULT_TENANT_ID


class FAQConfigCreate(SQLModel):
    active: bool = Field(
        default=True
    )
    # FAQ configuration fields
    score_threshold: Optional[float] = Field(default=0.9, description="分数阈值，范围0.8-1.0")
    embedding_model: Optional[str] = Field(default="BAAI/bge-m3", description="Embedding模型ID")
    question_in_retrieval: Optional[bool] = Field(default=True, description="问题是否参与检索")
    question_in_response: Optional[bool] = Field(default=False, description="问题是否参与回答")
    answer_in_retrieval: Optional[bool] = Field(default=False, description="答案是否参与检索")
    answer_in_response: Optional[bool] = Field(default=True, description="答案是否参与回答")


class FAQConfigEntity(FAQConfigCreate, table=True):
    __tablename__ = "pai_chatbot_faq_config"

    id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4().hex))
    chatbot_id: str = Field(default=None, index=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, index=True)
    # FAQ configuration fields as direct columns
    score_threshold: Optional[float] = Field(default=0.9, sa_column=Column(sa.Float), description="分数阈值，范围0.8-1.0")
    embedding_model: Optional[str] = Field(default="BAAI/bge-m3", sa_column=Column(sa.String), description="Embedding模型ID")
    question_in_retrieval: Optional[bool] = Field(default=True, sa_column=Column(sa.Boolean), description="问题是否参与检索")
    question_in_response: Optional[bool] = Field(default=False, sa_column=Column(sa.Boolean), description="问题是否参与回答")
    answer_in_retrieval: Optional[bool] = Field(default=False, sa_column=Column(sa.Boolean), description="答案是否参与检索")
    answer_in_response: Optional[bool] = Field(default=True, sa_column=Column(sa.Boolean), description="答案是否参与回答")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
