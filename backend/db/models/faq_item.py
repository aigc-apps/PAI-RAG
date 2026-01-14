import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, Text
from datetime import datetime, timezone
from typing import Optional
from common.system_constants import DEFAULT_TENANT_ID


class FAQItemCreate(SQLModel):
    question: str = Field(default=None, sa_column=Column(Text))
    answer: str = Field(default=None, sa_column=Column(Text))
    chatbot_id: str = Field(default=None)
    file_id: Optional[str] = Field(default=None)
    active: bool = Field(default=True)


class FAQItemEntity(FAQItemCreate, table=True):
    __tablename__ = "pai_chatbot_faq"

    id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4().hex))
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, index=True)
    chatbot_id: str = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
