import uuid
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, JSON
from typing import List


class MessageCreate(SQLModel):
    thread_id: str = Field(default=None)
    role: str = Field(default=None)
    content: List[dict] = Field(default=[], sa_column=Column("content", JSON))
    attachments: List[dict] = Field(default=[], sa_column=Column("attachments", JSON))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )


class MessageRead(MessageCreate):
    id: str = Field(default=None, primary_key=True)


class MessageEntity(SQLModel, table=True):
    __tablename__ = "pai_message"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4()), primary_key=True)
    thread_id: str = Field(
        default=None, foreign_key="pai_thread.id", ondelete="CASCADE", nullable=False
    )

    role: str = Field(default=None)  # e.g., "user", "assistant", "system"
    content: List[dict] = Field(default=[], sa_column=Column("content", JSON))
    attachments: List[dict] = Field(default=[], sa_column=Column("attachments", JSON))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
