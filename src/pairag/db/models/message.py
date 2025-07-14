import uuid
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime


class MessageCreate(SQLModel):
    thread_id: str = Field(default=None)
    role: str = Field(default=None)
    content: str = Field(default=None)
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
    content: str = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
