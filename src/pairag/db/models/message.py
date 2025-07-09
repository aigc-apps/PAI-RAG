import uuid
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime


class MessageEntity(SQLModel, table=True):
    __tablename__ = "pai_message"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4()), primary_key=True)
    thread_id: str = Field(default=None, foreign_key="pai_thread.id")
    thread_index: int = Field(default=None)  # Index of the message in the thread

    role: str = Field(default=None)  # e.g., "user", "assistant", "system"
    content: str = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
