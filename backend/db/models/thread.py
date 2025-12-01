import uuid
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Text


class ThreadCreate(SQLModel):
    user_id: str = Field(default="PAI-RAG Assistant")
    title: str = Field(default=None, sa_column=Column(Text))


class ThreadRead(ThreadCreate):
    id: str = Field(default=None, primary_key=True)
    archived: bool = Field(default=False)


class ThreadEntity(SQLModel, table=True):
    __tablename__ = "pai_thread"

    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
    user_id: str = Field(default="PAI-RAG Assistant", nullable=False)
    title: str = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
