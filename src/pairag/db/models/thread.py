import uuid
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime


class ThreadCreate(SQLModel):
    user_id: str = Field(default="default_user")
    title: str = Field(default=None)


class ThreadRead(ThreadCreate):
    id: str = Field(default=None, primary_key=True)
    archived: bool = Field(default=False)


class ThreadEntity(SQLModel, table=True):
    __tablename__ = "pai_thread"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(default="default_user", nullable=False)
    title: str = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
