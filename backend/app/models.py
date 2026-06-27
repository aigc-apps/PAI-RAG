from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON, BigInteger, Text


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"
    id: str = Field(primary_key=True, max_length=64)
    user_id: Optional[str] = Field(default=None, index=True, max_length=64)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    title: Optional[str] = Field(default=None, max_length=200)
    last_response_id: Optional[str] = Field(default=None, max_length=64)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class ConversationItem(SQLModel, table=True):
    __tablename__ = "conversation_items"
    id: str = Field(primary_key=True, max_length=64)
    conversation_id: str = Field(index=True, max_length=64)
    seq: int = Field(sa_column=Column(BigInteger))
    type: str = Field(max_length=32)        # message|reasoning|function_call|function_call_output
    role: Optional[str] = Field(default=None, max_length=16)
    content: dict = Field(default_factory=dict, sa_column=Column(JSON))
    response_id: Optional[str] = Field(default=None, index=True, max_length=64)
    user_id: Optional[str] = Field(default=None, index=True, max_length=64)
    created_at: datetime = Field(default_factory=_now)


class UserRow(SQLModel, table=True):
    __tablename__ = "users"
    id: str = Field(primary_key=True, max_length=64)
    display_name: Optional[str] = Field(default=None, max_length=200)
    created_at: datetime = Field(default_factory=_now)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class ResponseRow(SQLModel, table=True):
    __tablename__ = "responses"
    id: str = Field(primary_key=True, max_length=64)
    conversation_id: Optional[str] = Field(default=None, index=True, max_length=64)
    previous_response_id: Optional[str] = Field(default=None, index=True, max_length=64)
    model: str = Field(max_length=128)
    status: str = Field(max_length=32)
    usage: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    error: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class MemoryRow(SQLModel, table=True):
    __tablename__ = "memory_items"
    id: str = Field(primary_key=True, max_length=64)
    user_id: str = Field(index=True, max_length=64)
    text: str = Field(sa_column=Column("text", Text))
    kind: str = Field(default="fact", max_length=32)
    source_response_id: Optional[str] = Field(default=None, max_length=64)
    status: str = Field(default="active", max_length=16)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
