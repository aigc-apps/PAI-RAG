from datetime import datetime, timezone
import uuid
from sqlmodel import Column, DateTime, Field, Index, SQLModel
from enum import Enum


class ChangeEventType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


class ChangeEventSource(str, Enum):
    EMBEDDING = "embedding"
    LLM = "llm"
    RERANK = "rerank"
    KNOWLEDGEBASE = "knowledgebase"
    MCP = "mcp"
    TRACE = "trace"
    WEBSEARCH = "websearch"
    CHATBOT = "chatbot"
    PROMPT = "prompt"
    GUARDRAIL = "guardrail"
    EVALUATION = "evaluation"
    VECTORDB = "vectordb"



class ChangeEvent(SQLModel, table=True):
    __tablename__ = "pai_config_change_event"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    source_id: str = Field(default=None)
    event_type: str = Field(default=ChangeEventType.ADD)
    event_source: str = Field(default=ChangeEventSource.KNOWLEDGEBASE)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )


    __table_args__ = (
        Index('idx_change_event_created_at', 'created_at'),
    )
