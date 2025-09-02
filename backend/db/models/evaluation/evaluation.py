from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from typing import List, Optional

class EvalChatBotConfig(SQLModel):
    model_id: str = Field(default="")
    mcp_ids: List[str] = Field(default=[])
    kb_ids: List[str] = Field(default=[])
    enable_search: bool = Field(default=True)
    enable_vision: bool = Field(default=True)
    enable_agent: bool = Field(default=False)
    enable_input_guardrail: Optional[bool] = Field(default=False)
    enable_output_guardrail: Optional[bool] = Field(default=False)
    guardrail_hint: Optional[str] = Field(default=None)

class EvalCreate(SQLModel):
    name: str = Field(default=None)
    description: str = Field(default=None)
    chatbot_id: str = Field(default="")
    chatbot_config: dict = Field(
        default=lambda: EvalChatBotConfig(), sa_column=Column("chatbot_config", JSON)
    )


# table entity
class EvalEntity(SQLModel, table=True):
    __tablename__ = "pai_evaluation"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    name: str = Field(default=None, unique=True)
    description: str = Field(default=None)

    chatbot_id: str = Field(default=None)

    chatbot_config: dict = Field(
        default=lambda: EvalChatBotConfig(), sa_column=Column("chatbot_config", JSON)
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
