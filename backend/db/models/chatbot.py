import re
import uuid
from sqlmodel import Field, SQLModel
from pydantic import field_validator, field_serializer
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, JSON, Text
from typing import List, Optional
from common.system_constants import DEFAULT_TENANT_ID


class ChatBotCreate(SQLModel):
    app_id: str = Field(default=None)
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    model_id: str = Field(default=None)
    vision_model_id: Optional[str] = Field(default=None)
    mcp_ids: List[str] = Field(default=[])
    kb_ids: List[str] = Field(default=[])
    enable_search: bool = Field(default=False)
    enable_chatdb: Optional[bool] = Field(default=False)
    enable_vision: bool = Field(default=True)
    enable_agent: bool = Field(default=False)
    enable_input_guardrail: Optional[bool] = Field(default=False)
    enable_output_guardrail: Optional[bool] = Field(default=False)
    guardrail_hint: Optional[str] = Field(default=None, sa_column=Column(Text))
    enable_faq: Optional[bool] = Field(default=False)
    enable_auto_metadata_filter: Optional[bool] = Field(default=False)
    faq_config: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    prompts: Optional[dict] = Field(default={})


    @field_validator("app_id")
    def validate_app_id(cls, v):
        if not v:
            raise ValueError("app_id is required")
        if not re.match("^[a-zA-Z][a-zA-Z0-9][\-_a-zA-Z0-9]{1,62}$", v):
            raise ValueError("app_id must be a valid name with length between 3 and 64")
        return v

    @field_validator("model_id")
    def validate_model_id(cls, v):
        if not v:
            raise ValueError("model_id is required")
        return v


class ChatBotEntity(ChatBotCreate, table=True):
    __tablename__ = "pai_chatbot_model"

    id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4().hex))
    mcp_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    kb_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    faq_config: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    prompts: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
