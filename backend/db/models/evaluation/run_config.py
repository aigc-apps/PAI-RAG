from datetime import datetime, timezone
import uuid
from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from typing import List, Optional
from common.system_constants import DEFAULT_TENANT_ID


class RunConfigCreate(SQLModel):
    name: Optional[str] = Field(
        default=None,
        description="Name of the run config"
    )
    model_id: str = Field(default="")
    mcp_ids: List[str] = Field(default=[])
    kb_ids: List[str] = Field(default=[])
    enable_search: bool = Field(default=True)
    enable_vision: bool = Field(default=True)
    enable_agent: bool = Field(default=False)
    enable_input_guardrail: Optional[bool] = Field(default=False)
    enable_output_guardrail: Optional[bool] = Field(default=False)
    guardrail_hint: Optional[str] = Field(default=None)
    prompts: Optional[dict] = Field(default={})


class RunConfigEntity(RunConfigCreate, table=True):
    __tablename__ = "pai_run_config"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    dataset_id: str = Field(
        foreign_key="pai_dataset.id",
        description="Reference to the evaluation task",
        ondelete="CASCADE",
    )
    mcp_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    kb_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    prompts: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime)
    )

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
