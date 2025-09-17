from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from typing import List, Optional

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

class RunConfigEntity(RunConfigCreate, table=True):
    __tablename__ = "pai_run_config"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    dataset_id: str = Field(
        foreign_key="pai_dataset.id",
        description="Reference to the evaluation task"
    )
    mcp_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    kb_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime)
    )
