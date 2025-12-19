from datetime import datetime, timezone
import uuid
from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional

class EvaluatorConfigCreate(SQLModel):
    name: str = Field(default="")
    type: str = Field(default="") # ExactMatch, LLMJudge
    model_id: str = Field(default="")
    model_provider_name: Optional[str] = Field(default="openai_like")
    case_sensitive: bool = Field(default=False)
    ignore_punctuation: bool = Field(default=False)

class EvaluatorConfigEntity(EvaluatorConfigCreate, table=True):
    __tablename__ = "pai_evaluator_config"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)

    dataset_id: str = Field(
        foreign_key="pai_dataset.id",
        description="Reference to the evaluation task",
        ondelete="CASCADE",
    )

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
