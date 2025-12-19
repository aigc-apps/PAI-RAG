from datetime import datetime, timezone
import uuid
from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, Text
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional

class DatasetCreate(SQLModel):
    name: str = Field(default=None)
    description: str = Field(default=None, sa_column=Column(Text))
    type: str = Field(default="") # "built-in" or "custom"

class DatasetEntity(DatasetCreate, table=True):
    __tablename__ = "pai_dataset"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

class DatasetSampleEntity(SQLModel, table=True):
    __tablename__ = "pai_dataset_sample"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    dataset_id: str = Field(
        foreign_key="pai_dataset.id",
        description="Reference to the evaluation task",
        ondelete="CASCADE",
    )
    input: str = Field(
        description="The user input/query for evaluation",
        sa_column=Column(Text),
    )
    expected_output: Optional[str] = Field(
        default=None,
        description="The expected/correct response for this input",
        sa_column=Column(Text),
    )
    eval_metadata: Optional[dict] = Field(default={}, sa_column=Column("eval_metadata", JSON))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
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
