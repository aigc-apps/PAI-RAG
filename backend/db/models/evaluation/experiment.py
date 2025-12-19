from datetime import datetime, timezone
import uuid
from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, Text
from typing import Optional, List
from common.system_constants import DEFAULT_TENANT_ID


class ExperimentCreate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    sample_ids: Optional[list[str]] = None  # List of dataset IDs to run the experiment on
    run_config_id: str = Field(
        foreign_key="pai_run_config.id",
        description="Reference to the run task",
        ondelete="CASCADE",
    )
    evaluator_config_id: str = Field(
        foreign_key="pai_evaluator_config.id",
        description="Reference to the evaluation task",
        ondelete="CASCADE",
    )


class ExperimentEntity(SQLModel, table=True):
    __tablename__ = "pai_experiment"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    dataset_id: str = Field(
        foreign_key="pai_dataset.id",
        description="Reference to the evaluation task",
        ondelete="CASCADE",
    )
    name: Optional[str] = Field(
        default=None,
        description="Name of the experiment"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the experiment",
        sa_column=Column(Text),
    )
    samples_count: Optional[int] = Field(
        default=0,
        description="Number of samples in the experiment"
    )
    run_config_id: str = Field(
        foreign_key="pai_run_config.id",
        description="Reference to the run task",
        ondelete="CASCADE",
    )
    evaluator_config_id: str = Field(
        foreign_key="pai_evaluator_config.id",
        description="Reference to the evaluation task",
        ondelete="CASCADE",
    )
    avg_score: Optional[float] = Field(
        default=None,
        description="Average score of the experiment"
    )
    status: str = Field(
        default="pending",
        description="Current status (pending, running, completed, failed)"
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

class ExperimentSampleEntity(SQLModel, table=True):
    __tablename__ = "pai_experiment_sample_entity"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    experiment_id: str = Field(
        foreign_key="pai_experiment.id",
        description="Reference to the experiment",
        ondelete="CASCADE",
    )
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)

    dataset_id: str = Field(
        foreign_key="pai_dataset.id",
        description="Reference to the dataset entry used",
        ondelete="CASCADE",
    )
    sample_id: str = Field(
        foreign_key="pai_dataset_sample.id",
        description="Reference to the dataset sample entry used",
        ondelete="CASCADE",
    )
    actual_output: Optional[str] = Field(
        default=None,
        description="The actual output from model during experiment",
        sa_column=Column(Text),
    )
    trace_id: Optional[str] = Field(
        default="",
        description="Trace ID for the request"
    )
    status: str = Field(
        default="pending",
        description="Execution status (pending, running, success, error)"
    )
    score: Optional[float] = Field(
        default=None,
        description="Score of the experiment run"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for the evaluation score (if applicable)",
        sa_column=Column(Text),
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if status is error",
        sa_column=Column(Text),
    )
    execution_metadata: Optional[List[dict]] = Field(
        default=[],
        sa_column=Column("execution_metadata", JSON),
        description="Additional execution metadata (function_call, observations, etc.)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime)
    )

    @field_serializer("created_at", "updated_at", "started_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
