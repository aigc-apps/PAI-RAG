from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from typing import Optional, List
from db.models.evaluation.evaluation import EvalRunConfig

class ExperimentCreate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    dataset_ids: Optional[list[str]] = None  # List of dataset IDs to run the experiment on
    run_config: dict = Field(
        default=lambda: EvalRunConfig(), sa_column=Column("run_config", JSON)
    )

class ExperimentEntity(SQLModel, table=True):
    __tablename__ = "pai_experiment"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    eval_id: str = Field(
        foreign_key="pai_evaluation.id",
        description="Reference to the evaluation task"
    )
    name: Optional[str] = Field(
        default=None,
        description="Name of the experiment"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the experiment"
    )
    samples_count: Optional[int] = Field(
        default=0,
        description="Number of samples in the experiment"
    )
    run_config: dict = Field(
        default=lambda: EvalRunConfig(), sa_column=Column("run_config", JSON)
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

class ExperimentRunResultEntity(SQLModel, table=True):
    __tablename__ = "pai_experiment_run_result"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    experiment_id: str = Field(
        foreign_key="pai_experiment.id",
        description="Reference to the experiment"
    )
    dataset_id: str = Field(
        foreign_key="pai_evaluation_dataset.id",
        description="Reference to the dataset entry used"
    )
    actual_output: Optional[str] = Field(
        default=None,
        description="The actual output from model during experiment"
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
        description="Reason for the evaluation score (if applicable)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if status is error"
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
