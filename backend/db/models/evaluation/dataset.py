from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from typing import Optional

class EvalDatasetSample(SQLModel):
    input: str = Field(
        description="The user input/query for evaluation"
    )
    expected_output: Optional[str] = Field(
        default=None,
        description="The expected/correct response for this input"
    )
    eval_metadata: Optional[dict] = Field(default={}, sa_column=Column("eval_metadata", JSON))

class EvaluationDatasetEntity(SQLModel, table=True):
    __tablename__ = "pai_evaluation_dataset"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    eval_id: str = Field(
        foreign_key="pai_evaluation.id",
        description="Reference to the evaluation task"
    )
    input: str = Field(
        description="The user input/query for evaluation"
    )
    expected_output: Optional[str] = Field(
        default=None,
        description="The expected/correct response for this input"
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
