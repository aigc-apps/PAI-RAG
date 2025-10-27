from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime

class EvaluatorConfigCreate(SQLModel):
    name: str = Field(default="")
    type: str = Field(default="") # ExactMatch, LLMJudge
    model_id: str = Field(default="")
    case_sensitive: bool = Field(default=False)
    ignore_punctuation: bool = Field(default=False)

class EvaluatorConfigEntity(EvaluatorConfigCreate, table=True):
    __tablename__ = "pai_evaluator_config"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
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
