from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime

class EvaluationCreate(SQLModel):
    name: str = Field(default=None)
    description: str = Field(default=None)
    type: str = Field(default="") # "built-in" or "custom"


# table entity
class EvaluationEntity(EvaluationCreate, table=True):
    __tablename__ = "pai_evaluation"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
