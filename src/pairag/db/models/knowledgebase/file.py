from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime


class KnowledgebaseFileEntity(SQLModel, table=True):
    __tablename__ = "pai_knowledgebase_file"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    knowledgebase: str = Field(default=None, foreign_key="pai_knowledgebase.name")

    file_name: str = Field(default=None, unique=True)
    file_path: str = Field(default=None)
    file_type: str = Field(default=None)
    file_size: int = Field(default=None)
    file_md5: str = Field(default=None)

    status: str = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime)
    )
    update_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime)
    )
