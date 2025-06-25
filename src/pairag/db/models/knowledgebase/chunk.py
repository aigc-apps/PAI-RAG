from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime


class KnowledgebaseChunkEntity(SQLModel, table=True):
    __tablename__ = "pai_knowledgebase_chunk"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    # ref
    file_id: str = Field(default=None, foreign_key="pai_knowledgebase_file.id")
    knowledgebase: str = Field(default=None, foreign_key="pai_knowledgebase.name")

    text: str = Field(default=None)
    chunk_metadata: dict = Field(
        default=lambda: {}, sa_column=Column("chunk_metadata", JSON)
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime)
    )
