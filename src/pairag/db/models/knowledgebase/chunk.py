from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from llama_index.core.schema import TextNode
from pairag.common.knowledgebase.types import ChunkStatus


class KbChunkEntity(SQLModel, table=True):
    __tablename__ = "pai_knowledgebase_chunk"
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
    # ref
    file_id: str = Field(default=None, foreign_key="pai_knowledgebase_file.id")
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id")

    text: str = Field(default=None)
    chunk_metadata: dict = Field(default={}, sa_column=Column("chunk_metadata", JSON))

    status: ChunkStatus = Field(default=ChunkStatus.pending)
    active: bool = Field(
        default=True
    )  # active和status独立控制，status是处理状态，active是由用户控制是否显示在知识库

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime)
    )


def create_chunk_from_text_node(kb_id: str, file_id: str, node: TextNode):
    return KbChunkEntity(
        id=node.id_,
        knowledgebase_id=kb_id,
        file_id=file_id,
        kb_id=kb_id,
        text=node.text,
        chunk_metadata=node.metadata,
    )
