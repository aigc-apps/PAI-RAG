from datetime import datetime, timezone
from typing import Optional
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from llama_index.core.schema import TextNode
from common.knowledgebase.types import ChunkStatus
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo


class KbChunkModel(SQLModel):
    text: str = Field(default=None)
    chunk_metadata: dict = Field(default={}, sa_column=Column("chunk_metadata", JSON))

    status: ChunkStatus = Field(default=ChunkStatus.pending)

    # active和status独立控制，status是处理状态，active是由用户控制是否显示在知识库
    # active 设为false，即从向量库删除对应chunk.
    # active 设为true，即添加对应chunk to vector store.
    active: bool = Field(
        default=True
    )


class KbChunkEntity(KbChunkModel, table=True):
    __tablename__ = "pai_knowledgebase_chunk"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    # ref
    file_id: str = Field(default=None, foreign_key="pai_knowledgebase_file.id")
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id")
    index: Optional[int] = Field(default=0) # chunk index

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )


def create_chunk_from_text_node(kb_id: str, file_id: str, node: TextNode, index: int):
    return KbChunkEntity(
        id=node.id_,
        knowledgebase_id=kb_id,
        file_id=file_id,
        kb_id=kb_id,
        text=node.text,
        chunk_metadata=node.metadata,
        index=index,
    )

def create_text_node_from_chunk(chunk: KbChunkEntity):
    return TextNode(
        id_ = chunk.id,
        text = chunk.text,
        metadata = chunk.chunk_metadata,
        relationships = {
            NodeRelationship.SOURCE:RelatedNodeInfo(
                node_id=chunk.file_id, metadata={}
            )
        }
    )
