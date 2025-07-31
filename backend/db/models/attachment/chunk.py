from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from llama_index.core.schema import TextNode
from common.knowledgebase.types import ChunkStatus
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo


class AttachmentChunkModel(SQLModel):
    text: str = Field(default=None)
    chunk_metadata: dict = Field(default={}, sa_column=Column("chunk_metadata", JSON))

    status: ChunkStatus = Field(default=ChunkStatus.pending)

    # active和status独立控制，status是处理状态，active是由用户控制是否显示在知识库
    # active 设为false，即从向量库删除对应chunk.
    # active 设为true，即添加对应chunk to vector store.
    active: bool = Field(
        default=True
    )

class AttachmentChunkEntity(AttachmentChunkModel, table=True):
    __tablename__ = "pai_attachment_chunk"
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
    # ref
    file_id: str = Field(default=None, foreign_key="pai_attachment_file.id", ondelete="CASCADE")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )



def create_attachment_chunk_from_text_node(file_id: str, node: TextNode):
    return AttachmentChunkEntity(
        id=node.id_,
        file_id=file_id,
        text=node.text,
        chunk_metadata=node.metadata,
    )

def create_text_node_from_attachment_chunk(chunk: AttachmentChunkEntity):
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
