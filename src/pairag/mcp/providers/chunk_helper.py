from typing import List
from loguru import logger
from sqlalchemy import delete
from sqlmodel import select, update
from pairag.db.models.change_event import ChangeEventSource, ChangeEventType
from pairag.db.models.knowledgebase.chunk import (
    KbChunkEntity,
    create_chunk_from_text_node,
)
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from llama_index.core.schema import TextNode
from pairag.common.knowledgebase.types import FileStatus, ChunkStatus
from pairag.db.models.knowledgebase.embedding import EmbeddingModelEntity
from pairag.db.models.knowledgebase.file import KbFileEntity
from pairag.mcp.providers.config_change_manager import config_change_manager


@with_async_db_session
async def set_embedding_model_ready(
    session: AsyncSession,
    model_id: str,
):
    embedding_model = await session.get(EmbeddingModelEntity, model_id)
    if embedding_model is None:
        raise ValueError(
            status_code=404,
            detail=f"Embedding model {model_id} not found.",
        )

    embedding_model.is_ready = True
    session.add(embedding_model)
    await session.commit()
    session.refresh(embedding_model)

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.EMBEDDING,
        source_id=model_id,
        event_type=ChangeEventType.UPDATE
    )


@with_async_db_session
async def read_file_from_db(
    session: AsyncSession,
    file_id: str,
) -> KbFileEntity:
    file_entity = await session.get(KbFileEntity, file_id)
    assert file_entity is not None, f"File {file_id} not found."
    return file_entity


@with_async_db_session
async def save_file_to_db(
    session: AsyncSession,
    file_entity: KbFileEntity,
):
    session.add(file_entity)
    await session.commit()
    logger.info(f"[FileHelper] Add file entity {file_entity}.")


@with_async_db_session
async def update_file_status_async(
    session: AsyncSession,
    file_id: str,
    status: FileStatus,
    failed_reason: str = None,
):
    file = await session.get(KbFileEntity, file_id)
    file.status = status
    file.failed_reason = failed_reason
    session.add(file)
    await session.commit()
    logger.info(f"[FileHelper] Updated file {file_id} status to {status}.")


@with_async_db_session
async def save_chunks_to_db_async(
    session: AsyncSession,
    kb_id: str,
    file_id: str,
    chunk_nodes: List[TextNode],
):
    logger.info(f"[KnowledgebaseProvider] Start saving {len(chunk_nodes)} chunks.")
    chunk_records: List[KbChunkEntity] = [
        create_chunk_from_text_node(kb_id, file_id, chunk) for chunk in chunk_nodes
    ]
    select_statement = select(KbChunkEntity).where(
        KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id
    )
    existing_chunks = (await session.exec(select_statement)).all()
    existing_chunk_ids = [chunk.id for chunk in existing_chunks]

    # 构造 DELETE 语句
    del_statement = delete(KbChunkEntity).where(
        KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id
    )

    # 执行删除操作
    await session.exec(del_statement)
    logger.info(
        f"[KnowledgebaseProvider] Deleted {len(existing_chunks)} chunks for file {file_id}."
    )

    session.add_all(chunk_records)
    await session.commit()

    new_chunk_ids = [record.id for record in chunk_records]
    logger.info(f"[FileHelper] saved {len(chunk_records)} chunks.")
    return existing_chunk_ids, new_chunk_ids


@with_async_db_session
async def update_chunk_status_async(
    session: AsyncSession,
    chunk_ids: List[str],
    status: ChunkStatus,
) -> None:
    logger.info(f"[FileHelper] updating chunk {chunk_ids} status to {status}.")
    await session.exec(
        update(KbChunkEntity)
        .where(KbChunkEntity.id.in_(chunk_ids))
        .values(status=status.value)
    )
    await session.commit()
    logger.info(
        f"[FileHelper] successfully updated chunk {chunk_ids} status to {status}."
    )
