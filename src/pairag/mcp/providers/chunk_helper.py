from typing import List
from loguru import logger
from sqlmodel import update
from pairag.db.models.knowledgebase.chunk import (
    KbChunkEntity,
    create_chunk_from_text_node,
)
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from llama_index.core.schema import TextNode
from pairag.common.knowledgebase.types import FileStatus, ChunkStatus
from pairag.db.models.knowledgebase.file import KbFileEntity


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

    session.add_all(chunk_records)
    await session.commit()

    chunk_ids = [record.id for record in chunk_records]
    logger.info(f"[FileHelper] saved {len(chunk_records)} chunks.")
    return chunk_ids


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
