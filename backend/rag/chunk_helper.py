from typing import List
from loguru import logger
from sqlalchemy import delete
from sqlmodel import select, update
from config.providers.embedding_provider import create_embedding_model
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.knowledgebase.chunk import (
    KbChunkEntity,
    create_chunk_from_text_node,
)
from db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from llama_index.core.schema import TextNode
from common.knowledgebase.types import FileStatus, ChunkStatus
from db.models.knowledgebase.embedding import EmbeddingModelEntity
from db.models.knowledgebase.file import KbFileEntity
from llama_index.core.schema import Document
from config.providers.config_change_manager import config_change_manager


MAX_CACHE_SIZE = 3
embed_cache_dict = {}

@with_async_db_session
async def get_embedding_from_db(
    session: AsyncSession, model_id: str
) -> EmbeddingModelEntity:
    if model_id in embed_cache_dict:
        return embed_cache_dict[model_id]

    embedding_entity = (await session.exec(
        select(EmbeddingModelEntity).where(EmbeddingModelEntity.model_id == model_id)
    )).first()

    if not embedding_entity:
        raise ValueError(f"Embedding model {model_id} not found.")

    if not embedding_entity.is_ready:
        raise ValueError(f"Embedding model {model_id} is not downloaded, please check the download status.")

    embed_model = create_embedding_model(config=embedding_entity)
    if model_id not in embed_cache_dict and len(embed_cache_dict) >= MAX_CACHE_SIZE:
        first_key = next(iter(embed_cache_dict))
        embed_cache_dict.pop(first_key)

    embed_cache_dict[model_id] = embed_model
    return embed_cache_dict[model_id]


@with_async_db_session
async def set_embedding_model_ready(
    session: AsyncSession,
    id: str,
):
    embedding_model = await session.get(EmbeddingModelEntity, id)
    if embedding_model is None:
        raise ValueError(
            f"Embedding model {id} not found."
        )

    embedding_model.is_ready = True
    session.add(embedding_model)
    await session.commit()
    await session.refresh(embedding_model)

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.EMBEDDING,
        source_id=id,
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
    is_attachment: bool = False,
    documents: List[Document] = None,
):
    file = await session.get(KbFileEntity, file_id)
    if is_attachment and documents:
        if file.file_extension in [".xlsx"]:
            file.file_content = "\n".join([doc.text for doc in documents])
            file.file_content_length = len(file.file_content)
        else:
            file.file_content = documents[0].text
            file.file_content_length = len(documents[0].text)

    file.status = status
    file.failed_reason = failed_reason
    session.add(file)
    await session.commit()
    await session.flush()
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
        create_chunk_from_text_node(kb_id=kb_id, file_id=file_id, node=chunk, index=i) for (i, chunk) in enumerate(chunk_nodes)
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


@with_async_db_session
async def get_kb_chunk_ids(
    session: AsyncSession,
    kb_id: str,
    file_id: str,
) -> List[str]:
    chunk_ids = (await session.exec(
        select(KbChunkEntity.id).where(
            KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id
        )
    )).all()

    logger.info(f"[FileHelper] get {len(chunk_ids)} chunk ids for kb {kb_id} and file {file_id}")
    return chunk_ids


@with_async_db_session
async def get_file_id_source_map(
    session: AsyncSession,
    kb_id: str,
    file_ids: List[str],
):
    file_source_results = (await session.exec(
        select( KbFileEntity.id, KbFileEntity.file_source ).where(
            KbFileEntity.kb_id == kb_id,
            KbFileEntity.id.in_(file_ids),
        )
    )).all()
    logger.info(f"Get file_source_results: {file_source_results}")
    return {
        file_id: file_source
        for file_id, file_source in file_source_results
    }
