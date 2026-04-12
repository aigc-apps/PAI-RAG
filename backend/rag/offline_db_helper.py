from datetime import datetime, timezone
from typing import List
from db.models.knowledgebase.file_task import KbFileTaskEntity
from db.models.llm import LlmModelEntity
from loguru import logger
from sqlalchemy import delete, or_, func
from sqlmodel import select, update
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
from pairag.file.store.file_store_helper import file_store
from common.llm.openai.openai_like import OpenAILike
from service.factory.model_factory import create_embedding_model, create_openailike_llm
from service.factory.vectordb_factory import create_vector_store
from llama_index.core.vector_stores.types import BasePydanticVectorStore
import pandas as pd
from llama_index.core.embeddings import BaseEmbedding
from db.models.knowledgebase.knowledgebase import KbEntity
from service.knowledgebase.vectordb_service import VectordbService
from service.injection import get_vectordb_service, get_vector_table_mapping_service, get_llm_service, get_embedding_service

DEFAULT_ATTACHMENT_MAX_SIZE = 1000
MAX_CACHE_SIZE = 3


@with_async_db_session
async def create_vector_store_from_db(
    session: AsyncSession,
    kb_id: str,
    dimension: int,
    tenant_id: str,
) -> BasePydanticVectorStore:
    vectordb_service: VectordbService = await get_vectordb_service(session=session)
    vectordb_config = await vectordb_service.get_vectordb_config(tenant_id=tenant_id)
    if not vectordb_config:
        raise ValueError(f"VectorDB config not found for knowledgebase {kb_id}.")
    table_mapping_service = await get_vector_table_mapping_service(session=session)
    table_name = await table_mapping_service.get_vector_table_name(tenant_id=tenant_id, kb_id=kb_id)
    if not table_name:
        raise ValueError(f"Vector table name not found for knowledgebase {kb_id}.")
    vector_store = create_vector_store(
        kb_id, dimension, vector_config=vectordb_config, table_name=table_name,
    )
    return vector_store



@with_async_db_session
async def get_knowledgebase_from_db(
    session: AsyncSession,
    kb_id: str,
    tenant_id: str,
) -> KbEntity:
    select_statement = select(KbEntity).where(KbEntity.id == kb_id, KbEntity.tenant_id == tenant_id)
    knowledgebase = (await session.exec(select_statement)).first()
    if not knowledgebase:
        logger.error(f"[FileHelper] Knowledgebase {kb_id} not found, skip getting knowledgebase.")
        raise ValueError(f"Knowledgebase {kb_id} not found.")
    return knowledgebase


@with_async_db_session
async def get_embedding_from_db(
    session: AsyncSession, model_id: str, provider_name: str, tenant_id: str,
) -> BaseEmbedding:
    embedding_service = await get_embedding_service(session=session)

    embedding_entity = await embedding_service.get_embedding_model_by_provider_model_id(provider_name=provider_name, model_id=model_id, tenant_id=tenant_id)
    if not embedding_entity:
        raise ValueError(f"Embedding model {model_id} not found.")

    return create_embedding_model(config=embedding_entity)


@with_async_db_session
async def set_embedding_model_ready(
    session: AsyncSession,
    id: str,
    tenant_id: str,
):
    try:
        select_statement = select(EmbeddingModelEntity).where(EmbeddingModelEntity.id == id, EmbeddingModelEntity.tenant_id == tenant_id)
        embedding_model = (await session.exec(select_statement)).first()
        if not embedding_model:
            logger.error(f"[FileHelper] Embedding model {id} not found, skip setting embedding model ready.")
            raise ValueError(f"Embedding model {id} not found.")
        if embedding_model is None:
            raise ValueError(
                f"Embedding model {id} not found."
            )

        embedding_model.is_ready = True
        session.add(embedding_model)
        await session.commit()
        await session.refresh(embedding_model)
    except Exception as e:
        logger.error(f"Error setting embedding model {id} ready: {e}")
        await session.rollback()
        raise e


@with_async_db_session
async def get_llm_model_from_db(
    session: AsyncSession,
    model_id: str,
    tenant_id: str,
    provider_name: str,
) -> LlmModelEntity:
    llm_service = await get_llm_service(session=session)
    llm_entity = await llm_service.get_llm_model_by_provider_model_id(
        provider_name=provider_name,
        model_id=model_id,
        tenant_id=tenant_id,
    )
    return llm_entity

@with_async_db_session
async def get_openailike_llm_from_db(
    session: AsyncSession,
    model_id: str,
    tenant_id: str,
    provider_name: str,
) -> OpenAILike:
    llm_service = await get_llm_service(session=session)

    llm_entity = await llm_service.get_llm_model_by_provider_model_id(
        provider_name=provider_name,
        model_id=model_id,
        tenant_id=tenant_id,
    )
    if not llm_entity:
        raise ValueError(f"LLM model {model_id} not found.")
    return create_openailike_llm(config=llm_entity)


@with_async_db_session
async def should_cancel_file_task(
    session: AsyncSession,
    kb_id: str,
    file_id: str,
    tenant_id: str,
    file_part: int = 0,
    file_version: int = 0,
) -> bool:
    select_statement = select(KbFileEntity).where(KbFileEntity.id == file_id, KbFileEntity.tenant_id == tenant_id)
    file_entity = (await session.exec(select_statement)).first()
    if not file_entity:
        logger.warning(f"File entity {file_id} not found, cancel the task.")
        return True

    # If a newer file version uploaded, cancel the task.
    if file_entity.file_version != file_version:
        logger.warning(f"File entity {file_id} version changed from {file_version} to {file_entity.file_version}, cancel the task.")
        return True

    # If file status is cancelled or failed, cancel the task.
    if file_entity.status in [FileStatus.cancelled, FileStatus.failed]:
        logger.warning(f"File entity {file_id} status is {file_entity.status}, cancel the task.")
        return True


    file_task = (await session.exec(
        select(KbFileTaskEntity).where(
            KbFileTaskEntity.kb_id == kb_id,
            KbFileTaskEntity.file_id == file_id,
            KbFileTaskEntity.file_part == file_part,
            KbFileTaskEntity.tenant_id == tenant_id,
        )
    )).first()

    if not file_task:
        logger.warning(f"File task for file {file_id} part {file_part} not found, cancel the task.")
        return True

    if file_task.status == FileStatus.cancelled:
        logger.warning(f"File task for file {file_id} part {file_part} is cancelled, cancel the task.")
        return True

    if file_task.file_version != file_version:
        logger.warning(f"File task for file {file_id} part {file_part} is outdated, cancel the task.")
        return True

    return False


@with_async_db_session
async def read_file_from_db(
    session: AsyncSession,
    file_id: str,
    tenant_id: str,
) -> KbFileEntity:
    select_statement = select(KbFileEntity).where(KbFileEntity.id == file_id, KbFileEntity.tenant_id == tenant_id)
    file_entity = (await session.exec(select_statement)).first()
    if not file_entity:
        logger.error(f"[FileHelper] File {file_id} not found, skip reading file.")
        raise ValueError(f"File {file_id} not found.")
    return file_entity


@with_async_db_session
async def save_file_to_db(
    session: AsyncSession,
    file_entity: KbFileEntity,
):
    try:
        session.add(file_entity)
        await session.commit()
        logger.info(f"[FileHelper] Add file entity {file_entity}.")
    except Exception as e:
        logger.error(f"[FileHelper] Error adding file entity {file_entity}: {e}")
        await session.rollback()



@with_async_db_session
async def update_file_content_async(
    session: AsyncSession,
    file_id: str,
    tenant_id: str,
    is_attachment: bool = False,
    documents: List[Document] = None,
):
    select_statement = select(KbFileEntity).where(KbFileEntity.id == file_id, KbFileEntity.tenant_id == tenant_id)
    file = (await session.exec(select_statement)).first()
    if not file:
        logger.error(f"[FileHelper] File {file_id} not found, skip updating file content.")
        raise ValueError(f"File {file_id} not found.")

    if is_attachment:
        if file.file_extension in [".xlsx", ".xls"]:
            file_data = await file_store.read_async(file_path=file.file_path, tenant_id=tenant_id)
            df = pd.read_excel(file_data)
            file.file_content = df.head(20).to_csv(index=False)
            if len(file.file_content) > DEFAULT_ATTACHMENT_MAX_SIZE or len(df) > 20:
                file.file_content = file.file_content[0:DEFAULT_ATTACHMENT_MAX_SIZE] + " \n\n [truncated] The content is too long, has been truncated."
            file.file_content_length = len(file.file_content)
        elif file.file_extension in [".csv"]:
            file_data = await file_store.read_async(file_path=file.file_path, tenant_id=tenant_id)
            df = pd.read_csv(file_data)
            file.file_content = df.head(10).to_csv(index=False)
            if len(file.file_content) > DEFAULT_ATTACHMENT_MAX_SIZE or len(df) > 10:
                file.file_content = file.file_content[0:DEFAULT_ATTACHMENT_MAX_SIZE] + " \n\n [truncated] The content is too long, has been truncated."
            file.file_content_length = len(file.file_content)
        elif documents:
            file.file_content = documents[0].text
            if len(file.file_content) > DEFAULT_ATTACHMENT_MAX_SIZE:
                file.file_content = file.file_content[0:DEFAULT_ATTACHMENT_MAX_SIZE] + " \n\n [truncated] The content is too long, has been truncated."
            file.file_content_length = len(file.file_content)
        else:
            file.file_content = ""
            file.file_content_length = 0
    try:
        session.add(file)
        await session.commit()
        await session.flush()
        logger.info(f"[FileHelper] Updated file {file_id} content.")
    except Exception as e:
        logger.error(f"[FileHelper] Error updating file {file_id} content: {e}")
        await session.rollback()



@with_async_db_session
async def update_file_status_async(
    session: AsyncSession,
    file_id: str,
    tenant_id: str,
    status: FileStatus,
    task_id: str = None,
    failed_reason: str = None,
    is_attachment: bool = False,
):
    select_statement = select(KbFileEntity).where(KbFileEntity.id == file_id, KbFileEntity.tenant_id == tenant_id)
    file = (await session.exec(select_statement)).first()
    if not file:
        logger.error(f"[FileHelper] File {file_id} not found, skip updating file status.")
        raise ValueError(f"File {file_id} not found.")

    # try update task status first, if there are still other tasks not in the same status, skip updating file status.
    if task_id:
        select_statement = select(KbFileTaskEntity).where(KbFileTaskEntity.id == task_id, KbFileTaskEntity.tenant_id == tenant_id)
        task = (await session.exec(select_statement)).first()
        if task:
            task.status = status
            task.failed_reason = failed_reason
            task.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(task)
            logger.info(f"[FileHelper] Updated file task {task_id} status to {status}.")

            if status != FileStatus.failed and status != FileStatus.cancelled:
                # 构造 COUNT 查询
                count_statement = select(func.count()).select_from(KbFileTaskEntity).where(
                    KbFileTaskEntity.file_id == file_id,
                    KbFileTaskEntity.kb_id == file.kb_id,
                    KbFileTaskEntity.status != status,
                    KbFileTaskEntity.id != task_id,
                    KbFileTaskEntity.tenant_id == tenant_id,
                )

                # 执行并获取标量结果
                status_count = (await session.exec(count_statement)).one()
                if status_count > 0:
                    logger.info(f"[FileHelper] There are still {status_count} file tasks not reaching status {status}, skip updating file {file_id} status to {status}.")
                    await session.commit()
                    await session.flush()
                    return

    file.status = status
    file.failed_reason = failed_reason
    try:
        session.add(file)
        await session.commit()
        await session.flush()
        logger.info(f"[FileHelper] Updated file {file_id} status to {status}.")
    except Exception as e:
        logger.error(f"[FileHelper] Error updating file {file_id} status to {status}: {e}")
        await session.rollback()


@with_async_db_session
async def save_chunks_to_db_async(
    session: AsyncSession,
    kb_id: str,
    file_id: str,
    file_part: int,
    chunk_nodes: List[TextNode],
    tenant_id: str,
):
    try:
        logger.info(f"[KnowledgebaseProvider] Start saving {len(chunk_nodes)} chunks.")
        chunk_records: List[KbChunkEntity] = [
            create_chunk_from_text_node(kb_id=kb_id, file_id=file_id, file_part=file_part,node=chunk, index=i, tenant_id=tenant_id) for (i, chunk) in enumerate(chunk_nodes)
        ]
        select_statement = select(KbChunkEntity).where(
            KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id, or_(KbChunkEntity.file_part.is_(None), KbChunkEntity.file_part == file_part), KbChunkEntity.tenant_id == tenant_id
        )
        existing_chunks = (await session.exec(select_statement)).all()
        existing_chunk_ids = [chunk.id for chunk in existing_chunks]

        # 构造 DELETE 语句
        del_statement = delete(KbChunkEntity).where(
            KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id, or_(KbChunkEntity.file_part.is_(None), KbChunkEntity.file_part == file_part), KbChunkEntity.tenant_id == tenant_id
        )

        # 执行删除操作
        await session.exec(del_statement)
        logger.info(
            f"[KnowledgebaseProvider] Deleted {len(existing_chunks)} chunks for file {file_id}, file_part {file_part}."
        )

        session.add_all(chunk_records)
        await session.commit()

        new_chunk_ids = [record.id for record in chunk_records]
        logger.info(f"[FileHelper] saved {len(chunk_records)} chunks.")
        return existing_chunk_ids, new_chunk_ids
    except Exception as e:
        logger.error(f"[FileHelper] Error saving chunks to db: {e}")
        await session.rollback()
        raise e

@with_async_db_session
async def update_chunk_status_async(
    session: AsyncSession,
    chunk_ids: List[str],
    status: ChunkStatus,
    tenant_id: str,
) -> None:
    try:
        logger.info(f"[FileHelper] updating chunk {chunk_ids} status to {status}.")
        await session.exec(
            update(KbChunkEntity)
            .where(KbChunkEntity.id.in_(chunk_ids), KbChunkEntity.tenant_id == tenant_id)
            .values(status=status.value)
        )
        await session.commit()
        logger.info(
            f"[FileHelper] successfully updated chunk {chunk_ids} status to {status}."
        )
    except Exception as e:
        logger.error(f"[FileHelper] failed to update chunk {chunk_ids} status to {status}. error: {e}")
        await session.rollback()


@with_async_db_session
async def get_kb_chunk_ids(
    session: AsyncSession,
    kb_id: str,
    file_id: str,
    tenant_id: str,
) -> List[str]:
    chunk_ids = (await session.exec(
        select(KbChunkEntity.id).where(
            KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id, KbChunkEntity.tenant_id == tenant_id
        )
    )).all()

    logger.info(f"[FileHelper] get {len(chunk_ids)} chunk ids for kb {kb_id} and file {file_id}")
    return chunk_ids



@with_async_db_session
async def clear_useless_file_resources_async(
    session: AsyncSession,
    kb_id: str,
    file_id: str,
    tenant_id: str,
    part_count: int = 0,
):
    try:
        clear_task_statement = delete(KbFileTaskEntity).where(
            KbFileTaskEntity.kb_id == kb_id,
            KbFileTaskEntity.file_id == file_id,
            KbFileTaskEntity.file_part > part_count,
            KbFileTaskEntity.tenant_id == tenant_id,
        )
        clear_task_result = await session.exec(clear_task_statement)

        filter_chunk_id_clause = select(KbChunkEntity.id).where(
            KbChunkEntity.kb_id == kb_id,
            KbChunkEntity.file_id == file_id,
            KbChunkEntity.file_part > part_count,
            KbChunkEntity.tenant_id == tenant_id,
        )
        chunk_ids_to_delete = (await session.exec(filter_chunk_id_clause)).all()
        clear_chunk_statement = delete(KbChunkEntity).where(
            KbChunkEntity.kb_id == kb_id,
            KbChunkEntity.file_id == file_id,
            KbChunkEntity.file_part > part_count,
            KbChunkEntity.tenant_id == tenant_id,
        )
        clear_chunk_result = await session.exec(clear_chunk_statement)
        logger.info(f"Deleted {clear_task_result.rowcount} file tasks and {clear_chunk_result.rowcount} chunks for file {file_id}.")

        await session.commit()
        return chunk_ids_to_delete
    except Exception as e:
        logger.error(f"Error deleting file tasks for file {file_id}: {e}")
        await session.rollback()


@with_async_db_session
async def delete_file_tasks_by_file_id_async(
    session: AsyncSession,
    kb_id: str,
    file_id: str,
    tenant_id: str,
):
    try:
        del_statement = delete(KbFileTaskEntity).where(
            KbFileTaskEntity.kb_id == kb_id,
            KbFileTaskEntity.file_id == file_id,
            KbFileTaskEntity.tenant_id == tenant_id,
        )
        result = await session.exec(del_statement)
        await session.commit()
        logger.info(f"Deleted {result.rowcount} file tasks for file {file_id}.")
    except Exception as e:
        logger.error(f"Error deleting file tasks for file {file_id}: {e}")
        await session.rollback()


@with_async_db_session
async def save_file_task_async(
    session: AsyncSession,
    task_entity: KbFileTaskEntity,
    tenant_id: str,
):
    logger.info(f"Saving file task entity {task_entity} for tenant {tenant_id}.")
    try:
        target_task_entity = (await session.exec(
            select(KbFileTaskEntity).where(
                KbFileTaskEntity.kb_id == task_entity.kb_id,
                KbFileTaskEntity.file_id == task_entity.file_id,
                KbFileTaskEntity.file_part == task_entity.file_part,
                KbFileTaskEntity.tenant_id == tenant_id,
            )
        )).first()

        if target_task_entity:
            logger.info(f"File task {task_entity.id} already exists, updating.")
            target_task_entity.file_version = task_entity.file_version
            target_task_entity.status = task_entity.status
            target_task_entity.failed_reason = task_entity.failed_reason
        else:
            target_task_entity = task_entity

        session.add(target_task_entity)
        await session.commit()
        await session.refresh(target_task_entity)
        logger.info(f"Saved file task entity {target_task_entity}.")

        return target_task_entity
    except Exception as e:
        logger.error(f"Error saving file task entity {task_entity}: {e}")
        await session.rollback()
        raise


@with_async_db_session
async def get_file_task_async(
    session: AsyncSession,
    task_id: str,
    tenant_id: str,
) -> KbFileTaskEntity:
    select_statement = select(KbFileTaskEntity).where(KbFileTaskEntity.id == task_id, KbFileTaskEntity.tenant_id == tenant_id)
    target_task_entity = (await session.exec(select_statement)).first()
    if not target_task_entity:
        logger.error(f"[FileHelper] File task {task_id} not found, skip getting file task.")
        raise ValueError(f"File task {task_id} not found.")
    return target_task_entity
