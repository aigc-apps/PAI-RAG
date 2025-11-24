### Knowledgebase configuration API ###
from datetime import datetime, timezone
import time
import traceback
from typing import List, Optional
from common.knowledgebase.types import FileStatus
from fastapi import APIRouter, Depends, File, Query, UploadFile
from pydantic import BaseModel, Field
from db.models.knowledgebase.metadata import KbMetadataEntity, FileMetadataEntity
from rag.split.excel_split import convert_xls_to_xlsx
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import or_
from rag.file_item_utils import to_file_entity
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.knowledgebase.chunk import KbChunkEntity, KbChunkModel, create_text_node_from_chunk
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.knowledgebase import (
    ChunkConfig,
    KbEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from config.providers.embedding_provider import embedding_provider
from config.providers.knowledgebase_provider import knowledgebase_provider
from pairag.file.store.file_store_helper import file_store
from api.response_model import ResponseModel, PagedResult, success_response, error_response
from tools.knowledgebase.knowledgebase_tool import MARKDOWN_IMAGE_PATTERN, kb_tool
from loguru import logger
import re
from pairag.file.models.file_item import FileItem
from api.v1.utils.paginate import get_pagination_meta

knowledgebase_router = APIRouter()


@knowledgebase_router.post("", response_model=ResponseModel[KbEntity])
async def create_knowledgebase(
    kb: KnowledgebaseCreate, session: AsyncSession = Depends(get_session)
):
    try:
        assert kb.embedding_model, "需要提供Embedding模型才能创建知识库。"
        # 验证embedding合法
        _ = embedding_provider.get_embedding_config(kb.embedding_model)

        kb.chunk_config = (kb.chunk_config or ChunkConfig()).model_dump()
        kb.retrieval_config = (kb.retrieval_config or RetrievalConfig()).model_dump()

        knowledgebase = KbEntity.model_validate(kb)
        knowledgebase_provider.add(knowledgebase)
        session.add(knowledgebase)
        await session.commit()
        await session.refresh(knowledgebase)

        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.KNOWLEDGEBASE,
            event_type=ChangeEventType.ADD,
            source_id=knowledgebase.id,
        )
        return success_response(data=knowledgebase, message="知识库创建成功。")

    except IntegrityError as e:
        # TODO: 这里有bug,logger.exception没有打印错误调用栈
        logger.exception(f"创建知识库失败。\nIntegrityError:{e}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return error_response(code=400, message="创建知识库失败: 知识库名称已存在。")
        else:
            return error_response(code=400, message=f"创建知识库失败: {e}.")
    except Exception:
        logger.exception(f"创建知识库失败。\nException:{traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"创建知识库失败: {traceback.format_exc()}.")


@knowledgebase_router.get("")
async def list_knowledgebases(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    query: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    # 基础查询条件：排除 default_attachments
    base_condition = KbEntity.name != "default_attachments"

    # 如果提供了查询参数，在 id、name、description 中搜索
    if query:
        query_lower = query.lower()
        search_condition = or_(
            func.lower(KbEntity.id).like(f"%{query_lower}%"),
            func.lower(KbEntity.name).like(f"%{query_lower}%"),
            func.lower(func.coalesce(KbEntity.description, "")).like(f"%{query_lower}%")
        )
        where_condition = base_condition & search_condition
    else:
        where_condition = base_condition

    # 计算总数
    total_results = await session.exec(
        select(func.count(KbEntity.id)).where(where_condition)
    )
    total_num = total_results.one_or_none() or 0
    pagination = get_pagination_meta(page, size, total_num)

    # 子查询：统计每个知识库的文件数量
    file_count_subquery = (
        select(
            KbFileEntity.kb_id,
            func.count(KbFileEntity.id).label('file_count')
        )
        .group_by(KbFileEntity.kb_id)
        .subquery()
    )

    # 获取分页结果，使用 LEFT JOIN 获取文件数量
    query = (
        select(
            KbEntity,
            func.coalesce(file_count_subquery.c.file_count, 0).label('file_count')
        )
        .outerjoin(
            file_count_subquery,
            KbEntity.id == file_count_subquery.c.kb_id
        )
        .where(where_condition)
        .order_by(KbEntity.created_at.desc())
        .offset(pagination.offset)
        .limit(size)
    )

    kb_results = await session.exec(query)
    kb_entities_with_counts = kb_results.all()

    # 构建返回结果，将文件数量添加到知识库实体中
    items = []
    for kb_entity, file_count in kb_entities_with_counts:
        # 将 KbEntity 转换为字典并添加 file_count 字段
        kb_dict = kb_entity.model_dump() if hasattr(kb_entity, 'model_dump') else kb_entity.__dict__
        kb_dict['file_count'] = int(file_count) if file_count else 0
        items.append(kb_dict)

    return success_response(
        data=PagedResult(
            items=items,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取知识库列表成功",
    )


@knowledgebase_router.get("/{kb_id}", response_model=ResponseModel[KbEntity])
async def read_knowledgebase(kb_id: str, session: AsyncSession = Depends(get_session)):
    knowledgebase = await session.get(KbEntity, kb_id)

    if not knowledgebase:
        return error_response(code=404, message=f"查询知识库失败: 知识库'{kb_id}'不存在。")

    return success_response(data=knowledgebase, message="查询知识库成功。")


@knowledgebase_router.put("/{kb_id}", response_model=ResponseModel[KbEntity])
async def update_knowledgebase(
    kb_id: str,
    new_kb: KnowledgebaseCreate,
    session: AsyncSession = Depends(get_session),
):
    knowledgebase = await session.get(KbEntity, kb_id)
    if not knowledgebase:
        return error_response(code=404, message=f"更新知识库失败: 知识库'{kb_id}'不存在。")

    try:
        knowledgebase.name = new_kb.name or knowledgebase.name
        knowledgebase.description = new_kb.description or knowledgebase.description
        knowledgebase.embedding_model = (
            new_kb.embedding_model or knowledgebase.embedding_model
        )
        if new_kb.chunk_config:
            knowledgebase.chunk_config = new_kb.chunk_config.model_dump()
        if new_kb.retrieval_config:
            knowledgebase.retrieval_config = new_kb.retrieval_config.model_dump()

        knowledgebase.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        knowledgebase_provider.update(knowledgebase)
        session.add(knowledgebase)
        await session.commit()
        await session.refresh(knowledgebase)

        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.KNOWLEDGEBASE,
            event_type=ChangeEventType.UPDATE,
            source_id=knowledgebase.id,
        )

        logger.info(f"Knowledgebase {kb_id} updated to {knowledgebase}.")

        return success_response(data=knowledgebase, message="更新知识库成功。")
    except Exception:
        logger.error(f"Failed to update knowledgebase {kb_id}: {traceback.format_exc()}")
        return error_response(message=f"更新知识库失败：{traceback.format_exc()}")


@knowledgebase_router.delete("/{kb_id}")
async def delete_knowledgebase(
    kb_id: str,
    session: AsyncSession = Depends(get_session),
):
    knowledgebase = await session.get(KbEntity, kb_id)

    if not knowledgebase:
        return error_response(code=404, message=f"删除知识库失败: 知识库'{kb_id}'不存在。")


    knowledgebase_provider.delete(kb_id)

    # delete related chunks
    kb_chunks = await session.exec(
        select(KbChunkEntity).where(KbChunkEntity.kb_id == kb_id)
    )
    for chunk in kb_chunks:
        await session.delete(chunk)

    # delete related files
    kb_files = await session.exec(
        select(KbFileEntity).where(KbFileEntity.kb_id == kb_id)
    )
    for file in kb_files:
        await session.delete(file)

    # delete related metadata
    kb_metadatas = await session.exec(
        select(KbMetadataEntity).where(KbMetadataEntity.kb_id == kb_id)
    )
    for metadata in kb_metadatas:
        await session.delete(metadata)


    file_metadatas = await session.exec(
        select(FileMetadataEntity).where(FileMetadataEntity.kb_id == kb_id)
    )
    for metadata in file_metadatas:
        await session.delete(metadata)

    await session.delete(knowledgebase)
    await session.commit()

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.KNOWLEDGEBASE,
        event_type=ChangeEventType.DELETE,
        source_id=knowledgebase.id,
    )

    logger.info(f"Knowledgebase {kb_id} has been deleted.")

    return success_response(message=f"知识库'{kb_id}'删除成功。")


@knowledgebase_router.post("/{kb_id}/files")
async def upload_files(
    kb_id: str,
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Uploading files to {kb_id}.")
    import app.worker as background_worker

    file_version = int(time.time())

    if not files:
        return error_response(code=400, message="没有上传任何文件。")

    try:
        try:
            _ = await knowledgebase_provider.aget_knowledgebase(kb_id)
        except ValueError:
            logger.error(f"没找到知识库{kb_id}")
            return error_response(code=400, message=f"没有找到知识库 {kb_id}。")

        file_entities = []
        for single_file in files:
            # Convert xls to xlsx
            file_name = single_file.filename
            file_data = single_file.file
            if single_file.filename.endswith(".xls"):
                file_data = convert_xls_to_xlsx(file_data)
                file_name = file_name[:-4] + ".xlsx"

            destination_file_path = f"{kb_id}/docs/{file_name}"
            file_store.save(
                file=file_data,
                file_path=destination_file_path,
            )
            file_item = FileItem.from_file(
                file=file_data,
                file_path=destination_file_path,
                kb_id=kb_id,
                file_name=file_name,
            )
            file_entity = (
                await session.exec(
                    select(KbFileEntity).where(
                        KbFileEntity.kb_id == kb_id,
                        KbFileEntity.file_name == file_item.file_name,
                    )
                )
            ).first()
            if not file_entity:
                file_entity = to_file_entity(file_item)
            else:
                file_entity.file_md5 = file_item.file_md5
                file_entity.file_size = file_item.file_size
                file_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            file_entity.file_version = file_version
            session.add(file_entity)


            await session.commit()
            logger.info(f"Saved file {file_entity} successfully.")
            background_worker.enqueue_file_tasks.delay(file_entity.id, file_entity.file_version, is_attachment=False)
            logger.info(f"Queued {file_entity.id} job successfully.")
            file_entities.append(file_entity)

        return success_response(data=file_entities, message="文件上传成功")
    except Exception as e:
        logger.error(f"Failed to upload file: {traceback.format_exc()}")
        await session.rollback()
        return error_response(message=f"Failed to save file to database: {e}")


@knowledgebase_router.get("/{kb_id}/files")
async def list_files(
    kb_id: str,
    file_name: Optional[str] = None,
    query: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
) :
    try:
        if file_name:
            kb_file = (await session.exec(
                select(KbFileEntity).where(KbFileEntity.file_name == file_name)
            )).first()
            if kb_file is None:
                return error_response(code=404, message=f"文件名'{file_name}'不存在")

            return success_response(data=kb_file, message="查询文件成功")
        else:
            file_select_statement = select(KbFileEntity).where(
                    KbFileEntity.kb_id == kb_id)

            if query:
                file_select_statement = file_select_statement.where(func.lower(KbFileEntity.file_name).like(func.lower(f"%{query}%")))

            if status:
                file_select_statement = file_select_statement.where(KbFileEntity.status == status)

            count_statement = select(func.count()).select_from(file_select_statement)
            total_results = await session.exec(count_statement)
            total_num = total_results.one_or_none()
            pagination = get_pagination_meta(page, size, total_num)

            file_results = await session.exec(
                file_select_statement
                .order_by(KbFileEntity.updated_at.desc())
                .offset(pagination.offset)
                .limit(size)
            )
            file_entities = file_results.all()

            return success_response(
                data=PagedResult(
                    items=file_entities,
                    total=pagination.total,
                    pages=pagination.pages,
                    page=pagination.page,
                    size=pagination.size,
                ),
                message="获取文件列表成功")
    except Exception as ex:
        logger.error(f"Failed to list files: {traceback.format_exc()}")
        return error_response(code=400, message=f"获取文件列表失败: {ex}")


@knowledgebase_router.put(
    "/{kb_id}/files/{file_id}", response_model=ResponseModel[KbFileEntity]
)
async def reprocess_file(
    kb_id: str,
    file_id: str,
    session: AsyncSession = Depends(get_session),
):
    import app.worker as background_worker
    file_res = await session.exec(
        select(KbFileEntity).where(
            KbFileEntity.id == file_id, KbFileEntity.kb_id == kb_id
        )
    )
    file_entity = file_res.first()
    if file_entity is None:
        return error_response(code=404, message=f"没有在知识库{kb_id}中找到文件{file_id}。")

    file_entity.status = FileStatus.pending
    file_entity.file_version = int(time.time())
    session.add(file_entity)
    await session.commit()
    logger.info(f"Re-process file {file_entity} successfully.")
    background_worker.enqueue_file_tasks.delay(file_entity.id, file_entity.file_version, is_attachment=False)

    return success_response(data=file_entity, message="文件入队成功。")


@knowledgebase_router.get(
    "/{kb_id}/files/{file_id}", response_model=ResponseModel[KbFileEntity]
)
async def get_kb_file(
    kb_id: str,
    file_id: str,
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
    session: AsyncSession = Depends(get_session),
):
    file_results = await session.exec(
        select(KbFileEntity)
        .where(KbFileEntity.kb_id == kb_id)
        .where(KbFileEntity.id == file_id)
        .order_by(KbFileEntity.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    file_entities = file_results.all()
    assert len(file_entities) <= 1
    logger.info(f"Get kb files: get {len(file_entities)} in total.")
    file_url = file_store.get_url(file_entities[0].file_path)
    file_entities[0].file_metadata["file_url"] = file_url
    return success_response(data=file_entities[0], message="查询知识库文件成功。")


@knowledgebase_router.delete("/{kb_id}/files/{file_id}")
async def delete_file(
    kb_id: str,
    file_id: str,
    session: AsyncSession = Depends(get_session),
):
    file_res = await session.exec(
        select(KbFileEntity).where(
            KbFileEntity.id == file_id, KbFileEntity.kb_id == kb_id
        )
    )
    file_entity = file_res.first()
    if file_entity is None:
        return error_response(code=404, message=f"没有在知识库{kb_id}中找到文件{file_id}。")

    chunks_res = await session.exec(
        select(KbChunkEntity).where(
            KbChunkEntity.file_id == file_id, KbChunkEntity.kb_id == kb_id
        )
    )
    chunk_entities = chunks_res.all()

    await kb_tool.adelete_doc(kb_id=kb_id, file_id=file_id)

    await session.delete(file_entity)
    for chunk_entity in chunk_entities:
        await session.delete(chunk_entity)

    await session.commit()
    logger.info(
        f"Delete file {file_id}@{kb_id}: deleted {len(chunk_entities)} chunks in total."
    )

    return success_response(data=None, message="删除知识库文件成功。")


class BatchDeleteFilesRequest(BaseModel):
    file_id_list: List[str] = Field(..., description="要删除的文件ID列表")


class BatchOperationRequest(BaseModel):
    operation: str = Field(..., description="操作类型: 'delete' 或 'reprocess'")
    file_id_list: List[str] = Field(..., description="要操作的文件ID列表")


@knowledgebase_router.post("/{kb_id}/files/batch", response_model=ResponseModel[dict])
async def batch_operations(
    kb_id: str,
    request: BatchOperationRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    批量操作知识库中的文件
    支持的操作：
    - delete: 批量删除文件
    - reprocess: 批量重新处理文件
    """
    if not request.file_id_list:
        return error_response(code=400, message="文件ID列表不能为空。")

    if request.operation not in ["delete", "reprocess"]:
        return error_response(
            code=400,
            message=f"不支持的操作类型: {request.operation}。支持的操作: delete, reprocess"
        )

    # 验证所有文件是否存在
    file_res = await session.exec(
        select(KbFileEntity).where(
            KbFileEntity.id.in_(request.file_id_list),
            KbFileEntity.kb_id == kb_id
        )
    )
    file_entities = file_res.all()
    found_file_ids = {entity.id for entity in file_entities}
    not_found_ids = [file_id for file_id in request.file_id_list if file_id not in found_file_ids]

    if not_found_ids:
        return error_response(
            code=404,
            message=f"以下文件在知识库{kb_id}中不存在: {', '.join(not_found_ids)}"
        )

    if request.operation == "delete":
        return await _batch_delete_files(kb_id, file_entities, request.file_id_list, session)
    elif request.operation == "reprocess":
        return await _batch_reprocess_files(kb_id, file_entities, session)


async def _batch_delete_files(
    kb_id: str,
    file_entities: List[KbFileEntity],
    file_id_list: List[str],
    session: AsyncSession,
) -> ResponseModel[dict]:
    """
    批量删除文件的内部实现
    """
    # 查询所有文件的chunks
    chunks_res = await session.exec(
        select(KbChunkEntity).where(
            KbChunkEntity.file_id.in_(file_id_list),
            KbChunkEntity.kb_id == kb_id
        )
    )
    chunk_entities = chunks_res.all()
    chunk_ids = [chunk.id for chunk in chunk_entities]
    total_chunks_deleted = len(chunk_entities)

    try:
        # 删除向量库中的chunks
        if chunk_ids:
            await kb_tool.adelete_chunks_from_vectordb(kb_id=kb_id, node_ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks from vector store for kb {kb_id}")

        # 删除数据库中的chunk_entities
        for chunk_entity in chunk_entities:
            await session.delete(chunk_entity)

        # 删除数据库中的file_entities
        for file_entity in file_entities:
            await session.delete(file_entity)

        # 提交所有删除操作
        await session.commit()

        logger.info(
            f"Batch delete files@{kb_id}: deleted {len(file_id_list)} files, "
            f"{total_chunks_deleted} chunks in total."
        )

        return success_response(
            data={
                "deleted_count": len(file_id_list),
                "total_chunks_deleted": total_chunks_deleted,
            },
            message=f"成功删除 {len(file_id_list)} 个文件，共删除 {total_chunks_deleted} 个切片。"
        )

    except Exception as e:
        await session.rollback()
        logger.exception(f"Failed to batch delete files@{kb_id}: {e}")
        return error_response(code=500, message=f"批量删除失败: {str(e)}")


async def _batch_reprocess_files(
    kb_id: str,
    file_entities: List[KbFileEntity],
    session: AsyncSession,
) -> ResponseModel[dict]:
    """
    批量重新处理文件的内部实现
    """
    import app.worker as background_worker

    file_version = int(time.time())
    reprocessed_count = 0

    try:
        for file_entity in file_entities:
            file_entity.status = FileStatus.pending
            file_entity.file_version = file_version
            session.add(file_entity)
            reprocessed_count += 1

        await session.commit()

        # 为每个文件入队处理任务
        for file_entity in file_entities:
            background_worker.enqueue_file_tasks.delay(
                file_entity.id,
                file_entity.file_version,
                is_attachment=False
            )
            logger.info(f"Queued file {file_entity.id} for reprocessing.")

        logger.info(
            f"Batch reprocess files@{kb_id}: queued {reprocessed_count} files for reprocessing."
        )

        return success_response(
            data={
                "reprocessed_count": reprocessed_count,
            },
            message=f"成功将 {reprocessed_count} 个文件加入重新处理队列。"
        )

    except Exception as e:
        await session.rollback()
        logger.exception(f"Failed to batch reprocess files@{kb_id}: {e}")
        return error_response(code=500, message=f"批量重新处理失败: {str(e)}")


@knowledgebase_router.post("/{kb_id}/files/batch_delete")
async def batch_delete_files(
    kb_id: str,
    request: BatchDeleteFilesRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    批量删除知识库中的文件
    流程：
    1. 验证所有文件是否存在，如有缺失直接返回错误
    2. 查询所有文件的chunks
    3. 删除向量库中的chunks
    4. 删除数据库中的chunk_entities
    5. 删除向量库中的doc
    6. 删除数据库中的file_entities
    """
    if not request.file_id_list:
        return error_response(code=400, message="文件ID列表不能为空。")

    # 第一步：验证所有文件是否存在
    file_res = await session.exec(
        select(KbFileEntity).where(
            KbFileEntity.id.in_(request.file_id_list),
            KbFileEntity.kb_id == kb_id
        )
    )
    file_entities = file_res.all()
    found_file_ids = {entity.id for entity in file_entities}
    not_found_ids = [file_id for file_id in request.file_id_list if file_id not in found_file_ids]

    if not_found_ids:
        return error_response(
            code=404,
            message=f"以下文件在知识库{kb_id}中不存在: {', '.join(not_found_ids)}"
        )

    # 第二步：查询所有文件的chunks
    chunks_res = await session.exec(
        select(KbChunkEntity).where(
            KbChunkEntity.file_id.in_(request.file_id_list),
            KbChunkEntity.kb_id == kb_id
        )
    )
    chunk_entities = chunks_res.all()
    chunk_ids = [chunk.id for chunk in chunk_entities]
    total_chunks_deleted = len(chunk_entities)

    try:
        # 第三步：删除向量库中的chunks
        if chunk_ids:
            await kb_tool.adelete_chunks_from_vectordb(kb_id=kb_id, node_ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks from vector store for kb {kb_id}")

        # 第四步：删除数据库中的chunk_entities
        for chunk_entity in chunk_entities:
            await session.delete(chunk_entity)

        # 第五步：删除数据库中的file_entities
        for file_entity in file_entities:
            await session.delete(file_entity)

        # 提交所有删除操作
        await session.commit()

        logger.info(
            f"Batch delete files@{kb_id}: deleted {len(request.file_id_list)} files, "
            f"{total_chunks_deleted} chunks in total."
        )

        return success_response(
            data={
                "deleted_count": len(request.file_id_list),
                "total_chunks_deleted": total_chunks_deleted,
            },
            message=f"成功删除 {len(request.file_id_list)} 个文件，共删除 {total_chunks_deleted} 个切片。"
        )

    except Exception as e:
        await session.rollback()
        logger.exception(f"Failed to batch delete files@{kb_id}: {e}")
        return error_response(code=500, message=f"批量删除失败: {str(e)}")


@knowledgebase_router.get("/{kb_id}/files/{file_id}/chunks")
async def list_chunks(
    kb_id: str,
    file_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):

    total_results = await session.exec(
        select(func.count()).select_from(
            select(KbChunkEntity).where(
                KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id
            )
        )
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    chunk_results = await session.exec(
        select(KbChunkEntity)
        .where(KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id)
        .order_by(KbChunkEntity.index)
        .offset(pagination.offset)
        .limit(size)
    )
    chunk_entities = chunk_results.all()
    for chunk_entity in chunk_entities:
        origin_text = chunk_entity.text
        pattern = MARKDOWN_IMAGE_PATTERN
        matches = re.findall(pattern, origin_text, re.DOTALL)
        chunk_entity.chunk_metadata["images_info"] = [{"url":file_store.get_url(src), "desc": desc} for src, desc in matches]
    return success_response(
        data=PagedResult(
            items=chunk_entities,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取切片列表成功")



class FileSourceParam(BaseModel):
    file_source: str = Field(default=None)



@knowledgebase_router.post("/{kb_id}/files/{file_id}/source", response_model=ResponseModel[KbFileEntity])
async def set_file_source(
    kb_id: str,
    file_id: str,
    body: FileSourceParam,
    session: AsyncSession = Depends(get_session),
):
    file_entity = await session.get(KbFileEntity, file_id)
    if not file_entity:
        return error_response(code=404, message="文件不存在。")

    if not body.file_source:
        return error_response(code=400, message="文件来源不能为空。")

    file_entity.file_source = body.file_source
    session.add(file_entity)
    await session.commit()
    await session.refresh(file_entity)

    return success_response(data=file_entity, message="更新文件来源成功")



@knowledgebase_router.put("/{kb_id}/files/{file_id}/chunks/{chunk_id}", response_model=ResponseModel[KbChunkEntity])
async def update_chunk(
    kb_id: str,
    file_id: str,
    chunk_id: str,
    update_kb_chunk: KbChunkModel,
    session: AsyncSession = Depends(get_session),
):
    sql_results = await session.exec(
        select(KbChunkEntity)
        .where(
            KbChunkEntity.id == chunk_id,
            KbChunkEntity.kb_id == kb_id,
            KbChunkEntity.file_id == file_id,
        ))
    kb_chunk_entities = sql_results.all()
    if len(kb_chunk_entities) != 1:
        return error_response(code=404, message=f"更新知识库切片失败: 切片'{chunk_id}'不存在 或 有误。")

    try:
        kb_chunk = kb_chunk_entities[0]
        # 更新chunk text
        if kb_chunk.text != update_kb_chunk.text:
            kb_chunk.text = update_kb_chunk.text
            node = create_text_node_from_chunk(kb_chunk)
            await kb_tool.adelete_chunks_from_vectordb(kb_id=kb_id, node_ids=[kb_chunk.id])
            await kb_tool.ainsert_chunks_to_vectordb(kb_id=kb_id, nodes=[node])
            logger.info(f"Update chunk text for {kb_chunk.id}")
        # 更新chunk active 若false : delete; 若true : insert
        if kb_chunk.active != update_kb_chunk.active:
            if not update_kb_chunk.active:
                await kb_tool.adelete_chunks_from_vectordb(kb_id=kb_id, node_ids=[kb_chunk.id])
            else:
                node = create_text_node_from_chunk(kb_chunk)
                await kb_tool.ainsert_chunks_to_vectordb(kb_id=kb_id, nodes=[node])
            kb_chunk.active = update_kb_chunk.active
            logger.info(f"Update chunk active to {kb_chunk.active} for {kb_chunk.id}")
        session.add(kb_chunk)
        await session.commit()
        await session.refresh(kb_chunk)
        return success_response(data=kb_chunk, message="更新知识库切片成功。")
    except Exception as ex:
        logger.error(f"Failed to update knowledgebase {kb_id} / file {file_id} / chunk {chunk_id}: {ex}")
        return error_response(message=f"更新知识库切片失败：{ex}")
