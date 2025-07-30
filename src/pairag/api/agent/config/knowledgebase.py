### Knowledgebase configuration API ###
from datetime import datetime, timezone
import traceback
from typing import List
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import JSONResponse
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.chat.models import DocRecord, NewRetrievalResponse, RetrievalRequest
from pairag.db.models.change_event import ChangeEventSource, ChangeEventType
from pairag.db.models.knowledgebase.chunk import KbChunkEntity, KbChunkModel, create_text_node_from_chunk
from pairag.db.models.knowledgebase.file import KbFileEntity
from pairag.db.models.knowledgebase.knowledgebase import (
    ChunkConfig,
    KbEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from pairag.db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.config_change_manager import config_change_manager
from pairag.mcp.providers.embedding_provider import embedding_provider
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider
from pairag.mcp.rag.file.store.file_store_helper import file_store
from pairag.api.response_model import ResponseModel, PagedResult, success_response, error_response
from pairag.mcp.tools.knowledgebase.knowledgebase_tool import kb_client
from loguru import logger
import re
from pairag.mcp.rag.file.models.file_item import FileItem
from pairag.api.agent.utils.paginate import get_pagination_meta

knowledgebase_router = APIRouter()
@knowledgebase_router.post(
    "/retrieval", response_model=ResponseModel[NewRetrievalResponse]
)
async def retrieval(
    retrieval_request: RetrievalRequest, session: AsyncSession = Depends(get_session)
):
    knowledgebase = await session.get(KbEntity, retrieval_request.knowledgebase_id)
    if knowledgebase is None:
        return error_response(
            code=404, message=f"找不到知识库{retrieval_request.knowledgebase_id}"
        )

    node_results = await kb_client.aquery(
        query=retrieval_request.query,
        knowledge_id=retrieval_request.knowledgebase_id,
        retrieval_setting=retrieval_request.retrieval_setting,
        metadata_condition=retrieval_request.metadata_condition,
    )
    logger.info(
        f"Retrieved {len(node_results)} for query '{retrieval_request.query}' against knowledgebase {retrieval_request.knowledgebase_id}."
    )
    records = []
    for score_node in node_results:
        origin_text = score_node.node.get_content()
        pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"'
        matches = re.findall(pattern, origin_text)

        score_node.node.metadata["images_info"] = [{"url":file_store.get_url(src), "desc": alt } for src, alt in matches]
        records.append(DocRecord(
            content=score_node.node.get_content(),
            score=score_node.score,
            title=score_node.node.metadata.get("file_name", "null"),
            metadata=score_node.node.metadata,
        ))
    return success_response(data=NewRetrievalResponse(records=records), message="查询成功。")

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
            return JSONResponse(
                content=error_response(code=400, message="创建知识库失败: 知识库名称已存在。"),
                status_code=400,
            )
        else:
            return JSONResponse(
                content=error_response(code=400, message=f"创建知识库失败: {e}."),
                status_code=400,
            )
    except Exception:
        logger.exception(f"创建知识库失败。\nException:{traceback.format_exc()}")
        await session.rollback()
        return JSONResponse(
            content=error_response(code=400, message=f"创建知识库失败: {traceback.format_exc()}."),
            status_code=400,
        )


@knowledgebase_router.get("")
async def list_knowledgebases(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    total_results = await session.exec(
        select(func.count()).select_from(KbEntity)
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    kb_results = await session.exec(
        select(KbEntity)
        .order_by(KbEntity.created_at.desc())
        .offset(pagination.offset)
        .limit(size)
    )
    kb_entities = kb_results.all()

    return success_response(
        data=PagedResult(
            items=kb_entities,
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
        return JSONResponse(
            content=error_response(code=404, message=f"查询知识库失败: 知识库'{kb_id}'不存在。"),
            status_code=404,
        )

    return success_response(data=knowledgebase, message="查询知识库成功。")


@knowledgebase_router.patch("/{kb_id}", response_model=ResponseModel[KbEntity])
async def update_knowledgebase(
    kb_id: str,
    new_kb: KnowledgebaseCreate,
    session: AsyncSession = Depends(get_session),
):
    knowledgebase = await session.get(KbEntity, kb_id)
    if not knowledgebase:
        return JSONResponse(
            content=error_response(code=404, message=f"更新知识库失败: 知识库'{kb_id}'不存在。"),
            status_code=404,
        )

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
        return JSONResponse(
            content=error_response(code=404, message=f"删除知识库失败: 知识库'{kb_id}'不存在。"),
            status_code=404,
        )

    knowledgebase_provider.delete(kb_id)
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
    logger.info(f"Uploading files to {kb_id}")
    import pairag.mcp.rag.file_worker as worker

    if not files:
        raise ValueError("No files provided.")

    knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
    if knowledgebase is None:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    file_names = []
    file_entities = []
    for file in files:
        file_name = file.filename
        destination_file_path = f"{kb_id}/docs/{file_name}"
        file_store.save(
            file=file.file,
            file_path=destination_file_path,
        )
        file_item = FileItem.from_file(
            file=file.file,
            file_path=destination_file_path,
            kb_id=kb_id,
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
            file_entity = file_item.to_file_entity()
        else:
            file_entity.file_md5 = file_item.file_md5
            file_entity.file_size = file_item.file_size
            file_entity.update_at = datetime.now(timezone.utc)
        session.add(file_entity)
        await session.commit()
        logger.info(f"Saved file {file_entity} successfully.")
        worker.process_file.delay(file_entity.id)
        logger.info(f"Queued {file_entity.id} job successfully.")
        file_names.append(file.filename)
        file_entities.append(file_entity)

    return success_response(data=file_entities, message="文件上传成功")


@knowledgebase_router.get("/{kb_id}/files")
async def list_files(
    kb_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
) :
    total_results = await session.exec(
        select(func.count())
        .select_from(select(KbFileEntity).where(KbFileEntity.kb_id == kb_id))
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    file_results = await session.exec(
        select(KbFileEntity)
        .where(KbFileEntity.kb_id == kb_id)
        .order_by(KbFileEntity.update_at.desc())
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
        .order_by(KbFileEntity.update_at.desc())
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

    node_ids = [chunk_entity.id for chunk_entity in chunk_entities]
    await kb_client.adelete_chunks_from_vectordb(kb_id=kb_id, node_ids=node_ids)

    await session.delete(file_entity)
    for chunk_entity in chunk_entities:
        await session.delete(chunk_entity)

    await session.commit()
    logger.info(
        f"Delete file {file_id}@{kb_id}: deleted {len(chunk_entities)} chunks in total."
    )

    return success_response(data=node_ids, message="删除知识库文件成功。")


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
        .offset(pagination.offset)
        .limit(size)
    )
    chunk_entities = chunk_results.all()
    for chunk_entity in chunk_entities:
        origin_text = chunk_entity.text
        pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"'
        matches = re.findall(pattern, origin_text)
        chunk_entity.chunk_metadata["images_info"] = [{"url":file_store.get_url(src), "desc": alt } for src, alt in matches]
    return success_response(
        data=PagedResult(
            items=chunk_entities,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取切片列表成功")

@knowledgebase_router.patch("/{kb_id}/files/{file_id}/chunks/{chunk_id}", response_model=ResponseModel[KbChunkEntity])
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
        return JSONResponse(
            content=error_response(code=404, message=f"更新知识库切片失败: 切片'{chunk_id}'不存在 或 有误。"),
            status_code=404,
        )
    try:
        kb_chunk = kb_chunk_entities[0]
        # 更新chunk text
        if kb_chunk.text != update_kb_chunk.text:
            kb_chunk.text = update_kb_chunk.text
            node = create_text_node_from_chunk(kb_chunk)
            await kb_client.adelete_chunks_from_vectordb(kb_id=kb_id, node_ids=[kb_chunk.id])
            await kb_client.ainsert_chunks_to_vectordb(kb_id=kb_id, nodes=[node])
            logger.info(f"Update chunk text for {kb_chunk.id}")
        # 更新chunk active 若false : delete; 若true : insert
        if kb_chunk.active != update_kb_chunk.active:
            if not update_kb_chunk.active:
                await kb_client.adelete_chunks_from_vectordb(kb_id=kb_id, node_ids=[kb_chunk.id])
            else:
                node = create_text_node_from_chunk(kb_chunk)
                await kb_client.ainsert_chunks_to_vectordb(kb_id=kb_id, nodes=[node])
            kb_chunk.active = update_kb_chunk.active
            logger.info(f"Update chunk active to {kb_chunk.active} for {kb_chunk.id}")
        session.add(kb_chunk)
        await session.commit()
        await session.refresh(kb_chunk)
        return success_response(data=kb_chunk, message="更新知识库切片成功。")
    except Exception as ex:
        logger.error(f"Failed to update knowledgebase {kb_id} / file {file_id} / chunk {chunk_id}: {ex}")
        return error_response(message=f"更新知识库切片失败：{ex}")
