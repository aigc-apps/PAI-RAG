### Knowledgebase configuration API ###
from datetime import datetime, timezone
import asyncio
from typing import List
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi_pagination import Page, Params, add_pagination
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.chat.models import DocRecord, NewRetrievalResponse, RetrievalRequest
from pairag.db.models.knowledgebase.chunk import KbChunkEntity, KbChunkModel
from pairag.db.models.knowledgebase.file import KbFileEntity
from pairag.db.models.knowledgebase.knowledgebase import (
    ChunkConfig,
    KbEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from pairag.db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.mcp_tool_provider import mcp_provider
from pairag.mcp.providers.embedding_provider import embedding_provider
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider
from pairag.mcp.rag.file.store.file_store_helper import file_store
from pairag.api.response_model import ResponseModel, success_response, error_response
from pairag.mcp.tools.knowledgebase.knowledgebase_tool import kb_client
from loguru import logger
import re
from pairag.mcp.rag.file.models.file_item import FileItem
from pairag.mcp.utils.metadata_utils import ensure_metadata_configs_is_valid

knowledgebase_router = APIRouter()
add_pagination(knowledgebase_router)

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
        query_str=retrieval_request.query,
        kb_id=retrieval_request.knowledgebase_id,
    )
    logger.info(
        f"Retrieved {len(node_results)} for query '{retrieval_request.query}' against knowledgebase {retrieval_request.knowledgebase_id}."
    )

    records = [
        DocRecord(
            content=score_node.node.get_content(),
            score=score_node.score,
            title=score_node.node.metadata.get("file_name", "null"),
            metadata=score_node.node.metadata,
        )
        for score_node in node_results
    ]

    return success_response(data=NewRetrievalResponse(records=records), message="查询成功。")

@knowledgebase_router.post("", response_model=ResponseModel[KbEntity])
async def create_knowledgebase(
    kb: KnowledgebaseCreate, session: AsyncSession = Depends(get_session)
):
    try:
        assert kb.embedding_model, "需要提供Embedding模型才能创建知识库。"
        ensure_metadata_configs_is_valid(kb.metadata_configs)
        # 验证embedding合法
        _ = embedding_provider.get_embedding_config(kb.embedding_model)

        kb.chunk_config = (kb.chunk_config or ChunkConfig()).model_dump()
        kb.retrieval_config = (kb.retrieval_config or RetrievalConfig()).model_dump()

        if kb.metadata_configs:
            kb.metadata_configs = [
                metadata_config.model_dump() for metadata_config in kb.metadata_configs
            ]

        knowledgebase = KbEntity.model_validate(kb)
        session.add(knowledgebase)
        await session.commit()
        await session.refresh(knowledgebase)
        asyncio.create_task(knowledgebase_provider.refresh())
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
    except Exception as e:
        logger.exception(f"创建知识库失败。\nException:{e}")
        await session.rollback()
        return JSONResponse(
            content=error_response(code=400, message=f"创建知识库失败: {e}."),
            status_code=400,
        )


@knowledgebase_router.get("", response_model=Page[KbEntity])
async def list_knowledgebases(
    params: Params = Depends(),
    session: AsyncSession = Depends(get_session),
):
    sql_query = (select(KbEntity))
    paginated_result = await paginate(session, sql_query, params)
    return paginated_result


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
        ensure_metadata_configs_is_valid(new_kb.metadata_configs)

        knowledgebase.name = new_kb.name or knowledgebase.name
        knowledgebase.description = new_kb.description or knowledgebase.description
        knowledgebase.embedding_model = (
            new_kb.embedding_model or knowledgebase.embedding_model
        )
        if new_kb.chunk_config:
            knowledgebase.chunk_config = new_kb.chunk_config.model_dump()
        if new_kb.retrieval_config:
            knowledgebase.retrieval_config = new_kb.retrieval_config.model_dump()
        if new_kb.metadata_configs:
            knowledgebase.metadata_configs = [
                metadata_config.model_dump()
                for metadata_config in new_kb.metadata_configs
            ]

        session.add(knowledgebase)
        await session.commit()
        await session.refresh(knowledgebase)

        asyncio.create_task(knowledgebase_provider.refresh())

        logger.info(f"Knowledgebase {kb_id} updated to {knowledgebase}.")

        return success_response(data=knowledgebase, message="更新知识库成功。")
    except Exception as ex:
        logger.error(f"Failed to update knowledgebase {kb_id}: {ex}")
        return error_response(message=f"更新知识库失败：{ex}")


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

    await session.delete(knowledgebase)
    await session.commit()

    asyncio.create_task(mcp_provider.refresh())

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

    """新知识库上传文件"""
    if kb_id not in knowledgebase_provider.knowledgebase_map:
        raise ValueError(f"Knowledgebase '{kb_id}' not found.")

    if not files:
        raise ValueError("No files provided.")

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


@knowledgebase_router.get("/{kb_id}/files", response_model=Page[KbFileEntity])
async def list_files(
    kb_id: str,
    params: Params = Depends(),
    session: AsyncSession = Depends(get_session),
) :
    sql_query = (select(KbFileEntity)
        .where(KbFileEntity.kb_id == kb_id)
        .order_by(KbFileEntity.update_at.desc()))
    paginated_result = await paginate(session, sql_query, params)
    return paginated_result


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


@knowledgebase_router.get("/{kb_id}/files/{file_id}/chunks", response_model=Page[KbChunkEntity])
async def list_chunks(
    kb_id: str,
    file_id: str,
    params: Params = Depends(),
    session: AsyncSession = Depends(get_session),
):
    sql_query = (
        select(KbChunkEntity)
        .where(KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id)
    )
    chunk_results = await paginate(session, sql_query, params)
    chunk_entities = chunk_results.items
    for chunk_entity in chunk_entities:
        images = chunk_entity.chunk_metadata.get("images", [])
        chunk_entity.chunk_metadata["images_info"] = []
        if images:
            origin_text = chunk_entity.text
            for image_file in images:
                image_url = file_store.get_url(image_file)
                pattern = rf'<img src="{re.escape(image_file)}" alt="([^"]*)"'
                match = re.search(pattern, origin_text)
                if match:
                    chunk_entity.chunk_metadata["images_info"].append(
                        {"url": image_url, "desc": match.group(1)}
                    )
                else:
                    chunk_entity.chunk_metadata["images_info"].append(
                        {"url": image_url, "desc": "null"}
                    )
                # origin_text = re.sub(r"<img[^>]*>", "", origin_text)
            chunk_entity.text = origin_text
    logger.info(f"Listing chunks: get {len(chunk_entities)} in total.")

    return chunk_results


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
        kb_chunk.text = update_kb_chunk.text
        kb_chunk.active = update_kb_chunk.active
        session.add(kb_chunk)
        await session.commit()
        await session.refresh(kb_chunk)
        return success_response(data=kb_chunk, message="更新知识库切片成功。")
    except Exception as ex:
        logger.error(f"Failed to update knowledgebase {kb_id} / file {file_id} / chunk {chunk_id}: {ex}")
        return error_response(message=f"更新知识库切片失败：{ex}")
