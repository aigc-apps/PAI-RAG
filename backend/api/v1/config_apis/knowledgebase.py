### Knowledgebase configuration API ###
from datetime import datetime, timezone
import time
import traceback
from typing import List, Optional
from common.knowledgebase.types import FileStatus
from fastapi import APIRouter, Depends, File, Query, UploadFile, Form
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession
from rag.file_item_utils import to_file_entity
from db.models.knowledgebase.chunk import KbChunkEntity, KbChunkModel
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.knowledgebase import (
    KbEntity,
    KnowledgebaseCreate,
)
from db.db_context import get_db_session
from sqlalchemy.exc import IntegrityError
from pairag.file.store.file_store_helper import file_store
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException
from service.injection import get_rag_service, get_file_service, get_chunk_service
from service.knowledgebase.rag_service import RagService
from service.knowledgebase.file_service import FileService
from service.knowledgebase.chunk_service import ChunkService
from loguru import logger
from utils.upload_file_utils import upload_form_files, upload_file_path_list

knowledgebase_router = APIRouter()


@knowledgebase_router.post("", response_model=ResponseModel[KbEntity])
async def create_knowledgebase(
    kb: KnowledgebaseCreate,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        knowledgebase = await rag_service.add_knowledgebase(kb)

        return success_response(data=knowledgebase, message="知识库创建成功。")
    except ValueError as e:
        logger.error(f"创建知识库失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except IntegrityError as e:
        logger.error(f"创建知识库失败。\nIntegrityError:{e}")
        if "UniqueViolationError" in str(e.orig):
            raise ApiException(code=400, message="创建知识库失败: 知识库名称已存在。")
        else:
            raise ApiException(code=400, message=f"创建知识库失败: {e}.")
    except Exception:
        logger.exception(f"创建知识库失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"创建知识库失败: {traceback.format_exc()}.")


@knowledgebase_router.get("")
async def list_knowledgebases(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    query: Optional[str] = None,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        paged_result = await rag_service.list_knowledgebases(page, size, query)
        return success_response(data=paged_result, message="获取知识库列表成功")
    except ValueError as e:
        logger.error(f"获取知识库列表失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"获取知识库列表失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"获取知识库列表失败: {traceback.format_exc()}.")


@knowledgebase_router.get("/{kb_id}", response_model=ResponseModel[KbEntity])
async def read_knowledgebase(
    kb_id: str,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        knowledgebase = await rag_service.get_knowledgebase(kb_id)
        return success_response(data=knowledgebase, message="查询知识库成功。")
    except ValueError as e:
        logger.error(f"查询知识库失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"查询知识库失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"查询知识库失败: {traceback.format_exc()}.")


@knowledgebase_router.put("/{kb_id}", response_model=ResponseModel[KbEntity])
async def update_knowledgebase(
    kb_id: str,
    new_kb: KnowledgebaseCreate,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        knowledgebase = await rag_service.update_knowledgebase(kb_id, new_kb)

        return success_response(data=knowledgebase, message="知识库更新成功。")
    except ValueError as e:
        logger.error(f"更新知识库失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"更新知识库失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"更新知识库失败: {traceback.format_exc()}.")


@knowledgebase_router.delete("/{kb_id}")
async def delete_knowledgebase(
    kb_id: str,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        await rag_service.delete_knowledgebase(kb_id)

        return success_response(data=None, message="知识库删除成功。")
    except ValueError as e:
        logger.error(f"删除知识库失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"删除知识库失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"删除知识库失败: {traceback.format_exc()}.")

@knowledgebase_router.get("/{kb_id}/files")
async def list_files(
    kb_id: str,
    file_name: Optional[str] = None,
    query: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        if file_name:
            file_entity = await rag_service.get_file_by_name(kb_id=kb_id, file_name=file_name)
            if not file_entity:
                raise ApiException.not_found(file_name, "文件")
            return success_response(data=file_entity, message="查询文件成功")
        else:
            page_result = await rag_service.list_files(kb_id, page, size, query, status)
            return success_response(data=page_result, message="查询文件列表成功")
    except ValueError as e:
        logger.error(f"查询文件失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"查询文件失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"查询文件失败: {traceback.format_exc()}.")


@knowledgebase_router.post("/{kb_id}/files")
async def upload_files(
    kb_id: str,
    files: Optional[List[UploadFile]] = File(...),
    file_path_list: Optional[List[str]] = Form(None),
    file_id_list: Optional[List[str]] = Form(None),
    file_sources: Optional[List[str]] = Form(None),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
    file_service: FileService = Depends(get_file_service),
):
    try:
        file_version = int(time.time())
        if not files and not file_id_list and not file_path_list:
            raise ApiException(code=400, message="没有上传任何文件。")
        _ = await rag_service.get_knowledgebase(kb_id=kb_id)
        if file_id_list:
            file_entities = await file_service.get_files_by_ids(file_ids=file_id_list)
            if not file_entities:
                raise ApiException.not_found(file_id_list, "文件ID列表")
        else:
            if files:
                file_items = upload_form_files(kb_id=kb_id, files=files)
            elif file_path_list:
                file_items = upload_file_path_list(kb_id=kb_id, file_path_list=file_path_list)

            file_paths = [file_item.file_path for file_item in file_items]
            file_entities = await file_service.get_files_by_path(kb_id=kb_id, file_paths=file_paths)

            file_entity_dict = {file_entity.file_path: file_entity for file_entity in file_entities}
            for file_item in file_items:
                if file_item.file_path not in file_entity_dict:
                    file_entity = to_file_entity(file_item)
                else:
                    file_entity.file_md5 = file_item.file_md5
                    file_entity.file_size = file_item.file_size
                    file_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

                file_entity.file_version = file_version
                file_entities.append(file_entity)

            if file_sources:
                assert len(file_sources) == len(file_entities), "文件来源列表长度与文件列表长度不一致"

            import app.worker as background_worker
            for i,file_entity in enumerate(file_entities):
                if file_sources:
                    file_entity.file_source = file_sources[i]
                background_worker.enqueue_file_tasks.delay(file_entity.id, file_entity.file_version, is_attachment=False)
                logger.info(f"Queued {file_entity.id} job successfully.")
                session.add(file_entity)

            logger.info(f"Uploaded {len(file_entities)} files successfully.")
            return success_response(data=file_entities, message="上传文件成功")
    except ValueError as e:
        logger.error(f"上传文件失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"上传文件失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"上传文件失败: {traceback.format_exc()}.")


@knowledgebase_router.get(
    "/{kb_id}/files/{file_id}", response_model=ResponseModel[KbFileEntity]
)
async def get_kb_file(
    kb_id: str,
    file_id: str,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        file_entity = await rag_service.get_file(kb_id=kb_id, file_id=file_id)
        if not file_entity:
            raise ApiException.not_found(file_id, "文件")

        file_url = file_store.get_url(file_entity.file_path)
        file_entity.file_metadata["file_url"] = file_url
        return success_response(data=file_entity, message="查询文件成功")
    except ValueError as e:
        logger.error(f"查询文件失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"查询文件失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"查询文件失败: {traceback.format_exc()}.")


@knowledgebase_router.put("/{kb_id}/files/{file_id}")
async def reprocess_file(
    kb_id: str,
    file_id: str,
    session: AsyncSession = Depends(get_db_session),
    file_service: FileService = Depends(get_file_service),
):
    try:
        file_entities = await file_service.get_files_by_ids(kb_id=kb_id, file_ids=[file_id])
        if not file_entities:
            raise ApiException.not_found(file_id, "文件")

        reprocessed_count = await _batch_reprocess_files(kb_id, file_entities, session)
        return success_response(data=reprocessed_count, message=f"成功将 {reprocessed_count} 个文件加入重新处理队列。")
    except ValueError as e:
        logger.error(f"重新处理文件失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"重新处理文件失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"重新处理文件失败: {traceback.format_exc()}.")


@knowledgebase_router.delete("/{kb_id}/files/{file_id}")
async def delete_file(
    kb_id: str,
    file_id: str,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        await rag_service.delete_file(kb_id=kb_id, file_id=file_id)
        return success_response(data=None, message="删除文件成功。")
    except ValueError as e:
        logger.error(f"删除文件失败。\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception:
        logger.error(f"删除文件失败。\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"删除文件失败: {traceback.format_exc()}.")



class BatchOperationRequest(BaseModel):
    operation: str = Field(..., description="操作类型: 'delete' 或 'reprocess'")
    file_id_list: List[str] = Field(..., description="要操作的文件ID列表")


@knowledgebase_router.post("/{kb_id}/files/batch", response_model=ResponseModel[dict])
async def batch_operations(
    kb_id: str,
    request: BatchOperationRequest,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
    file_service: FileService = Depends(get_file_service),
):
    """
    批量操作知识库中的文件
    支持的操作：
    - delete: 批量删除文件
    - reprocess: 批量重新处理文件
    """
    if not request.file_id_list:
        raise ApiException(code=400, message="文件ID列表不能为空。")

    if request.operation not in ["delete", "reprocess"]:
        raise ApiException(
            code=400,
            message=f"不支持的操作类型: {request.operation}。支持的操作: delete, reprocess"
        )

    # 验证所有文件是否存在
    file_entities = await file_service.get_files_by_ids(kb_id=kb_id, file_ids=request.file_id_list)
    found_file_ids = {entity.id for entity in file_entities}
    not_found_ids = [file_id for file_id in request.file_id_list if file_id not in found_file_ids]

    if not_found_ids:
        raise ApiException(
            code=404,
            message=f"以下文件在知识库{kb_id}中不存在: {', '.join(not_found_ids)}"
        )

    if request.operation == "delete":
        try:
            await rag_service.batch_delete_files(kb_id=kb_id, file_ids=request.file_id_list)
            return success_response(data=None, message="删除文件成功。")
        except ValueError as e:
            logger.error(f"删除文件失败。\nValueError:{e}")
            raise ApiException(code=400, message=str(e))
        except Exception:
            logger.error(f"删除文件失败。\nException:{traceback.format_exc()}")
            raise ApiException(code=400, message=f"删除文件失败: {traceback.format_exc()}.")
    elif request.operation == "reprocess":
        try:
            reprocessed_count = await _batch_reprocess_files(kb_id, file_entities, session)
            return success_response(data=reprocessed_count, message=f"成功将 {reprocessed_count} 个文件加入重新处理队列。")
        except ValueError as e:
            logger.error(f"重新处理文件失败。\nValueError:{e}")
            raise ApiException(code=400, message=str(e))
        except Exception:
            logger.error(f"重新处理文件失败。\nException:{traceback.format_exc()}")
            raise ApiException(code=400, message=f"重新处理文件失败: {traceback.format_exc()}.")


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

    for file_entity in file_entities:
        file_entity.status = FileStatus.pending
        file_entity.file_version = file_version
        session.add(file_entity)
        reprocessed_count += 1

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

    return reprocessed_count


class FileSourceParam(BaseModel):
    file_source: str = Field(default=None)



@knowledgebase_router.post("/{kb_id}/files/{file_id}/source", response_model=ResponseModel[KbFileEntity])
async def set_file_source(
    kb_id: str,
    file_id: str,
    body: FileSourceParam,
    session: AsyncSession = Depends(get_db_session),
):
    file_entity = await session.get(KbFileEntity, file_id)
    if not file_entity:
        raise ApiException.not_found(file_id, "文件")

    if not body.file_source:
        raise ApiException(code=400, message="文件来源不能为空。")

    file_entity.file_source = body.file_source
    session.add(file_entity)

    await session.refresh(file_entity)

    return success_response(data=file_entity, message="更新文件来源成功")


@knowledgebase_router.get("/{kb_id}/files/{file_id}/chunks")
async def list_chunks(
    kb_id: str,
    file_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        chunk_entities = await rag_service.list_chunks(kb_id=kb_id, file_id=file_id, page=page, size=size)

        return success_response(data=chunk_entities, message="获取切片列表成功")
    except Exception as ex:
        logger.error(f"Failed to list chunks for knowledgebase {kb_id} / file {file_id}: {ex}")
        raise ApiException(code=500, message=f"获取切片列表失败：{str(ex)}")


@knowledgebase_router.put("/{kb_id}/files/{file_id}/chunks/{chunk_id}", response_model=ResponseModel[KbChunkEntity])
async def update_chunk(
    kb_id: str,
    file_id: str,
    chunk_id: str,
    update_kb_chunk: KbChunkModel,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        kb_chunk = await rag_service.update_chunk(kb_id=kb_id, file_id=file_id, chunk_id=chunk_id, chunk=update_kb_chunk)
        return success_response(data=kb_chunk, message="更新知识库切片成功。")
    except Exception as ex:
        logger.error(f"Failed to update knowledgebase {kb_id} / file {file_id} / chunk {chunk_id}: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"更新知识库切片失败：{str(ex)}")


class AddChunkRequest(BaseModel):
    text: str = Field(..., description="Chunk text content")
    chunk_metadata: dict = Field(default={}, description="Chunk metadata")

@knowledgebase_router.delete("/{kb_id}/files/{file_id}/chunks/{chunk_id}")
async def delete_chunk(
    kb_id: str,
    file_id: str,
    chunk_id: str,
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        await rag_service.delete_chunk(chunk_id=chunk_id, kb_id=kb_id, file_id=file_id)
        return success_response(data=None, message="删除切片成功。")
    except Exception as ex:
        logger.error(f"Failed to delete chunk from knowledgebase {kb_id} / file {file_id} / chunk {chunk_id}: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"删除切片失败：{str(ex)}")

@knowledgebase_router.post("/{kb_id}/files/{file_id}/chunks", response_model=ResponseModel[KbChunkEntity])
async def add_chunk(
    kb_id: str,
    file_id: str,
    request: AddChunkRequest,
    session: AsyncSession = Depends(get_db_session),
    chunk_service: ChunkService = Depends(get_chunk_service),
    rag_service: RagService = Depends(get_rag_service),
):
    """
    Add a new chunk to a file.

    - chunk_text: The text content of the chunk
    - active: Defaults to True
    - chunk_index: Automatically set to max(index) + 1 for the file
    - chunk_metadata: Combines file_metadata + token_count
    """
    try:
        new_chunk = await rag_service.add_chunk(kb_id=kb_id, file_id=file_id, text=request.text, chunk_metadata=request.chunk_metadata)

        return success_response(data=new_chunk, message="添加切片成功。")
    except Exception as ex:
        logger.exception(f"Failed to add chunk to knowledgebase {kb_id} / file {file_id}: {ex}")
        raise ApiException(code=500, message=f"添加切片失败：{str(ex)}")
