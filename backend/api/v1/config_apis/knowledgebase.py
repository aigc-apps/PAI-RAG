### Knowledgebase configuration API ###
from datetime import datetime, timezone
import time
import traceback
from typing import List, Optional
from common.knowledgebase.types import FileStatus
from fastapi import APIRouter, Depends, File, Query, UploadFile, Form, Body
import json
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession
from rag.file_item_utils import to_file_entity
from db.models.knowledgebase.chunk import KbChunkEntity, KbChunkModel
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.knowledgebase import (
    KbEntity,
    KnowledgebaseCreate,
)
from pairag.file.nodeparsers.file_parser import ChunkConfig
from db.db_context import get_db_session
from pairag.file.store.file_store_helper import file_store
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException, handle_api_exceptions
from service.injection import get_rag_service, get_file_service, get_chunk_service, get_tenant_id, get_knowledgebase_service
from service.knowledgebase.rag_service import RagService
from service.knowledgebase.file_service import FileService
from service.knowledgebase.chunk_service import ChunkService
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from loguru import logger
from utils.list_api_utils import parse_comma_separated_list
from utils.upload_file_utils import upload_form_files_async, upload_file_names_async, StartParseTaskRequest
from common.i18n import i18n

knowledgebase_router = APIRouter()


@knowledgebase_router.delete("/metadata-schema-cache")
@handle_api_exceptions(action="clear metadata schema cache")
async def clear_metadata_schema_cache(
    tenant_id: str = Depends(get_tenant_id),
):
    """Clear metadata schema cache for the current tenant.
    The background task will automatically repopulate the cache."""
    from service.cache.metadata_schema_cache import metadata_schema_cache
    count = await metadata_schema_cache.clear_cache_by_tenant(tenant_id)
    return success_response(
        data={"cleared": count},
        message="Metadata schema cache cleared.",
    )


@knowledgebase_router.post("", response_model=ResponseModel[KbEntity])
async def create_knowledgebase(
    kb_data: KnowledgebaseCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
    knowledgebase_service: KnowledgebaseService = Depends(get_knowledgebase_service),
):
    knowledgebase = None
    kb_id = None
    try:
        knowledgebase = await rag_service.create_knowledgebase(kb_data=kb_data, tenant_id=tenant_id)
        kb_id = knowledgebase.id
        await session.commit()
        await knowledgebase_service.write_cache_after_commit(knowledgebase, tenant_id)
        return success_response(data=knowledgebase, message=i18n.t("api.knowledgebase.create_success"))
    except ValueError as e:
        logger.error(f"Failed to create knowledge base.\nValueError:{e}")
        await session.rollback()
        if kb_id:
            await knowledgebase_service.delete_cache_on_rollback(kb_id, tenant_id, kb_data.name)
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.exception(f"Failed to create knowledge base.\nException:{traceback.format_exc()}")
        await session.rollback()
        if kb_id:
            await knowledgebase_service.delete_cache_on_rollback(kb_id, tenant_id, kb_data.name)
        raise ApiException(code=400, message=i18n.t("api.knowledgebase.create_failed", error=str(e)))

@knowledgebase_router.get("")
@handle_api_exceptions(action="list knowledgebases", i18n_error_key="api.knowledgebase.list_failed", default_code=400)
async def list_knowledgebases(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    query: Optional[str] = None,
    ids: Optional[str]=Query(default=None, description="IDs separated by comma, e.g. abc,123"),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
    knowledgebase_service: KnowledgebaseService = Depends(get_knowledgebase_service),
):
    if not ids:
        paged_result = await rag_service.list_knowledgebases(tenant_id=tenant_id, page=page, size=size, query=query)
        return success_response(data=paged_result, message=i18n.t("api.knowledgebase.list_success"))
    kb_ids = parse_comma_separated_list(ids)
    total_result = await knowledgebase_service.get_knowledgebases_by_ids(tenant_id=tenant_id, kb_ids=kb_ids)
    return success_response(data=total_result, message=i18n.t("api.knowledgebase.list_success"))


@knowledgebase_router.get("/{kb_id}", response_model=ResponseModel[KbEntity])
@handle_api_exceptions(action="query knowledgebase", i18n_error_key="api.knowledgebase.query_failed", default_code=400)
async def read_knowledgebase(
    kb_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ApiException.not_found(kb_id, "Knowledgebase")
    return success_response(data=knowledgebase, message=i18n.t("api.knowledgebase.query_success"))


@knowledgebase_router.put("/{kb_id}", response_model=ResponseModel[KbEntity])
async def update_knowledgebase(
    kb_id: str,
    update_data: KnowledgebaseCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
    knowledgebase_service: KnowledgebaseService = Depends(get_knowledgebase_service),
):
    knowledgebase = None
    old_kb_name = None
    new_kb_name = None  # ← 新增：提前保存名称

    try:
        # ===== 1. commit 前取出旧名称 =====
        old_kb = await knowledgebase_service.get_knowledgebase(kb_id, tenant_id)
        if old_kb:
            old_kb_name = str(old_kb.name)  # ← 立即取出，转为纯字符串

        # ===== 2. 更新 =====
        knowledgebase = await rag_service.update_knowledgebase(
            kb_id=kb_id, update_data=update_data, tenant_id=tenant_id
        )

        # ===== 3. commit 前取出新名称 =====
        new_kb_name = str(knowledgebase.name)  # ← commit 前取出！

        # ===== 4. commit（之后 ORM 属性全部 expired）=====
        await session.commit()

        # ===== 5. commit 后只用纯字符串，不碰 ORM 属性 =====
        await knowledgebase_service.write_cache_after_commit(knowledgebase, tenant_id)

        return success_response(
            data=knowledgebase,
            message=i18n.t("api.knowledgebase.update_success"),
        )

    except ValueError as e:
        logger.error(f"Failed to update knowledge base.\nValueError:{e}")
        await session.rollback()
        # ===== 6. rollback 后也只用纯字符串 =====
        kb_name = new_kb_name or old_kb_name
        if kb_name:
            await knowledgebase_service.delete_cache_on_rollback(
                kb_id, tenant_id, kb_name  # ← 用字符串，不用 .name
            )
        raise ApiException(code=400, message=str(e))

    except Exception as e:
        logger.error(f"Failed to update knowledge base.\nException:{traceback.format_exc()}")
        await session.rollback()
        kb_name = new_kb_name or old_kb_name
        if kb_name:
            await knowledgebase_service.delete_cache_on_rollback(
                kb_id, tenant_id, kb_name  # ← 用字符串，不用 .name
            )
        raise ApiException(code=400, message=i18n.t(
            "api.knowledgebase.update_failed", error=str(e)
        ))

@knowledgebase_router.delete("/{kb_id}")
@handle_api_exceptions(action="delete knowledgebase", default_code=400)
async def delete_knowledgebase(
    kb_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    await rag_service.delete_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    return success_response(data=None, message=i18n.t("api.knowledgebase.delete_success"))

@knowledgebase_router.get("/{kb_id}/files")
@handle_api_exceptions(action="list files", default_code=400)
async def list_files(
    kb_id: str,
    file_name: Optional[str] = None,
    query: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    if file_name:
        file_entity = await rag_service.get_file_by_name(kb_id=kb_id, file_name=file_name, tenant_id=tenant_id)
        if not file_entity:
            raise ApiException.not_found(file_name, i18n.t("api.knowledgebase.file_resource_name"))
        return success_response(data=file_entity, message=i18n.t("api.knowledgebase.file_query_success"))
    page_result = await rag_service.list_files(kb_id=kb_id, tenant_id=tenant_id, page=page, size=size, query=query, status=status, source=source)
    return success_response(data=page_result, message=i18n.t("api.knowledgebase.file_list_success"))


@knowledgebase_router.get("/{kb_id}/file-content")
@handle_api_exceptions(action="get file content")
async def get_file_content(
    kb_id: str,
    file_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    max_chars: Optional[int] = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    """Return a file's full text + metadata, by file_id or by data-source doc_id.
    Powers the 召回测试 page's "查看文件" (fetch) tool."""
    from sqlmodel import select
    from db.models.knowledgebase.datasource import DataSourceDocumentEntity

    resolved = file_id
    if not resolved and doc_id:
        result = await session.exec(
            select(DataSourceDocumentEntity.file_id).where(
                DataSourceDocumentEntity.kb_id == kb_id,
                DataSourceDocumentEntity.doc_id == doc_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
            )
        )
        # fall back to treating doc_id as a file_id (search results expose
        # doc_id == file_id when source_doc_id is absent)
        resolved = result.first() or doc_id
    if not resolved:
        raise ApiException.not_found(doc_id or file_id or "", "File")
    data = await rag_service.get_file_content(
        kb_id=kb_id, file_id=resolved, tenant_id=tenant_id, max_chars=max_chars, offset=offset,
    )
    if not data:
        raise ApiException.not_found(resolved, "File")
    return success_response(data=data, message="Get file content success.")



# 启动解析任务
@knowledgebase_router.post("/{kb_id}/files/parse", response_model=ResponseModel[List[KbFileEntity]])
async def start_parse_task(
    kb_id: str,
    parse_request: StartParseTaskRequest,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
    file_service: FileService = Depends(get_file_service),
):
    logger.info(f"Start parsing task for knowledgebase {kb_id} with tenant {tenant_id}, parse_request: {parse_request}")
    try:
        kb_entity = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
        if not kb_entity:
            raise ValueError(i18n.t("api.knowledgebase.not_found", id=kb_id))

        # Validate chunk_config if provided
        if parse_request.chunk_config:
            try:
                ChunkConfig.model_validate(parse_request.chunk_config)
            except Exception as e:
                raise ApiException(code=400, message=f"chunk_config format is incorrect: {e}")

        file_items = await upload_file_names_async(kb_id=kb_id, parse_tasks=parse_request.files, tenant_id=tenant_id)

        file_version = int(time.time())
        file_names = [file_task.file_name for file_task in parse_request.files]
        existing_file_entities = await rag_service.get_files_by_names(
            kb_id=kb_id, file_names=file_names, tenant_id=tenant_id
        )
        existing_file_dict = { entity.file_name: entity for entity in existing_file_entities }

        import app.worker as background_worker
        file_entities = []
        for file_item in file_items:
            if file_item.file_name in existing_file_dict:
                file_entity = existing_file_dict[file_item.file_name]
                file_entity.file_md5 = file_item.file_md5
                file_entity.file_size = file_item.file_size
                file_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            else:
                file_entity = to_file_entity(file_item=file_item)

            # Update chunk_config if provided
            if parse_request.chunk_config:
                file_entity.chunk_config = parse_request.chunk_config

            file_entity.file_version = file_version
            session.add(file_entity)
            await session.commit()
            await session.refresh(file_entity)
            background_worker.enqueue_file_tasks.delay(file_entity.id, file_entity.file_version, is_attachment=False, tenant_id=tenant_id)
            file_entities.append(file_entity)

        logger.info(f"Uploaded {len(file_entities)} files successfully.")
        return success_response(data=file_entities, message=i18n.t("api.knowledgebase.parse_task_start_success"))
    except ApiException:
        raise
    except ValueError as e:
        logger.error(f"Failed to start parse task.\nValueError:{traceback.format_exc()}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to start parse task.\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.knowledgebase.parse_task_start_failed", error=str(e)))



@knowledgebase_router.post("/{kb_id}/files")
async def upload_files(
    kb_id: str,
    auto_parse: bool = Query(default=True),
    files: Optional[List[UploadFile]] = File(...),
    file_sources: Optional[List[str]] = Form(None),
    chunk_config: Optional[str] = Form(None, description="JSON string of chunk_config, shared by all files in this upload"),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
    file_service: FileService = Depends(get_file_service),
    knowledgebase_service: KnowledgebaseService = Depends(get_knowledgebase_service),
):
    try:
        file_version = int(time.time())
        if not files:
            raise ApiException(code=400, message=i18n.t("api.error.no_files"))


        knowledgebase = await knowledgebase_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
        if not knowledgebase:
            raise ApiException.not_found(kb_id, "Knowledgebase")

        kb_chunk_config = knowledgebase.chunk_config


        parsed_chunk_config = None
        if chunk_config:
            try:
                parsed_chunk_config = json.loads(chunk_config)
                if not isinstance(parsed_chunk_config, dict):
                    raise ValueError("chunk_config must be a JSON object")
                # Validate chunk_config
                ChunkConfig.model_validate(parsed_chunk_config)
            except json.JSONDecodeError as e:
                raise ApiException(code=400, message=f"chunk_config format is incorrect: {e}")
            except Exception as e:
                raise ApiException(code=400, message=i18n.t("api.knowledgebase.chunk_config_validation_failed", error=str(e)))

        file_items = await upload_form_files_async(kb_id=kb_id, files=files, tenant_id=tenant_id)

        file_names = [file_item.file_name for file_item in file_items]
        existing_file_entities = await file_service.get_files_by_names(kb_id=kb_id, file_names=file_names, tenant_id=tenant_id)

        existing_file_entity_dict = {file_entity.file_name: file_entity for file_entity in existing_file_entities}
        new_file_entities = []
        for file_item in file_items:
            if file_item.file_name not in existing_file_entity_dict:
                file_entity = to_file_entity(file_item=file_item)
            else:
                file_entity = existing_file_entity_dict[file_item.file_name]
                file_entity.file_md5 = file_item.file_md5
                file_entity.file_size = file_item.file_size
                file_entity.status = FileStatus.pending
                file_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            file_entity.file_version = file_version
            new_file_entities.append(file_entity)

        if file_sources:
            assert len(file_sources) == len(new_file_entities), i18n.t("api.knowledgebase.file_source_mismatch")

        # Apply chunk_config to all files if provided (shared by all files in this upload)
        for i, file_entity in enumerate(new_file_entities):
            if file_sources:
                file_entity.file_source = file_sources[i]

            # All files share the same chunk_config if provided
            if parsed_chunk_config:
                file_entity.chunk_config = parsed_chunk_config

            session.add(file_entity)

        # Commit first to ensure file records exist in DB before queueing tasks
        # This prevents race condition where Celery workers can't find uncommitted records
        await session.commit()

        # Queue background tasks after commit to avoid SQLite race condition
        if auto_parse:
            import app.worker as background_worker
            for file_entity in new_file_entities:
                background_worker.enqueue_file_tasks.delay(file_entity.id, file_entity.file_version, is_attachment=False, tenant_id=tenant_id)
                logger.info(f"Queued {file_entity.id} job successfully.")


        for file_entity in new_file_entities:
            await session.refresh(file_entity)


        response_entities = []
        for file_entity in new_file_entities:
            file_dict = file_entity.model_dump()
            if not file_entity.chunk_config:
                file_dict["chunk_config"] = kb_chunk_config
            response_entities.append(file_dict)

        logger.info(f"Uploaded {len(new_file_entities)} files successfully.")
        return success_response(data=response_entities, message="File upload successful")
    except ValueError as e:
        logger.error(f"File upload failed.\nValueError:{e}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"File upload failed.\nException:{traceback.format_exc()}")
        raise ApiException(code=400, message=f"File upload failed: {e}.")

@knowledgebase_router.get(
    "/{kb_id}/files/{file_id}", response_model=ResponseModel[KbFileEntity]
)
@handle_api_exceptions(action="get file", default_code=400)
async def get_kb_file(
    kb_id: str,
    file_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    file_entity = await rag_service.get_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
    if not file_entity:
        raise ApiException.not_found(file_id, "File")

    file_url = await file_store.get_url_async(file_path=file_entity.file_path, tenant_id=tenant_id)
    file_entity.file_metadata["file_url"] = file_url
    return success_response(data=file_entity, message="File query successful")


class ReprocessFileRequest(BaseModel):
    chunk_config: Optional[dict] = Field(default=None, description="Optional chunk configuration for the file")


@knowledgebase_router.put("/{kb_id}/files/{file_id}")
@handle_api_exceptions(action="reprocess file", default_code=400)
async def reprocess_file(
    kb_id: str,
    file_id: str,
    body: Optional[ReprocessFileRequest] = Body(None),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    file_service: FileService = Depends(get_file_service),
    rag_service: RagService = Depends(get_rag_service),
):
    file_entities = await file_service.get_files_by_ids(file_ids=[file_id], tenant_id=tenant_id)
    if not file_entities:
        raise ApiException.not_found(file_id, "File")

    # 如果提供了 chunk_config，先验证并更新
    chunk_config = None
    if body and body.chunk_config:
        try:
            ChunkConfig.model_validate(body.chunk_config)
            chunk_config = body.chunk_config
        except Exception as e:
            raise ApiException(code=400, message=f"chunk_config format is incorrect: {e}")

    reprocessed_count = await _batch_reprocess_files(
        kb_id=kb_id,
        file_entities=file_entities,
        session=session,
        tenant_id=tenant_id,
        chunk_config=chunk_config
    )
    return success_response(data=reprocessed_count, message=f"Successfully added {reprocessed_count} files to the reprocessing queue.")


@knowledgebase_router.delete("/{kb_id}/files/{file_id}")
@handle_api_exceptions(action="delete file", default_code=400)
async def delete_file(
    kb_id: str,
    file_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    await rag_service.delete_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
    return success_response(data=None, message="File deletion successful.")


class BatchOperationRequest(BaseModel):
    operation: str = Field(..., description="Operation type: 'delete' or 'reprocess'")
    file_id_list: List[str] = Field(..., description="List of file IDs to be operated on")
    chunk_config: Optional[dict] = Field(default=None, description="Optional chunk configuration for reprocess operation, shared by all files")


@knowledgebase_router.post("/{kb_id}/files/batch", response_model=ResponseModel[dict])
async def batch_operations(
    kb_id: str,
    request: BatchOperationRequest,
    tenant_id: str = Depends(get_tenant_id),
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
        raise ApiException(code=400, message="File ID list cannot be empty.")

    if request.operation not in ["delete", "reprocess"]:
        raise ApiException(
            code=400,
            message=f"Unsupported operation type: {request.operation}. Supported operations: delete, reprocess"
        )

    # 验证所有文件是否存在
    file_entities = await file_service.get_files_by_ids(file_ids=request.file_id_list, tenant_id=tenant_id)
    found_file_ids = {entity.id for entity in file_entities}
    not_found_ids = [file_id for file_id in request.file_id_list if file_id not in found_file_ids]

    if not_found_ids:
        raise ApiException(
            code=404,
            message=f"Files not found in knowledge base {kb_id}: {', '.join(not_found_ids)}"
        )

    if request.operation == "delete":
        try:
            await rag_service.batch_delete_files(kb_id=kb_id, file_ids=request.file_id_list, tenant_id=tenant_id)
            return success_response(data=None, message="File deletion successful.")
        except ValueError as e:
            logger.error(f"File deletion failed.\nValueError:{e}")
            raise ApiException(code=400, message=str(e))
        except Exception as e:
            logger.error(f"File deletion failed. \nException:{traceback.format_exc()}")
            raise ApiException(code=400, message=f"File deletion failed: {e}.")
    elif request.operation == "reprocess":
        try:
            chunk_config = None
            if request.chunk_config:
                try:
                    ChunkConfig.model_validate(request.chunk_config)
                    chunk_config = request.chunk_config
                except Exception as e:
                    raise ApiException(code=400, message=f"chunk_config format is incorrect: {e}")

            reprocessed_count = await _batch_reprocess_files(
                kb_id=kb_id,
                file_entities=file_entities,
                session=session,
                tenant_id=tenant_id,
                chunk_config=chunk_config
            )
            return success_response(data=reprocessed_count, message=f"Successfully added {reprocessed_count} files to the reprocessing queue.")
        except ApiException:
            raise
        except ValueError as e:
            logger.error(f"Reprocess file failed.\nValueError:{e}")
            raise ApiException(code=400, message=str(e))
        except Exception as e:
            logger.error(f"Reprocess file failed.\nException:{traceback.format_exc()}")
            raise ApiException(code=400, message=f"Reprocess file failed: {e}.")


async def _batch_reprocess_files(
    kb_id: str,
    file_entities: List[KbFileEntity],
    session: AsyncSession,
    tenant_id: str,
    chunk_config: Optional[dict] = None,
) -> ResponseModel[dict]:
    """
    批量重新处理文件的内部实现
    如果提供了 chunk_config，会在重新解析之前更新所有文件的 chunk_config
    """
    import app.worker as background_worker

    file_version = int(time.time())
    reprocessed_count = 0

    for file_entity in file_entities:
        file_entity.status = FileStatus.pending
        file_entity.file_version = file_version
        file_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # 如果提供了 chunk_config，更新文件的 chunk_config
        if chunk_config:
            file_entity.chunk_config = chunk_config
            logger.info(f"Updated chunk_config for file {file_entity.id} before reprocessing.")

        session.add(file_entity)
        reprocessed_count += 1

    # 先提交所有文件状态更新，再入队处理任务
    await session.commit()

    for file_entity in file_entities:
        await session.refresh(file_entity)
        background_worker.enqueue_file_tasks.delay(
            file_entity.id,
            file_entity.file_version,
            is_attachment=False,
            tenant_id=tenant_id,
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
    tenant_id: str = Depends(get_tenant_id),
    file_service: FileService = Depends(get_file_service),
    session: AsyncSession = Depends(get_db_session),
):
    file_entity = await file_service.get_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
    if not file_entity:
        raise ApiException.not_found(file_id, "File")

    if not body.file_source:
        raise ApiException(code=400, message="File source cannot be empty.")

    file_entity.file_source = body.file_source
    logger.info(f"Set file source: {file_entity.id} -> {body.file_source}")
    session.add(file_entity)
    await session.commit()
    await session.refresh(file_entity)

    return success_response(data=file_entity, message="File source updated successfully")


@knowledgebase_router.get("/{kb_id}/files/{file_id}/chunks")
@handle_api_exceptions(action="list chunks", default_code=500)
async def list_chunks(
    kb_id: str,
    file_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    chunk_entities = await rag_service.list_chunks(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id, page=page, size=size)
    return success_response(data=chunk_entities, message="Chunk list retrieval successful")


@knowledgebase_router.put("/{kb_id}/files/{file_id}/chunks/{chunk_id}", response_model=ResponseModel[KbChunkEntity])
@handle_api_exceptions(action="update chunk", default_code=500)
async def update_chunk(
    kb_id: str,
    file_id: str,
    chunk_id: str,
    update_kb_chunk: KbChunkModel,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    kb_chunk = await rag_service.update_chunk(kb_id=kb_id, file_id=file_id, chunk_id=chunk_id, chunk=update_kb_chunk, tenant_id=tenant_id)
    return success_response(data=kb_chunk, message="Chunk update successful")


class AddChunkRequest(BaseModel):
    text: str = Field(..., description="Chunk text content")
    chunk_metadata: dict = Field(default={}, description="Chunk metadata")

@knowledgebase_router.delete("/{kb_id}/files/{file_id}/chunks/{chunk_id}")
@handle_api_exceptions(action="delete chunk", default_code=500)
async def delete_chunk(
    kb_id: str,
    file_id: str,
    chunk_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    await rag_service.delete_chunk(chunk_id=chunk_id, kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
    return success_response(data=None, message="Chunk deletion successful.")

@knowledgebase_router.post("/{kb_id}/files/{file_id}/chunks", response_model=ResponseModel[KbChunkEntity])
@handle_api_exceptions(action="add chunk", default_code=500)
async def add_chunk(
    kb_id: str,
    file_id: str,
    request: AddChunkRequest,
    tenant_id: str = Depends(get_tenant_id),
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
    new_chunk = await rag_service.add_chunk(kb_id=kb_id, file_id=file_id, text=request.text, chunk_metadata=request.chunk_metadata, tenant_id=tenant_id)
    return success_response(data=new_chunk, message="Chunk addition successful.")
