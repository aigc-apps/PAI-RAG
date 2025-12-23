### Embedding configuration API ###
import time
import traceback
import asyncio
from fastapi import APIRouter, File, UploadFile, Form, Depends
from db.models.knowledgebase.knowledgebase import (
    KnowledgebaseCreate,
)
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import get_db_session
from rag.file_item_utils import to_file_entity
from common.chat.response_model import success_response
from common.knowledgebase.types import FileStatus
from datetime import datetime, timezone
from api.api_exception import ApiException
from service.knowledgebase.rag_service import RagService
from service.injection import get_rag_service, get_embedding_service, get_knowledgebase_service, get_file_service, get_file_task_service, get_tenant_id
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from service.knowledgebase.file_service import FileService
from service.knowledgebase.file_task_service import FileTaskService
from service.model.embedding_service import EmbeddingService
from common.knowledgebase.constants import ATTACHMENT_KNOWLEDGEBASE_NAME
from utils.upload_file_utils import upload_form_files_async
from loguru import logger

attachments_router = APIRouter()


MAX_CHECK_ATTEMPTS = 100
CHECK_INTERVAL = 3

@attachments_router.post("")
async def create_attachment_file(
    file_id: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    knowledgebase_service: KnowledgebaseService = Depends(get_knowledgebase_service),
    file_service: FileService = Depends(get_file_service),
    file_task_service: FileTaskService = Depends(get_file_task_service),
    rag_service: RagService = Depends(get_rag_service),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        knowledgebase = await knowledgebase_service.get_knowledgebase_by_name(ATTACHMENT_KNOWLEDGEBASE_NAME, tenant_id=tenant_id)
        default_embedding_config = await embedding_service.get_default_embedding(tenant_id=tenant_id)
        file_version = int(time.time())

        if not knowledgebase:
            logger.info(f"Creating default_attachments knowledgebase for tenant {tenant_id}")
            kb_create = KnowledgebaseCreate(
                name=ATTACHMENT_KNOWLEDGEBASE_NAME,
                description="附件知识库",
                embedding_model=default_embedding_config.model_id,
            )
            knowledgebase = await knowledgebase_service.create_knowledgebase(kb_data=kb_create, tenant_id=tenant_id)
            await session.commit() # commit for background worker to use the knowledgebase id
            await session.refresh(knowledgebase)  # refresh to ensure knowledgebase is persisted
            logger.info(f"Created default_attachments knowledgebase {knowledgebase.id} for tenant {tenant_id}")
        else:
            logger.info(f"Found existing default_attachments knowledgebase {knowledgebase.id} for tenant {tenant_id}")

        import app.worker as background_worker

        file_items = await upload_form_files_async(kb_id=knowledgebase.id, files=[file], tenant_id=tenant_id)

        file_item = file_items[0]
        file_entity = await file_service.get_file(kb_id=knowledgebase.id, file_id=file_id, tenant_id=tenant_id)

        if not file_entity:
            file_entity = to_file_entity(file_item=file_item)
            file_entity.id = file_id
            await file_service.create_file(file_data=file_entity, tenant_id=tenant_id)
        else:
            file_entity.file_md5 = file_item.file_md5
            file_entity.file_size = file_item.file_size
            file_entity.file_version = file_version
            file_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await file_service.update_file(file_id=file_entity.id, kb_id=knowledgebase.id, new_entity=file_entity, tenant_id=tenant_id)

        session.add(file_entity)
        await session.commit()

        background_worker.enqueue_attachments_file_tasks.delay(file_entity.id, file_entity.file_version, file_entity.file_extension, is_attachment=True, tenant_id=tenant_id)
        logger.info(f"Enqueued file {file_entity.id} for background processing...")
    except Exception as e:
        logger.error(f"Failed to save file {file_id} to database: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"文件{file_id}上传失败: {e}")

    attempt = 0
    while attempt < MAX_CHECK_ATTEMPTS:
        await asyncio.sleep(CHECK_INTERVAL)
        attempt += 1
        logger.info(f"Checking file {file_item.file_name} processing status... Attempt {attempt} of {MAX_CHECK_ATTEMPTS}")
        await session.refresh(file_entity)
        if file_entity.status == FileStatus.succeeded:
            logger.info(f"File {file_item.file_name} processing completed successfully")
            return success_response(data=file_entity, message=f"文件{file_item.file_name}上传成功")
        elif file_entity.status == FileStatus.failed:
            logger.error(f"File {file_item.file_name} processing failed: {file_entity.failed_reason}.")
            raise ApiException(code=500, message=f"上传失败, 错误信息: {file_entity.failed_reason}")

    # Cancel task when timeouts
    file_entity.status = FileStatus.cancelled
    file_entity.failed_reason = f"文件{file_item.file_name}上传超时。"
    try:
        await file_service.update_file(file_id=file_entity.id, kb_id=knowledgebase.id, new_entity=file_entity, tenant_id=tenant_id)
        await session.commit()
    except Exception:
        logger.error(f"Failed to save file {file_item.file_name} to database: {traceback.format_exc()}")
        await session.rollback()

    raise ApiException(code=400, message=f"文件{file_id}上传超时。")
