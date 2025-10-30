### Embedding configuration API ###
import time
import uuid
import asyncio
from db.models.knowledgebase.file_task import KbFileTaskEntity
from fastapi import APIRouter, File, UploadFile, Form, Depends
from db.models.knowledgebase.knowledgebase import (
    ChunkConfig,
    KbEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from rag.split.excel_split import convert_xls_to_xlsx
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import get_session
from config.providers.knowledgebase_provider import knowledgebase_provider
from pairag.file.store.file_store_helper import file_store
from pairag.file.models.file_item import FileItem
from rag.file_item_utils import to_file_entity
from db.models.knowledgebase.file import KbFileEntity
from api.response_model import success_response, error_response
from common.knowledgebase.types import FileStatus
from sqlmodel import select
from db.models.knowledgebase.embedding import (
    EmbeddingModelEntity,
)
from config.providers.config_change_manager import config_change_manager
from db.models.change_event import ChangeEventSource, ChangeEventType
from loguru import logger

attachments_router = APIRouter()


MAX_CHECK_ATTEMPTS = 60
CHECK_INTERVAL = 5


@attachments_router.post("")
async def create_attachment_file(
    file_id: str = Form(...), file: UploadFile = File(...), session: AsyncSession = Depends(get_session)
):
    import app.worker as background_worker

    knowledgebase = knowledgebase_provider.get_knowledgebase_by_name("default_attachments")
    default_embedding_results = await session.exec(select(EmbeddingModelEntity).where(EmbeddingModelEntity.is_default == True)) # noqa: E712
    default_embedding_entities = default_embedding_results.all()
    if len(default_embedding_entities) > 0:
        default_embedding_entity = default_embedding_entities[0]
        logger.info(f"Default embedding model found, and using {default_embedding_entity.model_id} for attachment knowledgebase.")
    else:
        all_embedding_results = await session.exec(select(EmbeddingModelEntity))
        all_embedding_entities = all_embedding_results.all()
        default_embedding_entity = all_embedding_entities[0]
        logger.info(f"No default embedding model was found, and using {default_embedding_entity.model_id} for attachment knowledgebase.")

    if not knowledgebase:
        kb = KnowledgebaseCreate(
            name="default_attachments",
            description="附件知识库",
            embedding_model=default_embedding_entity.model_id,
        )
        kb.chunk_config = (ChunkConfig()).model_dump()
        kb.retrieval_config = (RetrievalConfig()).model_dump()
        knowledgebase = KbEntity.model_validate(kb)
        session.add(knowledgebase)
        await session.commit()
        await session.refresh(knowledgebase)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.KNOWLEDGEBASE,
            event_type=ChangeEventType.ADD,
            source_id=knowledgebase.id,
        )

    # Save file to local storage
    file_name = file.filename
    file_data = file.file
    if file.filename.endswith(".xls"):
        file_data = convert_xls_to_xlsx(file_data)
        file_name = file_name[:-4] + ".xlsx"


    destination_file_path = f"{knowledgebase.name}/docs/{file_name}"
    file_store.save(
        file=file_data,
        file_path=destination_file_path,
    )
    file_item = FileItem.from_file(
        file=file_data,
        file_path=destination_file_path,
        kb_id=knowledgebase.id,
    )
    file_item.id = file_id
    file_entity : KbFileEntity = to_file_entity(file_item)
    file_entity.file_version = int(time.time())
    file_task_entity = KbFileTaskEntity(
        id=uuid.uuid4().hex,
        file_id=file_entity.id,
        status=FileStatus.pending,
        file_version=file_entity.file_version,
        kb_id=file_entity.kb_id,
        file_part=0,
        file_path=file_entity.file_path,
    )
    session.add(file_entity)
    session.add(file_task_entity)
    await session.commit()

    background_worker.enqueue_file_tasks.delay(file_entity.id, file_entity.file_version, is_attachment=True)
    logger.info(f"Enqueued file {file_item.file_name} for background processing...")
    attempt = 0
    while attempt < MAX_CHECK_ATTEMPTS:
        attempt += 1
        logger.info(f"Checking file {file_item.file_name} processing status... Attempt {attempt} of {MAX_CHECK_ATTEMPTS}")
        await session.refresh(file_entity)
        if file_entity.status == FileStatus.succeeded:
            logger.info(f"File {file_item.file_name} processing completed successfully")
            return success_response(data=file_entity, message=f"文件{file_item.file_name}上传成功")
        elif file_entity.status == FileStatus.failed:
            logger.error(f"File {file_item.file_name} processing failed: {file_entity.failed_reason}.")
            return error_response(code=500, data=file_entity, message=f"文件{file_item.file_name}上传失败")
        await asyncio.sleep(CHECK_INTERVAL)



    return error_response(code=400, data=file_entity, message=f"文件{file_item.file_name}上传超时。")
