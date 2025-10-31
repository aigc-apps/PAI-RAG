### Embedding configuration API ###
import time
import uuid
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
from rag.kb_file_client import kb_file_client
from sqlmodel import select
from db.models.knowledgebase.embedding import (
    EmbeddingModelEntity,
)
from config.providers.config_change_manager import config_change_manager
from db.models.change_event import ChangeEventSource, ChangeEventType
from rag.chunk_helper import update_file_status_async

from loguru import logger

attachments_router = APIRouter()



@attachments_router.post("")
async def create_attachment_file(
    file_id: str = Form(...), file: UploadFile = File(...), session: AsyncSession = Depends(get_session)
):
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
    # excel附件不入知识库
    if file_entity.file_extension not in ['.xlsx']:
        await kb_file_client.process_file_async(file_task_entity.id, is_attachment=True)
    else:
        await update_file_status_async(
                file_id=file_item.id, task_id=file_task_entity.id, status=FileStatus.succeeded, is_attachment=True, file_item=file_item
            )
    await session.refresh(file_entity)
    if file_entity.status == FileStatus.succeeded:
        return success_response(data=file_entity, message="文件上传成功")
    else:
        return error_response(data=file_entity, message="文件上传失败")
