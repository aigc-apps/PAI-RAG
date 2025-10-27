### Embedding configuration API ###
from fastapi import APIRouter, File, UploadFile, Form, Depends
from db.models.knowledgebase.knowledgebase import (
    ChunkConfig,
    KbEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import get_session
from config.providers.knowledgebase_provider import knowledgebase_provider
from pairag.file.store.file_store_helper import file_store
from pairag.file.models.file_item import FileItem
from rag.file_item_utils import to_file_entity
from db.models.knowledgebase.file import KbFileEntity
from api.response_model import success_response, error_response
from common.knowledgebase.types import FileStatus
from rag.knowledgebase_tool import kb_client
from sqlmodel import select
from db.models.knowledgebase.embedding import (
    EmbeddingModelEntity,
)
from config.providers.config_change_manager import config_change_manager
from db.models.change_event import ChangeEventSource, ChangeEventType

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

    file_name = file.filename
    destination_file_path = f"{knowledgebase.name}/docs/{file_name}"
    file_store.save(
        file=file.file,
        file_path=destination_file_path,
    )
    file_item = FileItem.from_file(
        file=file.file,
        file_path=destination_file_path,
        kb_id=knowledgebase.id,
    )
    file_item.id = file_id
    file_entity : KbFileEntity = to_file_entity(file_item)
    session.add(file_entity)
    await session.commit()
    await kb_client.process_file_async(file_entity.id, True)
    await session.refresh(file_entity)
    if file_entity.status == FileStatus.succeeded:
        return success_response(data=file_entity, message="文件上传成功")
    else:
        return error_response(data=file_entity, message="文件上传失败")
