### Embedding configuration API ###
import os
import asyncio
from fastapi import APIRouter, File, UploadFile, Form, Depends
from pairag.mcp.online_file_readers.pai_online_data_reader import PaiOnlineDataReader
from pairag.db.models.knowledgebase.knowledgebase import (
    ChunkConfig,
    KbEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.db_context import get_session
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider
from pairag.mcp.rag.file.store.file_store_helper import file_store
from pairag.mcp.rag.file.models.file_item import FileItem
from pairag.db.models.attachment.file import AttachmentFileEntity
from pairag.api.response_model import success_response, error_response
from pairag.common.knowledgebase.types import FileStatus
from pairag.mcp.tools.knowledgebase.knowledgebase_tool import kb_client



attachments_router = APIRouter()
ATTACHMENTS_DIR = "localdata/attachments"
ATTACHMENTS_TMP_DIR = "localdata/attachments/tmp"
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
os.makedirs(ATTACHMENTS_TMP_DIR, exist_ok=True)

data_reader = PaiOnlineDataReader()


@attachments_router.post("/upload")
async def upload_attachment_file(
    file_id: str = Form(...), file: UploadFile = File(...), session: AsyncSession = Depends(get_session)
):
    knowledgebase = knowledgebase_provider.get_knowledgebase_by_name("default_attachments")
    if not knowledgebase:
        kb = KnowledgebaseCreate(
            name="default_attachments",
            description="附件知识库",
            embedding_model="text-embedding-v4"
        )
        kb.chunk_config = (ChunkConfig()).model_dump()
        kb.retrieval_config = (RetrievalConfig()).model_dump()
        knowledgebase = KbEntity.model_validate(kb)
        session.add(knowledgebase)
        await session.commit()
        await session.refresh(knowledgebase)
        asyncio.create_task(knowledgebase_provider.refresh())

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
    file_entity : AttachmentFileEntity = file_item.to_attachment_file_entity(file_id)
    session.add(file_entity)
    await session.commit()
    await kb_client.process_file_async(file_entity.id, True)
    await session.refresh(file_entity)
    if file_entity.status == FileStatus.succeeded:
        return success_response(data=file_entity, message="文件上传成功")
    else:
        return error_response(data=file_entity, message="文件上传失败")
