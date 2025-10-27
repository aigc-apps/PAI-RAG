from typing import Any, Dict, BinaryIO
from pairag.file.store.file_store_helper import file_store
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import with_async_db_session
from db.models.knowledgebase.file import KbFileEntity
from config.providers.knowledgebase_provider import knowledgebase_provider
from pairag.file.models.file_item import FileItem
from rag.file_item_utils import to_file_entity
from common.knowledgebase.types import FileStatus
from rag.kb_file_client import kb_file_client
import time
import uuid
from db.models.knowledgebase.file_task import KbFileTaskEntity
import requests
def to_openai_tool(tool_meatadata) -> Dict[str, Any]:
        """To OpenAI tool."""
        return {
            "type": "function",
            "function": {
                "name": tool_meatadata.name,
                "description": tool_meatadata.description,
                "parameters": tool_meatadata.get_parameters_dict(),
            },
        }

@with_async_db_session
async def aget_file_url_from_db(session: AsyncSession, file: BinaryIO, file_name: str):
    knowledgebase = knowledgebase_provider.get_knowledgebase_by_name("default_chat_docs")
    destination_file_path = f"{knowledgebase.name}/docs/{file_name}"
    file_store.save(
        file=file,
        file_path=destination_file_path,
    )
    file_item = FileItem.from_file(
        file=file,
        file_path=destination_file_path,
        kb_id=knowledgebase.id,
    )
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
    await kb_file_client.process_file_async(file_task_entity.id, False)
    await session.refresh(file_entity)
    if file_entity.status == FileStatus.succeeded:
        return file_store.get_url(destination_file_path)
    else:
        return None



def get_binary_content_from_oss_url(oss_url):
    response = requests.get(oss_url)

    if response.status_code != 200:
        raise IOError(f"Failed to download file from {oss_url}. Status code: {response.status_code}")


    return response.content
