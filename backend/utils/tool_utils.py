from typing import Any, Dict, BinaryIO
from pairag.file.store.file_store_helper import file_store
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import with_async_db_session
from db.models.knowledgebase.file import KbFileEntity
from config.providers.knowledgebase_provider import knowledgebase_provider
from pairag.file.models.file_item import FileItem
from rag.file_item_utils import to_file_entity
from common.knowledgebase.types import FileStatus
from rag.knowledgebase_tool import kb_client
import requests
import io
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
    session.add(file_entity)
    await session.commit()
    await kb_client.process_file_async(file_entity.id, True)
    await session.refresh(file_entity)
    if file_entity.status == FileStatus.succeeded:
        return file_store.get_url(destination_file_path)
    else:
        return None



def get_binary_io_from_oss_url(oss_url):
    response = requests.get(oss_url)

    if response.status_code != 200:
        raise IOError(f"Failed to download file from {oss_url}. Status code: {response.status_code}")

    file_bytes = response.content

    binary_io = io.BytesIO(file_bytes)

    return binary_io
