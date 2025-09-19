import json
from llama_index.core.tools import FunctionTool

from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import with_async_db_session
from db.models.knowledgebase.file import KbFileEntity
from pairag.file.store.file_store_helper import file_store

@with_async_db_session
async def aget_file_content_from_db(session: AsyncSession, file_id: str):
    processed_file_entity = await session.get(KbFileEntity, file_id)
    if processed_file_entity.file_extension in [".jpeg", ".png", ".jpg"]:
        return file_store.get_url(processed_file_entity.file_path)
    content =  processed_file_entity.file_content
    if processed_file_entity.file_content_length > 200:
        content =  content[0:1000] + " \n\n [truncated] The content is too long, has been truncated."
    return content

async def aget_file_content(file_id: str, file_name: str = None):
    """Get read file tool"""
    content = await aget_file_content_from_db(file_id=file_id)
    return json.dumps({"data": content}, ensure_ascii=False)


async def aget_file_reader():
    read_file_tool = FunctionTool.from_defaults(
        async_fn=aget_file_content,
        name="read-file",
        description="根据提供的附件ID读取文件的内容。",
    )
    return read_file_tool
