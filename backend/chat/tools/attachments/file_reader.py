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
    return content, processed_file_entity.file_name

async def aget_file_content(file_id: str):
    """Get read file tool"""
    content, file_name = await aget_file_content_from_db(file_id=file_id)
    result = f"📄 文件“{file_name}” (ID:{file_id}) 的内容如下：\n\n {content}"
    return json.dumps({"data": result}, ensure_ascii=False)


async def aget_file_reader():
    read_file_tool = FunctionTool.from_defaults(
        async_fn=aget_file_content,
        name="read-file",
        description="根据提供的附件ID读取文件的内容。",
    )
    return read_file_tool
