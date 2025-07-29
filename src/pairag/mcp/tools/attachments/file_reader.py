import json
from llama_index.core.tools import FunctionTool
from sqlmodel import select

from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.db_context import with_async_db_session
from pairag.db.models.attachment.file import AttachmentFileEntity

@with_async_db_session
async def aget_file_content_from_db(session: AsyncSession, file_id: str):
    file_res = await session.exec(
        select(AttachmentFileEntity).where(
            AttachmentFileEntity.frontend_file_id == file_id
        )
    )
    processed_file_entity = file_res.first()
    content =  processed_file_entity.file_content.decode('utf-8', errors='ignore')
    if processed_file_entity.file_content_length > 200:
        content =  content[0:1000] + " \n\n 文件内容太长，已经被截断。如果需要更多信息，请使用【文件检索】工具。"
    return content

async def aget_file_content(file_id: str, file_name: str = None):
    """Get read file tool"""
    content = await aget_file_content_from_db(file_id)
    return json.dumps({"data": content}, ensure_ascii=False)


async def aget_file_reader():
    read_file_tool = FunctionTool.from_defaults(
        async_fn=aget_file_content,
        name="read-file",
        description="根据提供的附件ID读取文件的内容。",
    )
    return read_file_tool
