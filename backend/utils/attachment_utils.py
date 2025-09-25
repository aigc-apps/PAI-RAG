
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import with_async_db_session
from db.models.knowledgebase.file import KbFileEntity

@with_async_db_session
async def acheck_content_length_from_db(session: AsyncSession, file_id: str):
    processed_file_entity = await session.get(KbFileEntity, file_id)
    if processed_file_entity.file_content_length > 1000:
        return True
    else:
        return False

async def is_attachment_truncated(file_id: str):
    """Get read file tool"""
    return await acheck_content_length_from_db(file_id=file_id)
