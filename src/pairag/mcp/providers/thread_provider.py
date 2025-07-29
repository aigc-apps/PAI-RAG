from typing import Dict
from sqlmodel import select
from pairag.db.models.thread import ThreadEntity
from pairag.db.models.message import MessageEntity, MessageRead
from pairag.db.models.attachment.file import AttachmentFileEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger


@with_async_db_session
async def fetch_threads(session: AsyncSession):
    logger.info("[ThreadProvider] Start fetching threads.")
    sql_results = await session.exec(select(ThreadEntity))
    thread_results = sql_results.all()

    logger.info(f"[ThreadProvider] fetched {len(thread_results)} threads.")
    return {thread.id: thread for thread in thread_results}


@with_async_db_session
async def fetch_thread_messages(session: AsyncSession):
    logger.info("[ThreadProvider] Start fetching messages.")
    sql_results = await session.exec(select(MessageEntity))
    message_results = sql_results.all()

    logger.info(f"[ThreadProvider] fetched {len(message_results)} messages.")
    return {message.id: message for message in message_results}

@with_async_db_session
async def delete_related_attachments_in_messages(session: AsyncSession, thread_id: str):
    logger.info("[ThreadProvider] Start deleting related attachments in messages.")
    sql_results = await session.exec(
        select(MessageEntity)
        .where(MessageEntity.thread_id == thread_id)
    )
    message_entities = sql_results.all()
    message_models = [
        MessageRead.model_validate(
            message,
        )
        for message in message_entities
    ]
    message_ids = [message.id for message in message_models]
    file_sql_results = await session.exec(
        select(AttachmentFileEntity)
        .where(AttachmentFileEntity.message_id.in_(message_ids))
    )
    attachment_file_entities = file_sql_results.all()
    for attachment_file_entity in attachment_file_entities:
        await session.delete(attachment_file_entity)
        await session.commit()
    logger.info("[ThreadProvider] Deleted related attachments in messages successfully.")

class ThreadProvider:
    def __init__(self):
        self.thread_map: Dict[str, ThreadEntity] = {}

    async def refresh(self):
        logger.info("[ThreadProvider] Start refreshing threads.")
        self.thread_map = await fetch_threads()
        self.message_map = await fetch_thread_messages()
        logger.info(
            f"[ThreadProvider] refreshed {len(self.thread_map)} threads and {len(self.message_map)} messages."
        )

    def get_thread(self, thread_id: str) -> ThreadEntity:
        assert thread_id in self.thread_map, f"Thread '{thread_id}' not found."
        return self.thread_map[thread_id]

    async def delete_related_attachments(self, thread_id: str):
        await delete_related_attachments_in_messages(thread_id)


thread_provider = ThreadProvider()
