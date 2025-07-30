# import traceback
# from typing import Dict, Type
# from sqlmodel import SQLModel
# from typing import Dict
# from sqlmodel import select
# from pydantic import Field
# from pairag.db.models.thread import ThreadEntity
# from pairag.db.models.message import MessageEntity, MessageRead
# from pairag.db.models.attachment.file import AttachmentFileEntity
# from pairag.mcp.providers.base_provider import BaseConfigProvider
# from pairag.db.db_context import with_async_db_session
# from sqlmodel.ext.asyncio.session import AsyncSession
# from loguru import logger
# @with_async_db_session
# async def delete_related_attachments_in_messages(session: AsyncSession, thread_id: str):
#     logger.info("[thread tool] Start deleting related attachments in messages.")
#     sql_results = await session.exec(
#         select(MessageEntity)
#         .where(MessageEntity.thread_id == thread_id)
#     )
#     message_entities = sql_results.all()
#     message_models = [
#         MessageRead.model_validate(
#             message,
#         )
#         for message in message_entities
#     ]
#     message_ids = [message.id for message in message_models]
#     file_sql_results = await session.exec(
#         select(AttachmentFileEntity)
#         .where(AttachmentFileEntity.message_id.in_(message_ids))
#     )
#     attachment_file_entities = file_sql_results.all()
#     for attachment_file_entity in attachment_file_entities:
#         await session.delete(attachment_file_entity)
#         await session.commit()
#     logger.info("[thread tool] Deleted related attachments in messages successfully.")
