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
#     logger.info("[ThreadProvider] Start deleting related attachments in messages.")
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
#     logger.info("[ThreadProvider] Deleted related attachments in messages successfully.")

# class ThreadProvider(BaseConfigProvider):
#     name_to_entry_id: Dict[str, str] = Field(default={})
#     entity_class: Type[SQLModel] = ThreadEntity
#     message_entity_class: Type[SQLModel] = MessageEntity
#     def add(self, entry: ThreadEntity):
#         super().add(entry)
#         self.name_to_entry_id[entry.name] = entry.id

#     def update(self, entry: ThreadEntity):
#         super().update(entry)
#         self.name_to_entry_id[entry.name] = entry.id

#     def delete(self, entry_id: str):
#         super().delete(entry_id)
#         try:
#             for k, v in self.name_to_entry_id.items():
#                 if v == entry_id:
#                     del self.name_to_entry_id[k]
#                     break
#         except Exception:
#             logger.warning(f"Failed to delete entry with entry_id {entry_id}. error: {traceback.format_exc()}.")

#     def _load_entries(self, entries):
#         super()._load_entries(entries)
#         for entry_id, entry in self.config_map.items():
#             self.name_to_entry_id[entry.name] = entry_id

#     @with_async_db_session
#     async def full_load_from_db_async(self, session: AsyncSession):
#         entries = (await session.exec(select(self.entity_class))).all()
#         self._load_entries(entries)
#         message_entries = (await session.exec(select(self.message_entity_class))).all()
#         self._load_entries(message_entries)

#     async def delete_related_attachments(self, thread_id: str):
#         await delete_related_attachments_in_messages(thread_id)


# thread_provider = ThreadProvider()
