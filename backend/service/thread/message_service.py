"""Message Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

from db.models.message import MessageEntity, MessageCreate
from service.file.file_resource_service import FileResourceService


class MessageService:
    """Service layer for Message entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize MessageService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_message(self, message_id: str, tenant_id: str) -> Optional[MessageEntity]:
        """
        Get a single Message entity by ID.

        Args:
            message_id: Message entity ID

        Returns:
            MessageEntity if found, None otherwise
        """
        result = await self.session.exec(select(MessageEntity).where(MessageEntity.id == message_id, MessageEntity.tenant_id == tenant_id))
        return result.first()

    async def get_message_by_local_id(
        self, thread_id: str, local_id: str, tenant_id: str
    ) -> Optional[MessageEntity]:
        """
        Get a Message entity by thread_id and local_id.

        Args:
            thread_id: Thread ID
            local_id: Local ID

        Returns:
            MessageEntity if found, None otherwise
        """
        statement = select(MessageEntity).where(
            MessageEntity.thread_id == thread_id,
            MessageEntity.tenant_id == tenant_id,
            MessageEntity.local_id == local_id,
        )
        result = await self.session.exec(statement)
        return result.first()

    async def list_messages(
        self,
        thread_id: str,
        tenant_id: str,
        offset: int = 0,
        limit: int = 30,
    ) -> List[MessageEntity]:
        """
        List Message entities for a thread.

        Args:
            thread_id: Thread ID
            limit: Maximum number of messages to return

        Returns:
            List of MessageEntity
        """
        statement = (
            select(MessageEntity)
            .where(MessageEntity.thread_id == thread_id, MessageEntity.tenant_id == tenant_id)
            .order_by(MessageEntity.created_at)
            .offset(offset)
            .limit(limit)
        )
        results = await self.session.exec(statement)
        return list(results.all())

    async def get_message_ids_by_thread(
        self, thread_id: str, tenant_id: str
    ) -> List[str]:
        """
        Get all message IDs for a thread.

        Args:
            thread_id: Thread ID

        Returns:
            List of message IDs
        """
        statement = select(MessageEntity.id).where(
            MessageEntity.thread_id == thread_id, MessageEntity.tenant_id == tenant_id
        )
        results = await self.session.exec(statement)
        return list(results.all())

    async def create_message(
        self, message_data: MessageCreate, tenant_id: str
    ) -> MessageEntity:
        """
        Create a new Message entity.
        If message.local_id is provided and a message with that local_id exists,
        update the existing message instead of creating a new one.
        Note: Caller is responsible for committing the session.

        Args:
            message_data: Message creation data

        Returns:
            Created or updated MessageEntity (not yet committed)
        """
        thread_id = message_data.thread_id

        # Check if message with local_id already exists
        if message_data.local_id:
            existing_message = await self.get_message_by_local_id(
                thread_id=thread_id, local_id=message_data.local_id, tenant_id=tenant_id
            )
            if existing_message:
                # Update existing message
                existing_message.attachments = message_data.attachments
                existing_message.content = message_data.content
                existing_message.role = message_data.role
                # Update token_usage if provided
                if message_data.token_usage:
                    existing_message.token_usage = message_data.token_usage
                message_entity = existing_message
            else:
                # Create new message
                message_entity = MessageEntity.model_validate(message_data, update={"tenant_id": tenant_id})
        else:
            # Create new message
            message_entity = MessageEntity.model_validate(message_data, update={"tenant_id": tenant_id})

        self.session.add(message_entity)

        # Increment ref_count on attached files so they aren't GC'd while this
        # message references them. Actual cleanup happens on thread delete
        # (release_attachment_refs) or via the TTL/GC sweep (Phase 2).
        attachment_ids = [
            a.get("id") for a in (message_data.attachments or []) if a.get("id")
        ]
        if attachment_ids:
            file_service = FileResourceService(self.session)
            await file_service.increment_refs(file_ids=attachment_ids, tenant_id=tenant_id)

        # Flush to get the ID, but don't commit
        await self.session.flush()
        await self.session.refresh(message_entity)

        logger.info(f"Created/Updated Message entity: {message_entity.id}")
        return message_entity

    async def release_attachment_refs(
        self, thread_id: str, tenant_id: str
    ) -> None:
        """Decrement ref_count on all files attached to messages in a thread.

        Files with ref_count=0 and expires_at in the past are later swept by
        the GC worker (Phase 2). We do not hard-delete here to keep retries
        idempotent and to avoid losing files still referenced by other threads.
        """
        logger.info(f"[MessageService] Releasing attachment refs for thread {thread_id}.")

        stmt = select(MessageEntity.attachments).where(
            MessageEntity.thread_id == thread_id,
            MessageEntity.tenant_id == tenant_id,
        )
        rows = list((await self.session.exec(stmt)).all())
        file_ids: List[str] = []
        for att_list in rows:
            if not att_list:
                continue
            for att in att_list:
                fid = att.get("id") if isinstance(att, dict) else None
                if fid:
                    file_ids.append(fid)

        if not file_ids:
            logger.info(f"[MessageService] No attachments to release for thread {thread_id}.")
            return

        file_service = FileResourceService(self.session)
        await file_service.decrement_refs(file_ids=file_ids, tenant_id=tenant_id)
        await self.session.flush()
        logger.info(
            f"[MessageService] Released {len(file_ids)} attachment refs for thread {thread_id}."
        )
