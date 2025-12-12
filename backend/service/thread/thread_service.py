"""Thread Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.thread import ThreadEntity, ThreadCreate


class ThreadService:
    """Service layer for Thread entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ThreadService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_thread(self, thread_id: str, tenant_id: str) -> Optional[ThreadEntity]:
        """
        Get a single Thread entity by ID.

        Args:
            thread_id: Thread entity ID

        Returns:
            ThreadEntity if found, None otherwise
        """
        result = await self.session.exec(select(ThreadEntity).where(ThreadEntity.id == thread_id, ThreadEntity.tenant_id == tenant_id))
        return result.first()

    async def list_threads(
        self,
        tenant_id: str,
        offset: int = 0,
        limit: int = 10,
    ) -> List[ThreadEntity]:
        """
        List Thread entities with pagination.

        Args:
            offset: Offset for pagination
            limit: Limit for pagination

        Returns:
            List of ThreadEntity
        """
        statement = (
            select(ThreadEntity).where(ThreadEntity.tenant_id == tenant_id)
            .order_by(ThreadEntity.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        results = await self.session.exec(statement)
        return list(results.all())

    async def create_thread(self, thread_data: ThreadCreate, tenant_id: str) -> ThreadEntity:
        """
        Create a new Thread entity.
        Note: Caller is responsible for committing the session.

        Args:
            thread_data: Thread creation data

        Returns:
            Created ThreadEntity (not yet committed)

        Raises:
            ValueError: If thread creation fails (IntegrityError converted)
        """
        thread = ThreadEntity.model_validate(thread_data, update={"tenant_id": tenant_id})
        self.session.add(thread)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(thread)

            logger.info(f"Created Thread entity: {thread.id}")
            return thread

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Thread: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Conversation {thread_data} already exists."
                ) from e
            else:
                raise ValueError(f"Failed to add conversation: {str(e)}") from e

    async def update_thread_title(
        self, thread_id: str, title: str, tenant_id: str
    ) -> ThreadEntity:
        """
        Update thread title.
        Note: Caller is responsible for committing the session.

        Args:
            thread_id: Thread entity ID
            title: New title

        Returns:
            Updated ThreadEntity (not yet committed)

        Raises:
            ValueError: If Thread entity not found
        """
        result = await self.session.exec(select(ThreadEntity).where(ThreadEntity.id == thread_id, ThreadEntity.tenant_id == tenant_id))
        thread = result.first()
        if not thread:
            raise ValueError(f"Conversation {thread_id} not found.")

        thread.title = title
        self.session.add(thread)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(thread)

        logger.info(f"Updated Thread title: {thread.id} -> {title}")
        return thread

    async def delete_thread(self, thread_id: str, tenant_id: str) -> None:
        """
        Delete a Thread entity.
        Note: Caller is responsible for committing the session.

        Args:
            thread_id: Thread entity ID

        Raises:
            ValueError: If Thread entity not found
        """
        result = await self.session.exec(select(ThreadEntity).where(ThreadEntity.id == thread_id, ThreadEntity.tenant_id == tenant_id))
        thread = result.first()
        if not thread:
            raise ValueError(f"Conversation {thread_id} not found.")

        # Delete from database (staged, not committed)
        await self.session.delete(thread)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted Thread entity: {thread_id}")
