"""FileTask Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy import delete
from loguru import logger

from db.models.knowledgebase.file_task import KbFileTaskEntity
from common.chat.response_model import PagedResult


class FileTaskService:
    """Service layer for FileTask entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize FileTaskService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_file_task(self, task_id: str, tenant_id: str) -> Optional[KbFileTaskEntity]:
        """
        Get a single FileTask entity by ID.

        Args:
            task_id: FileTask entity ID

        Returns:
            KbFileTaskEntity if found, None otherwise
        """
        result = await self.session.exec(select(KbFileTaskEntity).where(KbFileTaskEntity.id == task_id, KbFileTaskEntity.tenant_id == tenant_id))
        return result.first()

    async def get_file_task_by_file_and_part(
        self, kb_id: str, file_id: str, file_part: int, tenant_id: str
    ) -> Optional[KbFileTaskEntity]:
        """
        Get a single FileTask entity by file_id, kb_id, and file_part.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            file_part: File part index

        Returns:
            KbFileTaskEntity if found, None otherwise
        """
        statement = (
            select(KbFileTaskEntity)
            .where(KbFileTaskEntity.kb_id == kb_id)
            .where(KbFileTaskEntity.file_id == file_id)
            .where(KbFileTaskEntity.file_part == file_part)
            .where(KbFileTaskEntity.tenant_id == tenant_id)
        )
        result = await self.session.exec(statement)
        return result.first()

    async def list_file_tasks(
        self,
        kb_id: str,
        tenant_id: str,
        file_id: Optional[str] = None,
        page: int = 1,
        size: int = 10,
        status: Optional[str] = None,
    ) -> PagedResult[List[KbFileTaskEntity]]:
        """
        List FileTask entities with pagination and optional filtering.

        Args:
            kb_id: Knowledgebase ID
            file_id: Optional file ID to filter by
            page: Page number (1-indexed)
            size: Page size
            status: Optional filter for task status

        Returns:
            PagedResult containing list of KbFileTaskEntity and pagination metadata
        """
        # Build base query
        base_query = select(KbFileTaskEntity).where(
            KbFileTaskEntity.kb_id == kb_id,
            KbFileTaskEntity.tenant_id == tenant_id
        )

        # Add file_id filter if provided
        if file_id:
            base_query = base_query.where(KbFileTaskEntity.file_id == file_id)

        # Add status filter if provided
        if status:
            base_query = base_query.where(KbFileTaskEntity.status == status)

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = (
            base_query.order_by(KbFileTaskEntity.file_part)
            .offset(offset)
            .limit(size)
        )
        results = await self.session.exec(paginated_query)
        tasks = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=tasks,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_file_task(
        self, task_data: KbFileTaskEntity, tenant_id: str
    ) -> KbFileTaskEntity:
        """
        Create a new FileTask entity.
        Note: Caller is responsible for committing the session.

        Args:
            task_data: FileTask entity data

        Returns:
            Created KbFileTaskEntity (not yet committed)

        Raises:
            ValueError: If task already exists (IntegrityError converted)
        """
        task_data.tenant_id = tenant_id
        self.session.add(task_data)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(task_data)

            logger.info(
                f"Created FileTask entity: {task_data.id} (file_id: {task_data.file_id}, file_part: {task_data.file_part})"
            )
            return task_data

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating FileTask: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"File task (file_id: {task_data.file_id}, file_part: {task_data.file_part}) already exists."
                ) from e
            else:
                raise ValueError(f"File task creation failed: {e}") from e

    async def update_file_task(
        self,
        task_id: str,
        kb_id: str,
        file_id: str,
        tenant_id: str,
        **update_fields,
    ) -> KbFileTaskEntity:
        """
        Update an existing FileTask entity.
        Note: Caller is responsible for committing the session.

        Args:
            task_id: FileTask entity ID
            kb_id: Knowledgebase ID (for validation)
            file_id: File ID (for validation)
            **update_fields: Fields to update

        Returns:
            Updated KbFileTaskEntity (not yet committed)

        Raises:
            ValueError: If FileTask entity not found or doesn't belong to kb_id/file_id
        """
        result = await self.session.exec(select(KbFileTaskEntity).where(KbFileTaskEntity.id == task_id, KbFileTaskEntity.tenant_id == tenant_id))
        task_entity = result.first()
        if not task_entity:
            raise ValueError(f"File task '{task_id}' does not exist.")

        if task_entity.kb_id != kb_id:
            raise ValueError(f"File task '{task_id}' does not belong to knowledge base '{kb_id}'.")

        if task_entity.file_id != file_id:
            raise ValueError(f"File task '{task_id}' does not belong to file '{file_id}'.")

        logger.info(f"Updating FileTask {task_id} with fields: {update_fields}")

        # Update fields
        for key, value in update_fields.items():
            if hasattr(task_entity, key) and value is not None:
                setattr(task_entity, key, value)

        task_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(task_entity)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(task_entity)

        logger.info(
            f"Updated FileTask entity: {task_entity.id} (file_id: {task_entity.file_id}, file_part: {task_entity.file_part})"
        )
        return task_entity

    async def delete_file_task(
        self, task_id: str, kb_id: str, file_id: str, tenant_id: str
    ) -> None:
        """
        Delete a FileTask entity.
        Note: Caller is responsible for committing the session.

        Args:
            task_id: FileTask entity ID
            kb_id: Knowledgebase ID (for validation)
            file_id: File ID (for validation)

        Raises:
            ValueError: If FileTask entity not found or doesn't belong to kb_id/file_id
        """
        result = await self.session.exec(select(KbFileTaskEntity).where(KbFileTaskEntity.id == task_id, KbFileTaskEntity.tenant_id == tenant_id))
        task_entity = result.first()
        if not task_entity:
            raise ValueError(f"File task '{task_id}' does not exist.")

        if task_entity.kb_id != kb_id:
            raise ValueError(f"File task '{task_id}' does not belong to knowledge base '{kb_id}'.")

        if task_entity.file_id != file_id:
            raise ValueError(f"File task '{task_id}' does not belong to file '{file_id}'.")

        # Delete from database (staged, not committed)
        await self.session.delete(task_entity)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(
            f"Deleted FileTask entity: {task_id} (file_id: {file_id}, file_part: {task_entity.file_part})"
        )

    async def get_file_tasks_by_file(
        self, kb_id: str, file_id: str, tenant_id: str
    ) -> List[KbFileTaskEntity]:
        """
        Get all FileTask entities for a file without pagination.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID

        Returns:
            List of all KbFileTaskEntity for the file (ordered by file_part)
        """
        statement = (
            select(KbFileTaskEntity)
            .where(KbFileTaskEntity.kb_id == kb_id)
            .where(KbFileTaskEntity.file_id == file_id)
            .where(KbFileTaskEntity.tenant_id == tenant_id)
            .order_by(KbFileTaskEntity.file_part)
        )
        results = await self.session.exec(statement)
        return list(results.all())

    async def get_file_tasks_by_kb(self, kb_id: str, tenant_id: str) -> List[KbFileTaskEntity]:
        """
        Get all FileTask entities for a knowledgebase without pagination.

        Args:
            kb_id: Knowledgebase ID

        Returns:
            List of all KbFileTaskEntity for the knowledgebase
        """
        statement = select(KbFileTaskEntity).where(
            KbFileTaskEntity.kb_id == kb_id,
            KbFileTaskEntity.tenant_id == tenant_id
        )
        results = await self.session.exec(statement)
        return list(results.all())

    async def delete_file_tasks_from_file(
        self, kb_id: str, file_id: str, tenant_id: str
    ) -> None:
        """
        Delete all FileTask entities for a file.
        Note: This directly deletes all tasks without querying first.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
        """
        # Directly delete all tasks for this file
        stmt = (
            delete(KbFileTaskEntity)
            .where(KbFileTaskEntity.kb_id == kb_id)
            .where(KbFileTaskEntity.file_id == file_id)
            .where(KbFileTaskEntity.tenant_id == tenant_id)
        )
        result = await self.session.execute(stmt)
        deleted_count = result.rowcount

        # Flush to ensure deletions are staged
        await self.session.flush()

        logger.info(
            f"Deleted {deleted_count} FileTask entities for file {file_id} in knowledgebase {kb_id}"
        )

    async def delete_file_tasks_from_kb(self, kb_id: str, tenant_id: str) -> None:
        """
        Delete all FileTask entities for a knowledgebase.
        Note: This directly deletes all tasks without querying first.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
        """
        # Directly delete all tasks for this knowledgebase
        stmt = delete(KbFileTaskEntity).where(KbFileTaskEntity.kb_id == kb_id, KbFileTaskEntity.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        deleted_count = result.rowcount

        # Flush to ensure deletions are staged
        await self.session.flush()

        logger.info(
            f"Deleted {deleted_count} FileTask entities from knowledgebase {kb_id}"
        )
