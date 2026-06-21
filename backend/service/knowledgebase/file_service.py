"""File Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import select, func, delete
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.knowledgebase.file import KbFileEntity
from common.chat.response_model import PagedResult



class FileService:
    """Service layer for File entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize FileService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_file_by_id(self, file_id: str, tenant_id: str) -> Optional[KbFileEntity]:
        """
        Get a single File entity by ID.
        """
        result = await self.session.exec(select(KbFileEntity).where(KbFileEntity.id == file_id, KbFileEntity.tenant_id == tenant_id))
        return result.first()

    async def get_files_by_ids(self, file_ids: List[str], tenant_id: str) -> List[KbFileEntity]:
        """
        Get multiple File entities by IDs.
        """
        if not file_ids:
            return []

        statement = select(KbFileEntity).where(KbFileEntity.id.in_(file_ids), KbFileEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        return list(result.all())

    async def get_file(self, kb_id: str, file_id: str, tenant_id: str) -> Optional[KbFileEntity]:
        """
        Get a single File entity by ID.

        Args:
            kb_id: Knowledgebase ID
            file_id: File entity ID

        Returns:
            KbFileEntity if found, None otherwise
        """
        logger.info(f"kb_id is {kb_id}, file_id is {file_id}, tenant_id is {tenant_id}")
        statement = select(KbFileEntity).where(
            KbFileEntity.kb_id == kb_id, KbFileEntity.id == file_id, KbFileEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        return result.first()

    async def get_files_by_names(
        self, kb_id: str, file_names: List[str], tenant_id: str
    ) -> List[KbFileEntity]:
        """
        Get multiple File entities by path within a knowledgebase.

        Args:
            kb_id: Knowledgebase ID
            file_names: List of file names

        Returns:
            List of KbFileEntity if found, None otherwise
        """
        if not file_names:
            return []

        statement = select(KbFileEntity).where(
            KbFileEntity.kb_id == kb_id, KbFileEntity.file_name.in_(file_names), KbFileEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        return list(result.all())

    async def get_file_by_name(
        self, kb_id: str, file_name: str, tenant_id: str
    ) -> Optional[KbFileEntity]:
        """
        Get a single File entity by name within a knowledgebase.

        Args:
            kb_id: Knowledgebase ID
            file_name: File name

        Returns:
            KbFileEntity if found, None otherwise
        """
        statement = select(KbFileEntity).where(
            KbFileEntity.kb_id == kb_id, KbFileEntity.file_name == file_name, KbFileEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        return result.first()

    async def list_files(
        self,
        kb_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        query: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
    ) -> PagedResult[List[KbFileEntity]]:
        """
        List File entities with pagination and optional filtering.

        Args:
            kb_id: Knowledgebase ID
            page: Page number (1-indexed)
            size: Page size
            query: Optional search query (searches in file_name)
            status: Optional filter for file status

        Returns:
            PagedResult containing list of KbFileEntity and pagination metadata
        """
        # Build base query
        base_query = select(KbFileEntity).where(KbFileEntity.kb_id == kb_id, KbFileEntity.tenant_id == tenant_id)

        # Add query filter if provided — match the storage name OR the human
        # title stored in file_metadata (data-source docs carry a real title;
        # manually-uploaded files match via file_name).
        if query:
            like = f"%{query.lower()}%"
            title_text = func.lower(
                func.coalesce(KbFileEntity.file_metadata["title"].as_string(), "")
            )
            base_query = base_query.where(
                or_(
                    func.lower(KbFileEntity.file_name).like(like),
                    title_text.like(like),
                )
            )

        # Add status filter if provided
        if status:
            base_query = base_query.where(KbFileEntity.status == status)

        # Add source filter if provided: "manual" = no datasource_key, otherwise
        # match a specific data source key in file_metadata.
        if source:
            ds_key = func.coalesce(
                KbFileEntity.file_metadata["datasource_key"].as_string(), ""
            )
            if source == "manual":
                base_query = base_query.where(ds_key == "")
            else:
                base_query = base_query.where(ds_key == source)

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = (
            base_query.order_by(KbFileEntity.updated_at.desc(), KbFileEntity.id.asc())
            .offset(offset)
            .limit(size)
        )
        results = await self.session.exec(paginated_query)
        files = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=files,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_file(self, file_data: KbFileEntity, tenant_id: str) -> KbFileEntity:
        """
        Create a new File entity.
        Note: Caller is responsible for committing the session.

        Args:
            file_data: File entity data

        Returns:
            Created KbFileEntity (not yet committed)

        Raises:
            ValueError: If file already exists (IntegrityError converted)
        """
        file_data.tenant_id = tenant_id
        self.session.add(file_data)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(file_data)

            logger.info(
                f"Created File entity: {file_data.id} (file_name: {file_data.file_name})"
            )
            return file_data

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating File: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"File '{file_data.file_name}' already exists in knowledge base."
                ) from e
            else:
                raise ValueError(f"File creation failed: {e}") from e

    async def update_file(
        self, file_id: str, kb_id: str, new_entity: KbFileEntity, tenant_id: str
    ) -> KbFileEntity:
        """
        Update an existing File entity.
        Note: Caller is responsible for committing the session.

        Args:
            file_id: File entity ID
            kb_id: Knowledgebase ID (for validation)
            new_entity: New File entity data

        Returns:
            Updated KbFileEntity (not yet committed)

        Raises:
            ValueError: If File entity not found or doesn't belong to kb_id
        """
        result = await self.session.exec(select(KbFileEntity).where(
            KbFileEntity.id == file_id,
            KbFileEntity.kb_id == kb_id,
            KbFileEntity.tenant_id == tenant_id
        ))
        file_entity = result.first()
        if not file_entity:
            raise ValueError(f"File '{file_id}' does not exist.")

        file_entity.file_metadata = new_entity.file_metadata
        file_entity.file_source = new_entity.file_source
        file_entity.file_name = new_entity.file_name
        file_entity.file_path = new_entity.file_path
        file_entity.file_extension = new_entity.file_extension
        file_entity.file_size = new_entity.file_size
        file_entity.file_md5 = new_entity.file_md5
        file_entity.file_version = new_entity.file_version
        file_entity.status = new_entity.status
        file_entity.failed_reason = new_entity.failed_reason
        file_entity.active = new_entity.active
        file_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(file_entity)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(file_entity)

        logger.info(
            f"Updated File entity: {file_entity.id} (file_name: {file_entity.file_name})"
        )
        return file_entity


    async def delete_file(self, file_id: str, kb_id: str, tenant_id: str) -> None:
        """
        Delete a File entity.
        Note: This will cascade delete related chunks.
        Note: Caller is responsible for committing the session.

        Args:
            file_id: File entity ID
            kb_id: Knowledgebase ID (for validation)

        Raises:
            ValueError: If File entity not found or doesn't belong to kb_id
        """
        result = await self.session.exec(select(KbFileEntity).where(
            KbFileEntity.id == file_id,
            KbFileEntity.kb_id == kb_id,
            KbFileEntity.tenant_id == tenant_id
        ))
        file_entity = result.first()
        if not file_entity:
            raise ValueError(f"File '{file_id}' does not exist.")

        if file_entity.kb_id != kb_id:
            raise ValueError(f"File '{file_id}' does not belong to knowledge base '{kb_id}'.")

        # Delete from database (staged, not committed)
        # CASCADE will handle related chunks
        await self.session.delete(file_entity)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(
            f"Deleted File entity: {file_id} (file_name: {file_entity.file_name})"
        )

    async def get_files_by_kb(self, kb_id: str, tenant_id: str) -> List[KbFileEntity]:
        """
        Get all File entities for a knowledgebase without pagination.

        Args:
            kb_id: Knowledgebase ID

        Returns:
            List of all KbFileEntity for the knowledgebase
        """
        statement = select(KbFileEntity).where(KbFileEntity.kb_id == kb_id, KbFileEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        return list(results.all())

    ## Batch
    async def delete_files_from_kb(self, kb_id: str, tenant_id: str) -> None:
        """
        Delete all File entities for a knowledgebase.
        Note: This directly deletes all files without querying first.
        Note: This will cascade delete related chunks.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
        """
        # Directly delete all files for this knowledgebase
        stmt = delete(KbFileEntity).where(KbFileEntity.kb_id == kb_id, KbFileEntity.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        deleted_count = result.rowcount

        logger.info(f"Deleted {deleted_count} File entities from knowledgebase {kb_id}")
