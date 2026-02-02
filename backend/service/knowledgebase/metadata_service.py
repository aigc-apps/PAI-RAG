"""Metadata Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List, Dict
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy import delete
from loguru import logger

from db.models.knowledgebase.metadata import (
    KbMetadataEntity,
    KbMetadataEntityCreate,
    FileMetadataEntity,
)
from common.chat.response_model import PagedResult
from common.knowledgebase.constants import DEFAULT_METADATA_KEYS

class MetadataService:
    """Service layer for Metadata entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize MetadataService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_metadata(
        self, kb_id: str, metadata_id: str, tenant_id: str
    ) -> Optional[KbMetadataEntity]:
        """
        Get a single Metadata entity by ID.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata entity ID

        Returns:
            KbMetadataEntity if found, None otherwise
        """
        statement = (
            select(KbMetadataEntity)
            .where(KbMetadataEntity.id == metadata_id)
            .where(KbMetadataEntity.kb_id == kb_id)
            .where(KbMetadataEntity.tenant_id == tenant_id)
        )
        result = await self.session.exec(statement)
        return result.first()

    async def get_metadata_by_name(
        self, kb_id: str, name: str, tenant_id: str
    ) -> Optional[KbMetadataEntity]:
        """
        Get a single Metadata entity by name.

        Args:
            kb_id: Knowledgebase ID
            name: Metadata name

        Returns:
            KbMetadataEntity if found, None otherwise
        """
        statement = (
            select(KbMetadataEntity)
            .where(KbMetadataEntity.kb_id == kb_id)
            .where(KbMetadataEntity.name == name)
            .where(KbMetadataEntity.tenant_id == tenant_id)
        )
        result = await self.session.exec(statement)
        return result.first()

    async def list_metadata(
        self,
        kb_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 20,
    ) -> PagedResult[List[Dict]]:
        """
        List Metadata entities with file count and pagination.

        Args:
            kb_id: Knowledgebase ID
            page: Page number (1-indexed)
            size: Page size

        Returns:
            PagedResult containing list of metadata dicts with count field and pagination metadata
        """
        # Get total count
        count_query = select(func.count(KbMetadataEntity.id)).where(
            KbMetadataEntity.kb_id == kb_id,
            KbMetadataEntity.tenant_id == tenant_id,
        )
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Subquery: count files per metadata_id
        file_count_subquery = (
            select(
                FileMetadataEntity.metadata_id,
                func.count(FileMetadataEntity.id).label("count"),
            )
            .where(FileMetadataEntity.kb_id == kb_id)
            .where(FileMetadataEntity.tenant_id == tenant_id)
            .group_by(FileMetadataEntity.metadata_id)
            .subquery()
        )

        # Main query: LEFT JOIN to get file count
        offset = (page - 1) * size
        query = (
            select(
                KbMetadataEntity,
                func.coalesce(file_count_subquery.c.count, 0).label("count"),
            )
            .outerjoin(
                file_count_subquery,
                KbMetadataEntity.id == file_count_subquery.c.metadata_id,
            )
            .where(KbMetadataEntity.kb_id == kb_id)
            .where(KbMetadataEntity.tenant_id == tenant_id)
            .offset(offset)
            .limit(size)
        )

        results = await self.session.exec(query)
        metadata_with_counts = results.all()

        # Build result list with count
        items = []
        for metadata_entity, count in metadata_with_counts:
            metadata_dict = metadata_entity.model_dump()
            metadata_dict["count"] = int(count) if count else 0
            items.append(metadata_dict)

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=items,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_metadata(
        self,
        kb_id: str,
        metadata_create: KbMetadataEntityCreate,
        tenant_id: str,
    ) -> KbMetadataEntity:
        """
        Create a new Metadata entity.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
            metadata_data: Metadata creation data

        Returns:
            Created KbMetadataEntity (not yet committed)

        Raises:
            ValueError: If metadata name already exists (IntegrityError converted)
        """

        if metadata_create.name in DEFAULT_METADATA_KEYS:
            raise ValueError(f"Metadata name '{metadata_create.name}' is a system reserved name and cannot be created.")

        metadata_entity = KbMetadataEntity.model_validate(
            metadata_create, update={"kb_id": kb_id, "tenant_id": tenant_id}
        )

        self.session.add(metadata_entity)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(metadata_entity)

            logger.info(
                f"Created Metadata entity: {metadata_entity.id} (name: {metadata_entity.name}, kb_id: {kb_id})"
            )
            return metadata_entity

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Metadata: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Metadata name '{metadata_create.name}' already exists in knowledge base."
                ) from e
            elif "Duplicate entry" in str(e.orig):
                raise ValueError(
                    f"Metadata name '{metadata_create.name}' already exists in knowledge base."
                ) from e
            else:
                raise ValueError(f"Metadata creation failed: {e}") from e

    async def update_metadata(
        self,
        kb_id: str,
        metadata_id: str,
        update_data: KbMetadataEntityCreate,
        tenant_id: str,
    ) -> KbMetadataEntity:
        """
        Update an existing Metadata entity.
        Note: This only updates the metadata entity itself.
        For updating related files' file_metadata JSON field, use RagService.update_metadata.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata entity ID
            update_data: Updated Metadata data

        Returns:
            Updated KbMetadataEntity (not yet committed)

        Raises:
            ValueError: If Metadata entity not found
        """
        metadata_entity = await self.get_metadata(kb_id, metadata_id, tenant_id)
        if not metadata_entity:
            raise ValueError(f"Metadata '{metadata_id}' does not exist.")

        try:
            # Update metadata_entity
            metadata_entity.name = update_data.name
            metadata_entity.value_type = update_data.value_type
            metadata_entity.description = update_data.description
            metadata_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.session.add(metadata_entity)

            # Flush to ensure changes are staged
            await self.session.flush()
            await self.session.refresh(metadata_entity)

            logger.info(
                f"Updated Metadata entity: {metadata_entity.id} (name: {metadata_entity.name})"
            )
            return metadata_entity
        except IntegrityError as e:
            logger.error(f"IntegrityError when updating Metadata: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Metadata name '{update_data.name}' already exists in knowledge base."
                ) from e
            elif "Duplicate entry" in str(e.orig):
                raise ValueError(
                    f"Metadata name '{update_data.name}' already exists in knowledge base."
                ) from e
            else:
                raise ValueError(f"Metadata update failed: {e}") from e

    async def delete_metadata(self, kb_id: str, metadata_id: str, tenant_id: str) -> None:
        """
        Delete a Metadata entity.
        Note: This only deletes the metadata entity itself.
        For deleting related FileMetadataEntity and updating files' file_metadata JSON,
        use RagService.delete_metadata.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata entity ID

        Raises:
            ValueError: If Metadata entity not found
        """
        metadata_entity = await self.get_metadata(kb_id, metadata_id, tenant_id)
        if not metadata_entity:
            raise ValueError(f"Metadata '{metadata_id}' does not exist.")

        # Delete Metadata entity only
        await self.session.delete(metadata_entity)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted Metadata entity: {metadata_id} (name: {metadata_entity.name})")


    async def get_all_metadata(self, kb_id: str, tenant_id: str) -> List[KbMetadataEntity]:
        """
        Get all Metadata entities for a knowledgebase without pagination.

        Args:
            kb_id: Knowledgebase ID

        Returns:
            List of all KbMetadataEntity for the knowledgebase
        """
        statement = select(KbMetadataEntity).where(KbMetadataEntity.kb_id == kb_id, KbMetadataEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        return list(results.all())


    async def batch_delete_metadata(
        self, kb_id: str, metadata_ids: List[str], tenant_id: str
    ) -> None:
        """
        Delete multiple Metadata entities in batch.
        Note: This only deletes the metadata entities themselves.
        For deleting related FileMetadataEntity and updating files' file_metadata JSON,
        use RagService.batch_delete_metadata.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
            metadata_ids: List of metadata entity IDs to delete

        Raises:
            ValueError: If any metadata not found or doesn't belong to kb_id
        """
        if not metadata_ids:
            return


        # Directly delete all metadata for this knowledgebase
        stmt = delete(KbMetadataEntity).where(KbMetadataEntity.id.in_(metadata_ids))
        stmt = stmt.where(KbMetadataEntity.kb_id == kb_id)
        stmt = stmt.where(KbMetadataEntity.tenant_id == tenant_id)
        await self.session.execute(stmt)

        # Flush to ensure deletions are staged
        await self.session.flush()

    async def delete_metadata_by_kb_id(self, kb_id: str, tenant_id: str) -> None:
        """
        Delete all Metadata entities for a knowledgebase.
        Note: This directly deletes all metadata without querying first.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
        """
        # Directly delete all metadata for this knowledgebase
        stmt = delete(KbMetadataEntity).where(KbMetadataEntity.kb_id == kb_id, KbMetadataEntity.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        deleted_count = result.rowcount

        # Flush to ensure deletions are staged
        await self.session.flush()

        logger.info(
            f"Deleted {deleted_count} Metadata entities from knowledgebase {kb_id}"
        )
