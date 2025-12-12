"""FileMetadataRelation Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select, delete
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

from db.models.knowledgebase.metadata import FileMetadataEntity


class FileMetadataRelationService:
    """Service layer for FileMetadataEntity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize FileMetadataRelationService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_file_metadata_relations(
        self, kb_id: str, tenant_id: str, file_id: Optional[str] = None, metadata_id: Optional[str] = None,
    ) -> List[FileMetadataEntity]:
        """
        Get FileMetadataEntity relations.

        Args:
            kb_id: Knowledgebase ID
            file_id: Optional file ID to filter by
            metadata_id: Optional metadata ID to filter by

        Returns:
            List of FileMetadataEntity
        """
        statement = select(FileMetadataEntity).where(
            FileMetadataEntity.kb_id == kb_id,
            FileMetadataEntity.tenant_id == tenant_id
        )
        if file_id:
            statement = statement.where(FileMetadataEntity.file_id == file_id)
        if metadata_id:
            statement = statement.where(FileMetadataEntity.metadata_id == metadata_id)

        results = await self.session.exec(statement)
        return list(results.all())


    async def get_file_metadata_relations_by_metadata_id(
        self, kb_id: str, metadata_id: str, tenant_id: str
    ) -> List[FileMetadataEntity]:
        """
        Get all FileMetadataEntity relations for a metadata.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata ID

        Returns:
            List of FileMetadataEntity
        """
        return await self.get_file_metadata_relations(
            kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id
        )

    async def get_file_metadata_relations_by_file_id(
        self, kb_id: str, file_id: str, tenant_id: str
    ) -> List[FileMetadataEntity]:
        """
        Get all FileMetadataEntity relations for a file.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID

        Returns:
            List of FileMetadataEntity
        """
        return await self.get_file_metadata_relations(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)

    async def create_file_metadata_relation(
        self, kb_id: str, file_id: str, metadata_id: str, tenant_id: str
    ) -> FileMetadataEntity:
        """
        Create a new FileMetadataEntity relation.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            metadata_id: Metadata ID

        Returns:
            Created FileMetadataEntity (not yet committed)
        """
        file_metadata_relation = FileMetadataEntity(
            kb_id=kb_id, file_id=file_id, metadata_id=metadata_id, tenant_id=tenant_id
        )
        self.session.add(file_metadata_relation)

        # Flush to get the ID, but don't commit
        await self.session.flush()
        await self.session.refresh(file_metadata_relation)

        logger.info(
            f"Created FileMetadataEntity relation: {file_metadata_relation.id} (file_id: {file_id}, metadata_id: {metadata_id})"
        )
        return file_metadata_relation

    async def delete_file_metadata_relation(
        self, kb_id: str, file_id: str, metadata_id: str, tenant_id: str
    ) -> None:
        """
        Delete a FileMetadataEntity relation.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            metadata_id: Metadata ID

        Raises:
            ValueError: If relation not found
        """
        statement = (
            select(FileMetadataEntity)
            .where(FileMetadataEntity.kb_id == kb_id)
            .where(FileMetadataEntity.tenant_id == tenant_id)
            .where(FileMetadataEntity.file_id == file_id)
            .where(FileMetadataEntity.metadata_id == metadata_id)
        )
        result = await self.session.exec(statement)
        relation = result.first()

        if not relation:
            raise ValueError(
                f"FileMetadataEntity relation not found (file_id: {file_id}, metadata_id: {metadata_id})"
            )

        await self.session.delete(relation)
        await self.session.flush()

        logger.info(
            f"Deleted FileMetadataEntity relation (file_id: {file_id}, metadata_id: {metadata_id})"
        )

    async def delete_file_metadata_relations_by_file_id(
        self, kb_id: str, file_id: str, tenant_id: str
    ) -> None:
        """
        Delete all FileMetadataEntity relations for a file.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
        """
        if not file_id:
            return

        # Directly delete all file metadata relations for this file
        stmt = delete(FileMetadataEntity).where(
            FileMetadataEntity.file_id == file_id,
            FileMetadataEntity.kb_id == kb_id,
            FileMetadataEntity.tenant_id == tenant_id
        )
        result = await self.session.execute(stmt)
        deleted_count = result.rowcount

        # Flush to ensure deletions are staged
        await self.session.flush()

        logger.info(
            f"Deleted {deleted_count} FileMetadataEntity relations for file {file_id}"
        )

    async def delete_file_metadata_relations_by_metadata_id(
        self, kb_id: str, metadata_id: str, tenant_id: str
    ) -> None:
        """
        Delete all FileMetadataEntity relations for a metadata.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata ID
        """
        if not metadata_id:
            return

        # Directly delete all file metadata relations for this metadata
        stmt = delete(FileMetadataEntity).where(
            FileMetadataEntity.metadata_id == metadata_id,
            FileMetadataEntity.kb_id == kb_id,
            FileMetadataEntity.tenant_id == tenant_id
        )
        result = await self.session.execute(stmt)
        deleted_count = result.rowcount

        # Flush to ensure deletions are staged
        await self.session.flush()

        logger.info(
            f"Deleted {deleted_count} FileMetadataEntity relations for metadata {metadata_id}"
        )

    async def delete_file_metadata_relations_by_kb_id(self, kb_id: str, tenant_id: str) -> None:
        """
        Delete all FileMetadataEntity relations for a knowledgebase.

        Args:
            kb_id: Knowledgebase ID
        """
        if not kb_id:
            return

        # Directly delete all file metadata relations for this knowledgebase
        stmt = delete(FileMetadataEntity).where(FileMetadataEntity.kb_id == kb_id, FileMetadataEntity.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        deleted_count = result.rowcount

        # Flush to ensure deletions are staged
        await self.session.flush()

        logger.info(
            f"Deleted {deleted_count} FileMetadataEntity relations for knowledgebase {kb_id}"
        )
