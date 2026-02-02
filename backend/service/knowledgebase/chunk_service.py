"""Chunk Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select, func, delete
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.knowledgebase.chunk import KbChunkEntity, KbChunkModel
from common.chat.response_model import PagedResult
from memory.utils import estimate_tokens_in_text
from pairag.file.store.file_store_helper import file_store
import re

MARKDOWN_IMAGE_PATTERN = r'!\[.*?\]\((.*?)\)\s*\n*\s*图片的描述:\s*(.*?)(?=\n\n|$)'


class ChunkService:
    """Service layer for Chunk entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ChunkService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_chunk(self, chunk_id: str, tenant_id: str) -> Optional[KbChunkEntity]:
        """
        Get a single Chunk entity by ID.

        Args:
            chunk_id: Chunk entity ID

        Returns:
            KbChunkEntity if found, None otherwise
        """
        result = await self.session.exec(select(KbChunkEntity).where(KbChunkEntity.id == chunk_id, KbChunkEntity.tenant_id == tenant_id))
        return result.first()

    async def list_chunks(
        self,
        kb_id: str,
        file_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
    ) -> PagedResult[List[KbChunkEntity]]:
        """
        List Chunk entities with pagination.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            page: Page number (1-indexed)
            size: Page size

        Returns:
            PagedResult containing list of KbChunkEntity and pagination metadata
        """
        # Build base query
        base_query = select(KbChunkEntity).where(
            KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id, KbChunkEntity.tenant_id == tenant_id
        )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results (ordered by index)
        offset = (page - 1) * size
        paginated_query = (
            base_query.order_by(KbChunkEntity.index).offset(offset).limit(size)
        )
        results = await self.session.exec(paginated_query)
        chunks = list(results.all())
        for chunk_entity in chunks:
            origin_text = chunk_entity.text
            pattern = MARKDOWN_IMAGE_PATTERN
            matches = re.findall(pattern, origin_text, re.DOTALL)
            chunk_entity.chunk_metadata["images_info"] = [{"url": await file_store.get_url_async(file_path=src, tenant_id=tenant_id), "desc": desc} for src, desc in matches]
        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=chunks,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_chunk(
        self,
        kb_id: str,
        file_id: str,
        text: str,
        tenant_id: str,
        chunk_metadata: Optional[dict] = None,
        file_metadata: Optional[dict] = None,
        active: bool = True,
    ) -> KbChunkEntity:
        """
        Create a new Chunk entity.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            text: Chunk text content
            chunk_metadata: Optional chunk metadata
            file_metadata: Optional file metadata to merge
            active: Whether the chunk is active (default: True)

        Returns:
            Created KbChunkEntity (not yet committed)

        Raises:
            ValueError: If file doesn't exist or doesn't belong to kb_id
        """
        # Get max index for the file
        max_index_result = await self.session.exec(
            select(func.max(KbChunkEntity.index)).where(
                KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id, KbChunkEntity.tenant_id == tenant_id
            )
        )
        max_index = max_index_result.one_or_none() or -1
        new_index = max_index + 1

        # Build chunk_metadata: merge file_metadata + chunk_metadata + token_count
        merged_metadata = {}
        if file_metadata:
            merged_metadata.update(file_metadata)
        if chunk_metadata:
            merged_metadata.update(chunk_metadata)
        merged_metadata["doc_id"] = file_id

        # Calculate token_count
        token_count = estimate_tokens_in_text(text)
        merged_metadata["token_count"] = token_count

        # Create new chunk entity
        new_chunk = KbChunkEntity(
            kb_id=kb_id,
            file_id=file_id,
            text=text,
            chunk_metadata=merged_metadata,
            index=new_index,
            active=active,
            file_part=0,
            file_version=0,
            tenant_id=tenant_id,
        )

        self.session.add(new_chunk)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(new_chunk)

            logger.info(
                f"Created Chunk entity: {new_chunk.id} (index: {new_chunk.index}, file_id: {file_id})"
            )
            return new_chunk

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Chunk: {e.orig}")
            raise ValueError(f"Chunk creation failed: {e}") from e

    async def update_chunk(
        self,
        chunk_id: str,
        kb_id: str,
        file_id: str,
        tenant_id: str,
        new_chunk: KbChunkModel,
    ) -> KbChunkEntity:
        """
        Update an existing Chunk entity.
        Note: Caller is responsible for committing the session.

        Args:
            chunk_id: Chunk entity ID
            kb_id: Knowledgebase ID (for validation)
            file_id: File ID (for validation)

        Returns:
            Updated KbChunkEntity (not yet committed)

        Raises:
            ValueError: If Chunk entity not found or doesn't belong to kb_id/file_id
        """
        result = await self.session.exec(select(KbChunkEntity).where(KbChunkEntity.id == chunk_id, KbChunkEntity.tenant_id == tenant_id))
        chunk = result.first()
        if not chunk:
            raise ValueError(f"Chunk '{chunk_id}' does not exist.")

        if chunk.kb_id != kb_id or chunk.file_id != file_id:
            raise ValueError(
                f"Chunk '{chunk_id}' does not belong to knowledge base '{kb_id}' or file '{file_id}'."
            )

        logger.info(f"Updating Chunk {chunk_id} with fields: {new_chunk.model_dump()}")


        if chunk.text != new_chunk.text:
            chunk.text = new_chunk.text
            token_count = estimate_tokens_in_text(new_chunk.text)
            if chunk.chunk_metadata:
                chunk.chunk_metadata["token_count"] = token_count
            else:
                chunk.chunk_metadata = {"token_count": token_count}

        chunk.active = new_chunk.active
        logger.info(f"Update chunk active to {chunk.active} for {chunk.id}")

        self.session.add(chunk)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(chunk)

        logger.info(f"Updated Chunk entity: {chunk.id} (index: {chunk.index})")
        return chunk

    async def delete_chunk(self, kb_id: str, file_id: str, chunk_id: str, tenant_id: str) -> None:
        """
        Delete a Chunk entity.
        Note: Caller is responsible for committing the session.

        Args:
            chunk_id: Chunk entity ID
            kb_id: Knowledgebase ID (for validation)
            file_id: File ID (for validation)

        Raises:
            ValueError: If Chunk entity not found or doesn't belong to kb_id/file_id
        """
        result = await self.session.exec(select(KbChunkEntity).where(KbChunkEntity.id == chunk_id, KbChunkEntity.tenant_id == tenant_id))
        chunk = result.first()
        if not chunk:
            raise ValueError(f"Chunk '{chunk_id}' does not exist.")

        if chunk.kb_id != kb_id or chunk.file_id != file_id:
            raise ValueError(
                f"Chunk '{chunk_id}' does not belong to knowledge base '{kb_id}' or file '{file_id}'."
            )

        # Delete from database (staged, not committed)
        await self.session.delete(chunk)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted Chunk entity: {chunk_id} (index: {chunk.index})")

    async def get_chunks_by_file(
        self, kb_id: str, file_id: str, tenant_id: str
    ) -> List[KbChunkEntity]:
        """
        Get all Chunk entities for a file without pagination.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID

        Returns:
            List of all KbChunkEntity for the file (ordered by index)
        """
        statement = (
            select(KbChunkEntity)
            .where(
                KbChunkEntity.kb_id == kb_id, KbChunkEntity.file_id == file_id, KbChunkEntity.tenant_id == tenant_id
            )
            .order_by(KbChunkEntity.index)
        )
        results = await self.session.exec(statement)
        return list(results.all())

    async def get_chunks_by_kb(self, kb_id: str, tenant_id: str) -> List[KbChunkEntity]:
        """
        Get all Chunk entities for a knowledgebase without pagination.

        Args:
            kb_id: Knowledgebase ID

        Returns:
            List of all KbChunkEntity for the knowledgebase
        """
        statement = select(KbChunkEntity).where(KbChunkEntity.kb_id == kb_id, KbChunkEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        return list(results.all())


    async def delete_chunks_from_kb(self, kb_id: str, tenant_id: str) -> List[str]:
        """
        Delete all Chunk entities for a knowledgebase.
        Note: This directly deletes all chunks without querying first.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase ID
        """
        if not kb_id:
            return []

        select_statement = select(KbChunkEntity).where(
            KbChunkEntity.kb_id == kb_id, KbChunkEntity.tenant_id == tenant_id
        )
        chunks = await self.session.exec(select_statement)
        chunk_ids = [chunk.id for chunk in chunks]

        if len(chunk_ids) == 0:
            return []

        # Directly delete all chunks for this knowledgebase
        delete_statement = delete(KbChunkEntity).where(KbChunkEntity.id.in_(chunk_ids))
        result = await self.session.execute(delete_statement)
        deleted_count = result.rowcount

        # Flush to ensure deletions are staged
        await self.session.flush()

        logger.info(
            f"Deleted {deleted_count} Chunk entities from knowledgebase {kb_id}"
        )

        return chunk_ids

    async def delete_chunks_from_file(self, file_id: str, kb_id: str, tenant_id: str) -> List[str]:
        """
        Delete all Chunk entities for a file.
        """
        if not file_id or not kb_id:
            return []

        select_statement = select(KbChunkEntity).where(
            KbChunkEntity.file_id == file_id,
            KbChunkEntity.kb_id == kb_id,
            KbChunkEntity.tenant_id == tenant_id
        )
        chunks = await self.session.exec(select_statement)
        chunk_ids = [chunk.id for chunk in chunks]

        if len(chunk_ids) == 0:
            return []

        # Directly delete all chunks for this file
        delete_statement = delete(KbChunkEntity).where(
            KbChunkEntity.id.in_(chunk_ids)
        )
        result = await self.session.execute(delete_statement)
        deleted_count = result.rowcount
        # Flush to ensure deletions are staged
        await self.session.flush()

        logger.info(f"Deleted {deleted_count} Chunk entities from file {file_id} in knowledgebase {kb_id}")
        return chunk_ids
