"""Knowledgebase Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import or_, and_
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.knowledgebase.knowledgebase import (
    KnowledgebaseCreate,
    KbEntity,
    RetrievalConfig,
)
from pairag.file.nodeparsers.file_parser import ChunkConfig
from db.models.knowledgebase.file import KbFileEntity
from common.chat.response_model import PagedResult
from service.cache.redis_cache import cache_manager, kb_key, kb_name_key
from common.knowledgebase.constants import FAQ_KNOWLEDGEBASE_NAME

class KnowledgebaseService:
    """Service layer for Knowledgebase entity CRUD operations using dependency injection."""

    def __init__(
        self,
        session: AsyncSession,
    ):
        """
        Initialize KnowledgebaseService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_knowledgebase(self, kb_id: str, tenant_id: str) -> Optional[KbEntity]:
        """
        Get a single Knowledgebase entity by ID.

        Args:
            kb_id: Knowledgebase entity ID

        Returns:
            KbEntity if found, None otherwise
        """
        cache_key = kb_key(tenant_id, kb_id)
        try:
            kb_data = await cache_manager.get_cache().get(cache_key)
            if kb_data:
                logger.info(f"Get knowledgebase entity from cache: {kb_id}")
                kb_entity = KbEntity.model_validate(kb_data)
                return kb_entity
        except Exception as e:
            logger.warning(f"Cache get operation failed for {cache_key}: {e}")

        result = await self.session.exec(select(KbEntity).where(KbEntity.id == kb_id, KbEntity.tenant_id == tenant_id))
        kb_entity = result.first()
        if kb_entity:
            try:
                await cache_manager.get_cache().set(cache_key, kb_entity.model_dump(mode="json"))
            except Exception as e:
                logger.warning(f"Cache set operation failed for {cache_key}: {e}")
        return kb_entity

    async def get_knowledgebase_by_name(self, name: str, tenant_id: str) -> Optional[KbEntity]:
        """
        Get a single Knowledgebase entity by name.

        Args:
            name: Knowledgebase name

        Returns:
            KbEntity if found, None otherwise
        """
        cache_key = kb_name_key(tenant_id, name)
        try:
            kb_data = await cache_manager.get_cache().get(cache_key)
            if kb_data:
                logger.info(f"Get knowledgebase entity from cache: {name}")
                kb_entity = KbEntity.model_validate(kb_data)
                return kb_entity
        except Exception as e:
            logger.warning(f"Cache get operation failed for {cache_key}: {e}, falling back to database")

        statement = select(KbEntity).where(KbEntity.name == name, KbEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        kb_entity = result.first()
        if kb_entity:
            try:
                await cache_manager.get_cache().set(cache_key, kb_entity.model_dump(mode="json"))
            except Exception as e:
                logger.warning(f"Cache set operation failed for {cache_key}: {e}")
        return kb_entity

    async def get_knowledgebases_by_ids(self, tenant_id: str, kb_ids: List[str]) -> List[KbEntity]:
        """
        Get Knowledgebase entities by IDs.
        """
        if not kb_ids:
            return []

        statement = select(KbEntity).where(KbEntity.id.in_(kb_ids), KbEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        return list(result.all())

    async def list_knowledgebases(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        query: Optional[str] = None,
    ) -> PagedResult[List[dict]]:
        """
        List Knowledgebase entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            query: Optional search query (searches in id, name, description)

        Returns:
            PagedResult containing list of KbEntity with file_count and pagination metadata
        """
        conditions = [KbEntity.tenant_id == tenant_id]
        conditions.append(~KbEntity.name.like(f"%_{FAQ_KNOWLEDGEBASE_NAME}"))

        base_condition = and_(*conditions)
        # Add search condition if provided
        if query:
            query_lower = query.lower()
            search_condition = or_(
                func.lower(KbEntity.id).like(f"%{query_lower}%"),
                func.lower(KbEntity.name).like(f"%{query_lower}%"),
                func.lower(func.coalesce(KbEntity.description, "")).like(
                    f"%{query_lower}%"
                ),
            )
            where_condition = base_condition & search_condition
        else:
            where_condition = base_condition

        # Get total count
        count_query = select(func.count(KbEntity.id)).where(where_condition)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # 考虑性能，这里直接使用子查询计算文件数量，而不是通过 FileService
        # 子查询可以在一次数据库查询中完成，性能更优
        file_count_subquery = (
            select(
                KbFileEntity.kb_id,
                func.count(KbFileEntity.id).label("file_count"),
            )
            .group_by(KbFileEntity.kb_id)
            .subquery()
        )

        # Get paginated results with file count
        offset = (page - 1) * size
        paginated_query = (
            select(
                KbEntity,
                func.coalesce(file_count_subquery.c.file_count, 0).label("file_count"),
            )
            .outerjoin(file_count_subquery, KbEntity.id == file_count_subquery.c.kb_id)
            .where(where_condition)
            .order_by(KbEntity.created_at.desc(), KbEntity.id.asc())
            .offset(offset)
            .limit(size)
        )

        results = await self.session.exec(paginated_query)
        kb_entities_with_counts = results.all()

        # Build result list with file_count
        items = []
        for kb_entity, file_count in kb_entities_with_counts:
            kb_dict = (
                kb_entity.model_dump(mode="json")
                if hasattr(kb_entity, "model_dump")
                else kb_entity.__dict__
            )
            kb_dict["file_count"] = int(file_count) if file_count else 0
            items.append(kb_dict)

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=items,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_knowledgebase(
        self, kb_data: KnowledgebaseCreate, tenant_id: str
    ) -> KbEntity:
        """
        Create a new Knowledgebase entity.
        Note: Caller is responsible for committing the session.

        Args:
            kb_data: Knowledgebase creation data

        Returns:
            Created KbEntity (not yet committed)

        Raises:
            ValueError: If name already exists (IntegrityError converted)
        """
        # Convert configs to dict
        kb_data.chunk_config = (
            (kb_data.chunk_config or ChunkConfig()).model_dump()
        )
        kb_data.retrieval_config = (
            (kb_data.retrieval_config or RetrievalConfig()).model_dump()
        )

        try:
            knowledgebase = KbEntity.model_validate(kb_data, update={"tenant_id": tenant_id})
            self.session.add(knowledgebase)

            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(knowledgebase)

            logger.info(
                f"Created Knowledgebase entity: {knowledgebase.id} (name: {knowledgebase.name})"
            )
            return knowledgebase
        except ValueError as e:
            logger.error(f"ValueError when creating Knowledgebase: {e}")
            raise ValueError(f"ValueError when creating Knowledgebase: {e}") from e

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Knowledgebase: {e.orig}")
            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Knowledgebase name '{kb_data.name}' already exists."
                ) from e
            elif "Duplicate entry" in str(e.orig):
                raise ValueError(f"Knowledgebase name '{kb_data.name}' already exists.") from e
            else:
                raise ValueError(f"Knowledgebase creation failed: {e.orig}") from e

    async def update_knowledgebase(
        self, kb_id: str, update_data: KnowledgebaseCreate, tenant_id: str
    ) -> KbEntity:
        """
        Update an existing Knowledgebase entity.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase entity ID
            update_data: Updated Knowledgebase data

        Returns:
            Updated KbEntity (not yet committed)

        Raises:
            ValueError: If Knowledgebase entity not found
        """
        cache_key = kb_key(tenant_id, kb_id)
        try:
            await cache_manager.get_cache().delete(cache_key)
        except Exception as e:
            logger.warning(f"Cache delete operation failed for {cache_key}: {e}")

        result = await self.session.exec(select(KbEntity).where(KbEntity.id == kb_id, KbEntity.tenant_id == tenant_id))
        knowledgebase = result.first()
        if not knowledgebase:
            raise ValueError(f"Knowledgebase '{kb_id}' does not exist.")

        cache_name_key = kb_name_key(tenant_id, knowledgebase.name)
        try:
            await cache_manager.get_cache().delete(cache_name_key)
        except Exception as e:
            logger.warning(f"Cache delete operation failed for {cache_name_key}: {e}")

        try:

            logger.info(f"Updating Knowledgebase {kb_id} with data: {update_data}")

            # Update fields
            if update_data.name is not None:
                knowledgebase.name = update_data.name
            if update_data.description is not None:
                knowledgebase.description = update_data.description
            if update_data.embedding_model is not None:
                knowledgebase.embedding_model = update_data.embedding_model
            if update_data.embedding_provider_name is not None:
                knowledgebase.embedding_provider_name = update_data.embedding_provider_name
            if update_data.chunk_config is not None:
                knowledgebase.chunk_config = update_data.chunk_config.model_dump()
            if update_data.retrieval_config is not None:
                knowledgebase.retrieval_config = update_data.retrieval_config.model_dump()

            knowledgebase.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            self.session.add(knowledgebase)

            # Flush to ensure changes are staged
            await self.session.flush()
            await self.session.refresh(knowledgebase)

            # Note: Cache will be written after transaction commit in API layer
            # to ensure cache consistency with database

            logger.info(
                f"Updated Knowledgebase entity: {knowledgebase.id} (name: {knowledgebase.name})"
            )
        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Knowledgebase: {e.orig}")
            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Knowledgebase name '{update_data.name}' already exists."
                ) from e
            elif "Duplicate entry" in str(e.orig):
                raise ValueError(f"Knowledgebase name '{update_data.name}' already exists.") from e
            else:
                raise ValueError(f"Knowledgebase creation failed: {e.orig}") from e
        return knowledgebase

    async def delete_knowledgebase(self, kb_id: str, tenant_id: str) -> None:
        """
        Delete a Knowledgebase entity.
        Note: This only deletes the knowledgebase entity itself.
        For deleting related entities (files, chunks, metadata), use RagService.delete_knowledgebase.
        Note: Caller is responsible for committing the session.

        Args:
            kb_id: Knowledgebase entity ID

        Raises:
            ValueError: If Knowledgebase entity not found
        """
        result = await self.session.exec(select(KbEntity).where(KbEntity.id == kb_id, KbEntity.tenant_id == tenant_id))
        knowledgebase = result.first()
        if not knowledgebase:
            raise ValueError(f"Knowledgebase '{kb_id}' does not exist.")

        # Delete both ID-based and name-based cache entries
        cache_key = kb_key(tenant_id, kb_id)
        cache_name_key = kb_name_key(tenant_id, knowledgebase.name)
        try:
            await cache_manager.get_cache().delete(cache_key)
        except Exception as e:
            logger.warning(f"Cache delete operation failed for {cache_key}: {e}")
        try:
            await cache_manager.get_cache().delete(cache_name_key)
        except Exception as e:
            logger.warning(f"Cache delete operation failed for {cache_name_key}: {e}")

        # Delete knowledgebase entity only
        await self.session.delete(knowledgebase)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted Knowledgebase entity: {kb_id} (name: {knowledgebase.name})")

    async def get_all_knowledgebases(self, tenant_id: str) -> List[KbEntity]:
        """
        Get all Knowledgebase entities without pagination.

        Returns:
            List of all KbEntity
        """
        statement = select(KbEntity).where(KbEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        return list(results.all())

    async def write_cache_after_commit(self, kb_entity: KbEntity, tenant_id: str) -> None:
        """
        Write cache after database transaction commit.
        This ensures cache consistency with database.

        Args:
            kb_entity: Knowledgebase entity to cache
            tenant_id: Tenant ID
        """
        cache_key = kb_key(tenant_id, kb_entity.id)
        cache_name_key = kb_name_key(tenant_id, kb_entity.name)
        try:
            await cache_manager.get_cache().set(cache_key, kb_entity.model_dump(mode="json"))
        except Exception as e:
            logger.warning(f"Cache set operation failed for {cache_key}: {e}")
        try:
            await cache_manager.get_cache().set(cache_name_key, kb_entity.model_dump(mode="json"))
        except Exception as e:
            logger.warning(f"Cache set operation failed for {cache_name_key}: {e}")
        logger.info(f"Written cache for knowledgebase {kb_entity.id} (name: {kb_entity.name}) after commit")

    async def delete_cache_on_rollback(self, kb_id: str, tenant_id: str, kb_name: Optional[str] = None) -> None:
        """
        Delete cache entries when database transaction rolls back.
        This ensures cache consistency with database.

        Args:
            kb_id: Knowledgebase ID
            tenant_id: Tenant ID
            kb_name: Optional knowledgebase name (if known)
        """
        cache_key = kb_key(tenant_id, kb_id)
        try:
            await cache_manager.get_cache().delete(cache_key)
        except Exception as e:
            logger.warning(f"Cache delete operation failed for {cache_key}: {e}")
        if kb_name:
            cache_name_key = kb_name_key(tenant_id, kb_name)
            try:
                await cache_manager.get_cache().delete(cache_name_key)
            except Exception as e:
                logger.warning(f"Cache delete operation failed for {cache_name_key}: {e}")
        logger.info(f"Deleted cache for knowledgebase {kb_id} on rollback")
