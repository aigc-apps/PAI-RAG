"""Vector Table Mapping Service - Manages vector table name mapping."""

from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from service.cache.redis_cache import cache_manager, vector_table_name_key
from loguru import logger
import traceback
from db.models.knowledgebase.vector_table_mapping import (
    VectorTableMappingEntity,
    generate_vector_table_name,
)


class VectorTableMappingService:
    """Service layer for Vector Table Mapping operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize VectorTableMappingService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_vector_table_name(self, tenant_id: str, kb_id: str) -> str:
        """
        Get the vector table name for a given tenant_id and kb_id.

        If a mapping exists in the database, return the stored table_name.
        Otherwise, generate a new table name and store the mapping.

        Args:
            tenant_id: The tenant ID
            kb_id: The knowledgebase ID

        Returns:
            The vector table name
        """
        cache_key = vector_table_name_key(tenant_id, kb_id)
        table_name = await cache_manager.get_cache().get(cache_key)
        if table_name:
            logger.debug(f"Found vector table name in cache for tenant {tenant_id} and kb {kb_id}: {table_name}")
            return table_name

        # Try to find existing mapping
        mapping = await self._get_mapping(tenant_id, kb_id)

        if mapping:
            logger.debug(f"Found existing vector table mapping: {mapping.table_name}")
            await cache_manager.get_cache().set(cache_key, mapping.table_name)
            return mapping.table_name

        # Generate new table name
        table_name = generate_vector_table_name(tenant_id, kb_id)
        logger.debug(f"Generated new vector table name: {table_name}")

        # Store the mapping
        await self._create_mapping(tenant_id, kb_id, table_name)
        await cache_manager.get_cache().set(cache_key, table_name)
        return table_name

    async def _get_mapping(
        self, tenant_id: str, kb_id: str
    ) -> Optional[VectorTableMappingEntity]:
        """
        Get a mapping by tenant_id and kb_id.

        Args:
            tenant_id: The tenant ID
            kb_id: The knowledgebase ID

        Returns:
            VectorTableMappingEntity if found, None otherwise
        """
        statement = select(VectorTableMappingEntity).where(
            VectorTableMappingEntity.tenant_id == tenant_id,
            VectorTableMappingEntity.kb_id == kb_id,
        )
        result = await self.session.exec(statement)
        return result.first()

    async def _create_mapping(
        self, tenant_id: str, kb_id: str, table_name: str
    ) -> VectorTableMappingEntity:
        """
        Create a new vector table mapping.

        Args:
            tenant_id: The tenant ID
            kb_id: The knowledgebase ID
            table_name: The vector table name

        Returns:
            The created VectorTableMappingEntity
        """
        mapping = VectorTableMappingEntity(
            tenant_id=tenant_id,
            kb_id=kb_id,
            table_name=table_name,
        )
        try:
            self.session.add(mapping)
            await self.session.commit()
            await self.session.refresh(mapping)
            logger.info(f"Created vector table mapping: tenant={tenant_id}, kb={kb_id}, table={table_name}")
            return mapping
        except Exception as ex:
            logger.error(f"Failed to create vector table mapping: {traceback.format_exc()}")
            await self.session.rollback()
            if "UniqueViolationError" in str(ex.orig) or "Duplicate entry" in str(ex.orig):
                pass
            else:
                raise ValueError(f"Failed to create vector table mapping: {ex}") from ex

    async def delete_mapping(self, tenant_id: str, kb_id: str) -> bool:
        """
        Delete a vector table mapping.

        Args:
            tenant_id: The tenant ID
            kb_id: The knowledgebase ID

        Returns:
            True if mapping was deleted, False if not found
        """
        mapping = await self._get_mapping(tenant_id, kb_id)
        if mapping:
            cache_key = vector_table_name_key(tenant_id, kb_id)
            await cache_manager.get_cache().delete(cache_key)
            await self.session.delete(mapping)
            await self.session.commit()
            logger.info(f"Deleted vector table mapping: tenant={tenant_id}, kb={kb_id}")
            return True
        return False

    async def get_table_name_if_exists(self, tenant_id: str, kb_id: str) -> Optional[str]:
        """
        Get the vector table name only if mapping exists (without creating new one).

        Args:
            tenant_id: The tenant ID
            kb_id: The knowledgebase ID

        Returns:
            The table name if mapping exists, None otherwise
        """
        cache_key = vector_table_name_key(tenant_id, kb_id)
        table_name = await cache_manager.get_cache().get(cache_key)
        if table_name:
            logger.debug(f"Found vector table name in cache for tenant {tenant_id} and kb {kb_id}: {table_name}")
            return table_name

        mapping = await self._get_mapping(tenant_id, kb_id)
        if mapping:
            await cache_manager.get_cache().set(cache_key, mapping.table_name)
            return mapping.table_name
        return None
