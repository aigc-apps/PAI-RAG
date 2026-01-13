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
        try:
            table_name = await cache_manager.get_cache().get(cache_key)
            if table_name:
                logger.debug(f"Found vector table name in cache for tenant {tenant_id} and kb {kb_id}: {table_name}")
                return table_name
        except Exception as e:
            logger.warning(f"Cache get operation failed for {cache_key}: {e}, falling back to database")

        # Try to find existing mapping
        mapping = await self._get_mapping(tenant_id, kb_id)

        if mapping:
            logger.debug(f"Found existing vector table mapping: {mapping.table_name}")
            try:
                await cache_manager.get_cache().set(cache_key, mapping.table_name)
            except Exception as e:
                logger.warning(f"Cache set operation failed for {cache_key}: {e}")
            return mapping.table_name

        # Generate new table name
        table_name = generate_vector_table_name(tenant_id, kb_id)
        logger.debug(f"Generated new vector table name: {table_name}")

        # Store the mapping (with retry logic for race conditions)
        try:
            mapping = await self._create_mapping(tenant_id, kb_id, table_name)
            # Use the actual table_name from the mapping (in case we got an existing one)
            table_name = mapping.table_name
        except Exception as create_ex:
            # If creation failed, try one more time to get existing mapping
            # (another concurrent task might have created it)
            logger.warning(f"Failed to create mapping, retrying to get existing: {create_ex}")
            existing_mapping = await self._get_mapping(tenant_id, kb_id)
            if existing_mapping:
                logger.info(f"Found existing mapping after creation failure for tenant={tenant_id}, kb={kb_id}")
                table_name = existing_mapping.table_name
            else:
                raise

        try:
            await cache_manager.get_cache().set(cache_key, table_name)
        except Exception as e:
            logger.warning(f"Cache set operation failed for {cache_key}: {e}")
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
        Note: This method does not commit the transaction. The caller is responsible for committing.

        Args:
            tenant_id: The tenant ID
            kb_id: The knowledgebase ID
            table_name: The vector table name

        Returns:
            The created VectorTableMappingEntity
        """
        # First check if mapping already exists (race condition protection)
        existing_mapping = await self._get_mapping(tenant_id, kb_id)
        if existing_mapping:
            logger.debug(f"Mapping already exists for tenant={tenant_id}, kb={kb_id}, returning existing")
            return existing_mapping

        mapping = VectorTableMappingEntity(
            tenant_id=tenant_id,
            kb_id=kb_id,
            table_name=table_name,
        )
        try:
            self.session.add(mapping)
            try:
                await self.session.flush()
                await self.session.refresh(mapping)
                logger.info(f"Created vector table mapping: tenant={tenant_id}, kb={kb_id}, table={table_name}")
                return mapping
            except Exception as flush_ex:
                # Check if it's a "Session is already flushing" error
                error_str = str(flush_ex).lower()
                if "already flushing" in error_str or "invalidrequesterror" in error_str:
                    logger.warning(f"Session flush conflict detected, trying to get existing mapping for tenant={tenant_id}, kb={kb_id}")
                    # Remove the failed mapping from session to avoid conflicts
                    try:
                        self.session.expunge(mapping)
                    except Exception:
                        pass
                    # Try to get existing mapping (another concurrent task may have created it)
                    existing_mapping = await self._get_mapping(tenant_id, kb_id)
                    if existing_mapping:
                        logger.info(f"Found existing vector table mapping after flush conflict for tenant={tenant_id}, kb={kb_id}")
                        return existing_mapping
                    # If no existing mapping, log and re-raise
                    logger.error("No existing mapping found after flush conflict, re-raising error")
                    raise
                raise
        except Exception as ex:
            logger.error(f"Failed to create vector table mapping: {traceback.format_exc()}")
            # Check if it's a unique constraint violation
            error_msg = str(ex)
            if hasattr(ex, 'orig'):
                error_msg = str(ex.orig)

            if "UniqueViolationError" in error_msg or "Duplicate entry" in error_msg or "UNIQUE constraint" in error_msg:
                # If duplicate, try to get the existing mapping
                existing_mapping = await self._get_mapping(tenant_id, kb_id)
                if existing_mapping:
                    logger.info(f"Found existing vector table mapping after unique constraint violation for tenant={tenant_id}, kb={kb_id}")
                    return existing_mapping
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
            try:
                await cache_manager.get_cache().delete(cache_key)
            except Exception as e:
                logger.warning(f"Cache delete operation failed for {cache_key}: {e}")
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
        try:
            table_name = await cache_manager.get_cache().get(cache_key)
            if table_name:
                logger.debug(f"Found vector table name in cache for tenant {tenant_id} and kb {kb_id}: {table_name}")
                return table_name
        except Exception as e:
            logger.warning(f"Cache get operation failed for {cache_key}: {e}, falling back to database")

        mapping = await self._get_mapping(tenant_id, kb_id)
        if mapping:
            try:
                await cache_manager.get_cache().set(cache_key, mapping.table_name)
            except Exception as e:
                logger.warning(f"Cache set operation failed for {cache_key}: {e}")
            return mapping.table_name
        return None
