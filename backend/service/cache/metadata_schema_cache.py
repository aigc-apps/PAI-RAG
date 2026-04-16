"""Cache-aside layer for KB metadata schema + sample values.

On first access (or after cache invalidation), queries the DB for metadata
definitions and sample values, writes to Redis, and returns. Subsequent
reads hit Redis until TTL expires or the cache is explicitly cleared.
"""

import json
from typing import List, Optional

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.db_context import create_db_session
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.knowledgebase import KbEntity
from db.models.knowledgebase.metadata import KbMetadataEntity
from service.cache.redis_cache import cache_manager, kb_metadata_schema_key

# TTL for each KB's cached schema (seconds)
_SCHEMA_TTL = 600  # 10 minutes


class MetadataSchemaCache:
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_schema(self, tenant_id: str, kb_id: str) -> Optional[List[dict]]:
        """Read metadata schema for a KB (cache-aside).

        1. Try Redis cache first.
        2. On cache miss, query DB → build schema → write back to Redis.
        3. Return the schema list, or None if the KB has no metadata.

        Gracefully degrades to None on any cache/DB failure so that the
        caller (knowledgebase tool) can still work without metadata filtering.
        """
        # 1. Try cache
        try:
            cache = cache_manager.get_cache()
            key = kb_metadata_schema_key(tenant_id, kb_id)
            raw = await cache.get(key)
        except Exception:
            logger.opt(exception=True).warning(
                f"Redis unavailable when reading metadata schema for KB {kb_id}, skipping."
            )
            raw = None

        if raw is not None:
            try:
                schema = json.loads(raw) if isinstance(raw, str) else raw
                return schema if schema else None
            except (json.JSONDecodeError, TypeError):
                pass  # treat as cache miss

        # 2. Cache miss — read from DB
        try:
            async with create_db_session() as session:
                schema = await self._build_schema(session, tenant_id, kb_id)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to build metadata schema for KB {kb_id}, returning None."
            )
            return None

        if not schema:
            return None

        # 3. Write back to cache (best-effort)
        try:
            value = json.dumps(schema, ensure_ascii=False)
            await cache.set(key, value, ttl=_SCHEMA_TTL)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to write metadata schema cache for KB {kb_id}, continuing without cache."
            )

        return schema

    async def clear_cache_by_tenant(self, tenant_id: str) -> int:
        """Clear all metadata schema cache entries for a tenant.

        Next call to get_schema() will re-read from DB automatically.

        Returns:
            Number of KB cache entries cleared.
        """
        cache = cache_manager.get_cache()
        async with create_db_session() as session:
            result = await session.exec(
                select(KbEntity.id).where(KbEntity.tenant_id == tenant_id)
            )
            kb_ids = result.all()

        for kb_id in kb_ids:
            await cache.delete(kb_metadata_schema_key(tenant_id, kb_id))

        logger.info(
            f"MetadataSchemaCache cleared {len(kb_ids)} entries for tenant {tenant_id}."
        )
        return len(kb_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _build_schema(
        self, session: AsyncSession, tenant_id: str, kb_id: str
    ) -> List[dict]:
        """Query metadata definitions + sample values for one KB."""
        result = await session.exec(
            select(KbMetadataEntity).where(
                KbMetadataEntity.kb_id == kb_id,
                KbMetadataEntity.tenant_id == tenant_id,
            )
        )
        metadata_entities = list(result.all())
        if not metadata_entities:
            return []

        schema = []
        for meta in metadata_entities:
            sample_values = await self._sample_values(
                session, tenant_id, kb_id, meta.name
            )
            schema.append(
                {
                    "name": meta.name,
                    "value_type": meta.value_type,
                    "description": meta.description or "",
                    "sample_values": sample_values,
                }
            )

        return schema

    async def _sample_values(
        self,
        session: AsyncSession,
        tenant_id: str,
        kb_id: str,
        metadata_name: str,
        limit: int = 5,
    ) -> List[str]:
        """Get up to `limit` distinct non-empty values for a metadata field."""
        stmt = (
            select(
                KbFileEntity.file_metadata[metadata_name].as_string()
            )
            .where(
                KbFileEntity.kb_id == kb_id,
                KbFileEntity.tenant_id == tenant_id,
                KbFileEntity.file_metadata[metadata_name].as_string().isnot(None),
                KbFileEntity.file_metadata[metadata_name].as_string() != "",
            )
            .distinct()
            .limit(limit)
        )
        result = await session.exec(stmt)
        return [v for v in result.all() if v]


# Singleton
metadata_schema_cache = MetadataSchemaCache()
