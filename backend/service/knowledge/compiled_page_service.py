"""Compiled Page Service — Layer 2: Agent Knowledge Store management.

This service provides CRUD operations for agent-compiled knowledge pages.
Only the Knowledge Compiler Agent should invoke write operations.
Humans and the query service use READ operations.

Ownership: KNOWLEDGE COMPILER AGENT (exclusive write access).
"""

from datetime import datetime, timezone
from typing import Optional, List

from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import and_
from loguru import logger

from db.models.knowledge.compiled_page import CompiledPageEntity, CompiledPageCreate
from common.system_constants import DEFAULT_TENANT_ID


class CompiledPageService:
    """Service layer for Compiled Page (Layer 2) operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ─── READ Operations ──────────────────────────────────────────────

    async def get_page(self, page_id: str, tenant_id: str = DEFAULT_TENANT_ID) -> Optional[CompiledPageEntity]:
        """Get a compiled page by ID."""
        result = await self.session.exec(
            select(CompiledPageEntity).where(
                CompiledPageEntity.id == page_id,
                CompiledPageEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    async def get_page_by_title(self, title: str, tenant_id: str = DEFAULT_TENANT_ID) -> Optional[CompiledPageEntity]:
        """Get a compiled page by title."""
        result = await self.session.exec(
            select(CompiledPageEntity).where(
                CompiledPageEntity.title == title,
                CompiledPageEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    async def list_pages(
        self,
        tenant_id: str = DEFAULT_TENANT_ID,
        page_type: Optional[str] = None,
        stale_only: bool = False,
        page: int = 1,
        size: int = 20,
    ) -> dict:
        """List compiled pages with filters."""
        conditions = [CompiledPageEntity.tenant_id == tenant_id]
        if page_type:
            conditions.append(CompiledPageEntity.page_type == page_type)
        if stale_only:
            conditions.append(CompiledPageEntity.stale.is_(True))

        where = and_(*conditions)

        count_stmt = select(func.count(CompiledPageEntity.id)).where(where)
        total_result = await self.session.exec(count_stmt)
        total = total_result.one_or_none() or 0

        offset = (page - 1) * size
        stmt = (
            select(CompiledPageEntity)
            .where(where)
            .order_by(CompiledPageEntity.compiled_at.desc())
            .offset(offset)
            .limit(size)
        )
        result = await self.session.exec(stmt)
        items = list(result.all())

        return {"items": items, "total": total, "page": page, "size": size}

    async def find_pages_by_source(
        self, source_type: str, source_id: str, tenant_id: str = DEFAULT_TENANT_ID
    ) -> List[CompiledPageEntity]:
        """Find all compiled pages that reference a specific source in their provenance.

        This is critical for the staleness flow: when a wiki page or raw doc
        is updated, we need to find all compiled pages that depend on it.
        """
        # JSON query: find pages where compiled_from contains entries matching source
        # Using SQL JSON contains — works with both PostgreSQL (jsonb @>) and MySQL (JSON_CONTAINS)
        result = await self.session.exec(
            select(CompiledPageEntity).where(
                CompiledPageEntity.tenant_id == tenant_id,
            )
        )
        all_pages = list(result.all())

        # Filter in Python for cross-DB compatibility
        matching = []
        for p in all_pages:
            for src in (p.compiled_from or []):
                if src.get("source_type") == source_type and src.get("source_id") == source_id:
                    matching.append(p)
                    break
        return matching

    async def get_stale_pages(self, tenant_id: str = DEFAULT_TENANT_ID, limit: int = 50) -> List[CompiledPageEntity]:
        """Get stale pages ordered by creation date (for recompilation queue)."""
        result = await self.session.exec(
            select(CompiledPageEntity)
            .where(
                CompiledPageEntity.tenant_id == tenant_id,
                CompiledPageEntity.stale.is_(True),
            )
            .order_by(CompiledPageEntity.compiled_at.asc())
            .limit(limit)
        )
        return list(result.all())

    # ─── WRITE Operations (agent-only) ────────────────────────────────

    async def create_page(
        self, data: CompiledPageCreate, tenant_id: str = DEFAULT_TENANT_ID
    ) -> CompiledPageEntity:
        """Create a new compiled page (agent action)."""
        entity = CompiledPageEntity(
            tenant_id=tenant_id,
            title=data.title,
            page_type=data.page_type,
            content=data.content,
            compiled_from=data.compiled_from,
            compiler_model=data.compiler_model,
            confidence=data.confidence,
            tags=data.tags,
            compiled_at=datetime.now(timezone.utc).replace(tzinfo=None),
            stale=False,
            stale_reason=None,
        )
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        logger.info(f"Created compiled page: {entity.id} (title: {entity.title})")
        return entity

    async def update_page_content(
        self,
        page_id: str,
        content: str,
        compiled_from: List[dict],
        compiler_model: str,
        confidence: float,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> CompiledPageEntity:
        """Recompile an existing page with new content (agent action).

        This replaces content and resets staleness.
        """
        page = await self.get_page(page_id, tenant_id)
        if not page:
            raise ValueError(f"Compiled page '{page_id}' not found.")

        page.content = content
        page.compiled_from = compiled_from
        page.compiler_model = compiler_model
        page.confidence = confidence
        page.compiled_at = datetime.now(timezone.utc).replace(tzinfo=None)
        page.stale = False
        page.stale_reason = None
        page.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        self.session.add(page)
        await self.session.flush()
        await self.session.refresh(page)
        logger.info(f"Recompiled page: {page.id} (title: {page.title})")
        return page

    async def mark_stale(
        self, page_id: str, reason: str, tenant_id: str = DEFAULT_TENANT_ID
    ) -> Optional[CompiledPageEntity]:
        """Mark a compiled page as stale (triggered by source updates)."""
        page = await self.get_page(page_id, tenant_id)
        if not page:
            return None

        page.stale = True
        page.stale_reason = reason
        page.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        self.session.add(page)
        await self.session.flush()
        logger.info(f"Marked page stale: {page.id} (reason: {reason})")
        return page

    async def mark_stale_by_source(
        self, source_type: str, source_id: str, reason: str, tenant_id: str = DEFAULT_TENANT_ID
    ) -> int:
        """Mark all compiled pages that depend on a given source as stale.

        Returns the number of pages marked stale.
        """
        pages = await self.find_pages_by_source(source_type, source_id, tenant_id)
        count = 0
        for page in pages:
            if not page.stale:
                page.stale = True
                page.stale_reason = reason
                page.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                self.session.add(page)
                count += 1

        if count > 0:
            await self.session.flush()
            logger.info(f"Marked {count} compiled pages stale (source: {source_type}:{source_id})")
        return count

    async def delete_page(self, page_id: str, tenant_id: str = DEFAULT_TENANT_ID) -> None:
        """Delete a compiled page (agent cleanup)."""
        page = await self.get_page(page_id, tenant_id)
        if not page:
            raise ValueError(f"Compiled page '{page_id}' not found.")

        await self.session.delete(page)
        await self.session.flush()
        logger.info(f"Deleted compiled page: {page_id}")
