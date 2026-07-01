"""Wiki Page Service — Layer 0: Human Wiki management.

This service provides CRUD operations for human-authored wiki pages.
The agent uses this service in READ-ONLY mode. Only human-facing APIs
should invoke write operations.

Ownership: HUMANS ONLY (exclusive write access via this service).
"""

from datetime import datetime, timezone
from typing import Optional, List

from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.knowledge.wiki_page import WikiPageEntity, WikiPageCreate, WikiPageUpdate
from common.system_constants import DEFAULT_TENANT_ID


class WikiPageService:
    """Service layer for Wiki Page (Layer 0) CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ─── READ Operations (used by both humans and agent) ───────────────

    async def get_page(self, page_id: str, tenant_id: str = DEFAULT_TENANT_ID) -> Optional[WikiPageEntity]:
        """Get a wiki page by ID."""
        result = await self.session.exec(
            select(WikiPageEntity).where(
                WikiPageEntity.id == page_id,
                WikiPageEntity.tenant_id == tenant_id,
                WikiPageEntity.archived.is_(False),
            )
        )
        return result.first()

    async def get_page_by_title(self, title: str, tenant_id: str = DEFAULT_TENANT_ID) -> Optional[WikiPageEntity]:
        """Get a wiki page by title (unique per tenant)."""
        result = await self.session.exec(
            select(WikiPageEntity).where(
                WikiPageEntity.title == title,
                WikiPageEntity.tenant_id == tenant_id,
                WikiPageEntity.archived.is_(False),
            )
        )
        return result.first()

    async def list_pages(
        self,
        tenant_id: str = DEFAULT_TENANT_ID,
        category: Optional[str] = None,
        page: int = 1,
        size: int = 20,
    ) -> dict:
        """List wiki pages with optional category filter and pagination."""
        conditions = [
            WikiPageEntity.tenant_id == tenant_id,
            WikiPageEntity.archived.is_(False),
        ]
        if category:
            conditions.append(WikiPageEntity.category == category)

        where = and_(*conditions)

        # Count
        count_stmt = select(func.count(WikiPageEntity.id)).where(where)
        total_result = await self.session.exec(count_stmt)
        total = total_result.one_or_none() or 0

        # Paginate
        offset = (page - 1) * size
        stmt = (
            select(WikiPageEntity)
            .where(where)
            .order_by(WikiPageEntity.updated_at.desc())
            .offset(offset)
            .limit(size)
        )
        result = await self.session.exec(stmt)
        items = list(result.all())

        return {"items": items, "total": total, "page": page, "size": size}

    async def list_all_pages(self, tenant_id: str = DEFAULT_TENANT_ID) -> List[WikiPageEntity]:
        """List all active wiki pages (used by compiler for full scans)."""
        result = await self.session.exec(
            select(WikiPageEntity).where(
                WikiPageEntity.tenant_id == tenant_id,
                WikiPageEntity.archived.is_(False),
            )
        )
        return list(result.all())

    # ─── WRITE Operations (human-only, never called by agent) ──────────

    async def create_page(
        self, data: WikiPageCreate, edited_by: str = "system", tenant_id: str = DEFAULT_TENANT_ID
    ) -> WikiPageEntity:
        """Create a new wiki page.

        Raises:
            ValueError: If title already exists for this tenant.
        """
        try:
            entity = WikiPageEntity(
                tenant_id=tenant_id,
                title=data.title,
                category=data.category,
                content=data.content,
                tags=data.tags,
                source_doc_ids=data.source_doc_ids,
                last_edited_by=edited_by,
                version=1,
            )
            self.session.add(entity)
            await self.session.flush()
            await self.session.refresh(entity)
            logger.info(f"Created wiki page: {entity.id} (title: {entity.title})")
            return entity
        except IntegrityError as e:
            logger.error(f"IntegrityError creating wiki page: {e.orig}")
            raise ValueError(f"Wiki page title '{data.title}' already exists.") from e

    async def update_page(
        self, page_id: str, data: WikiPageUpdate, edited_by: str = "system", tenant_id: str = DEFAULT_TENANT_ID
    ) -> WikiPageEntity:
        """Update an existing wiki page. Increments version.

        This triggers the staleness flow: any compiled pages referencing
        this wiki page will be marked stale.

        Raises:
            ValueError: If page not found.
        """
        page = await self.get_page(page_id, tenant_id)
        if not page:
            raise ValueError(f"Wiki page '{page_id}' not found.")

        # Apply updates
        if data.title is not None:
            page.title = data.title
        if data.category is not None:
            page.category = data.category
        if data.content is not None:
            page.content = data.content
        if data.tags is not None:
            page.tags = data.tags
        if data.source_doc_ids is not None:
            page.source_doc_ids = data.source_doc_ids

        # Increment version and track editor
        page.version += 1
        page.last_edited_by = edited_by
        page.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        self.session.add(page)
        await self.session.flush()
        await self.session.refresh(page)

        logger.info(f"Updated wiki page: {page.id} (v{page.version}, by {edited_by})")
        return page

    async def archive_page(self, page_id: str, tenant_id: str = DEFAULT_TENANT_ID) -> WikiPageEntity:
        """Soft-delete a wiki page (set archived=True).

        Raises:
            ValueError: If page not found.
        """
        page = await self.get_page(page_id, tenant_id)
        if not page:
            raise ValueError(f"Wiki page '{page_id}' not found.")

        page.archived = True
        page.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        self.session.add(page)
        await self.session.flush()

        logger.info(f"Archived wiki page: {page.id}")
        return page
