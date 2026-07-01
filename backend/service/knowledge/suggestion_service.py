"""Suggestion Service — Agent → Human feedback queue.

Manages the suggestions queue where the agent proposes edits and humans resolve them.
The agent creates suggestions; humans accept, reject, or defer them.

This is the ONLY upward data flow in the system.
"""

from datetime import datetime, timezone
from typing import Optional, List

from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import and_
from loguru import logger

from db.models.knowledge.suggestion import SuggestionEntity, SuggestionCreate
from common.system_constants import DEFAULT_TENANT_ID


class SuggestionService:
    """Service layer for the Suggestions queue."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ─── READ Operations ──────────────────────────────────────────────

    async def get_suggestion(self, suggestion_id: str, tenant_id: str = DEFAULT_TENANT_ID) -> Optional[SuggestionEntity]:
        """Get a suggestion by ID."""
        result = await self.session.exec(
            select(SuggestionEntity).where(
                SuggestionEntity.id == suggestion_id,
                SuggestionEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    async def list_suggestions(
        self,
        tenant_id: str = DEFAULT_TENANT_ID,
        status: Optional[str] = None,
        target_page_id: Optional[str] = None,
        page: int = 1,
        size: int = 20,
    ) -> dict:
        """List suggestions with filters."""
        conditions = [SuggestionEntity.tenant_id == tenant_id]
        if status:
            conditions.append(SuggestionEntity.status == status)
        if target_page_id:
            conditions.append(SuggestionEntity.target_wiki_page_id == target_page_id)

        where = and_(*conditions)

        count_stmt = select(func.count(SuggestionEntity.id)).where(where)
        total_result = await self.session.exec(count_stmt)
        total = total_result.one_or_none() or 0

        offset = (page - 1) * size
        stmt = (
            select(SuggestionEntity)
            .where(where)
            .order_by(SuggestionEntity.created_at.desc())
            .offset(offset)
            .limit(size)
        )
        result = await self.session.exec(stmt)
        items = list(result.all())

        return {"items": items, "total": total, "page": page, "size": size}

    async def list_pending(self, tenant_id: str = DEFAULT_TENANT_ID, limit: int = 50) -> List[SuggestionEntity]:
        """Get all pending suggestions for review."""
        result = await self.session.exec(
            select(SuggestionEntity)
            .where(
                SuggestionEntity.tenant_id == tenant_id,
                SuggestionEntity.status == "pending",
            )
            .order_by(SuggestionEntity.confidence.desc())
            .limit(limit)
        )
        return list(result.all())

    # ─── WRITE: Agent creates suggestions ─────────────────────────────

    async def create_suggestion(
        self, data: SuggestionCreate, tenant_id: str = DEFAULT_TENANT_ID
    ) -> SuggestionEntity:
        """Agent creates a new suggestion for human review."""
        entity = SuggestionEntity(
            tenant_id=tenant_id,
            target_wiki_page_id=data.target_wiki_page_id,
            issue_type=data.issue_type,
            description=data.description,
            evidence=data.evidence,
            suggested_action=data.suggested_action,
            confidence=data.confidence,
            status="pending",
        )
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        logger.info(
            f"Created suggestion: {entity.id} "
            f"(target: {entity.target_wiki_page_id}, type: {entity.issue_type})"
        )
        return entity

    # ─── WRITE: Human resolves suggestions ────────────────────────────

    async def accept_suggestion(
        self, suggestion_id: str, resolved_by: str, note: Optional[str] = None, tenant_id: str = DEFAULT_TENANT_ID
    ) -> SuggestionEntity:
        """Human accepts a suggestion.

        After acceptance, the human should edit the wiki page accordingly.
        """
        suggestion = await self.get_suggestion(suggestion_id, tenant_id)
        if not suggestion:
            raise ValueError(f"Suggestion '{suggestion_id}' not found.")
        if suggestion.status != "pending":
            raise ValueError(f"Suggestion '{suggestion_id}' is already resolved (status: {suggestion.status}).")

        suggestion.status = "accepted"
        suggestion.resolved_by = resolved_by
        suggestion.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
        suggestion.resolution_note = note

        self.session.add(suggestion)
        await self.session.flush()
        logger.info(f"Suggestion accepted: {suggestion_id} by {resolved_by}")
        return suggestion

    async def reject_suggestion(
        self, suggestion_id: str, resolved_by: str, note: Optional[str] = None, tenant_id: str = DEFAULT_TENANT_ID
    ) -> SuggestionEntity:
        """Human rejects a suggestion (agent cannot override)."""
        suggestion = await self.get_suggestion(suggestion_id, tenant_id)
        if not suggestion:
            raise ValueError(f"Suggestion '{suggestion_id}' not found.")
        if suggestion.status != "pending":
            raise ValueError(f"Suggestion '{suggestion_id}' is already resolved (status: {suggestion.status}).")

        suggestion.status = "rejected"
        suggestion.resolved_by = resolved_by
        suggestion.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
        suggestion.resolution_note = note

        self.session.add(suggestion)
        await self.session.flush()
        logger.info(f"Suggestion rejected: {suggestion_id} by {resolved_by}")
        return suggestion

    async def defer_suggestion(
        self, suggestion_id: str, resolved_by: str, note: Optional[str] = None, tenant_id: str = DEFAULT_TENANT_ID
    ) -> SuggestionEntity:
        """Human defers a suggestion for later review."""
        suggestion = await self.get_suggestion(suggestion_id, tenant_id)
        if not suggestion:
            raise ValueError(f"Suggestion '{suggestion_id}' not found.")
        if suggestion.status != "pending":
            raise ValueError(f"Suggestion '{suggestion_id}' is already resolved (status: {suggestion.status}).")

        suggestion.status = "deferred"
        suggestion.resolved_by = resolved_by
        suggestion.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
        suggestion.resolution_note = note

        self.session.add(suggestion)
        await self.session.flush()
        logger.info(f"Suggestion deferred: {suggestion_id} by {resolved_by}")
        return suggestion
