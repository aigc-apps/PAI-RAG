"""Knowledge Event Handlers — Data Flow Orchestration.

Implements the unidirectional data flow between layers:
  Flow 1: Human edits wiki → mark stale → queue recompilation
  Flow 2: New doc uploaded → mark stale → queue recompilation
  Flow 3: Agent detects contradiction → create suggestion (never modify wiki)

These handlers are called by API endpoints/webhooks when source data changes.
They enforce the ownership rules and ensure proper event propagation.
"""

from typing import Optional

from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

from service.knowledge.knowledge_compiler_service import (
    KnowledgeCompilerService,
    KnowledgeEvent,
)
from common.system_constants import DEFAULT_TENANT_ID


class KnowledgeEventHandler:
    """Handles knowledge layer events and orchestrates data flow.

    This is the integration point between the existing PAI-RAG system
    and the new knowledge layer. External systems (API routes, webhooks,
    background tasks) call these methods to trigger the data flow.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.compiler = KnowledgeCompilerService(session)

    # ─── Flow 1: Wiki Page Changed ────────────────────────────────────

    async def on_wiki_page_created(
        self, page_id: str, title: str, tenant_id: str = DEFAULT_TENANT_ID
    ) -> dict:
        """Called when a human creates a new wiki page.

        Actions:
          - No staleness (new page, nothing depends on it yet)
          - Optionally trigger compilation if page covers a new topic
        """
        logger.info(f"Event: wiki page created — {page_id} ('{title}')")
        event = KnowledgeEvent(
            event_type="wiki_page_created",
            source_type="human_wiki",
            source_id=page_id,
            source_version=1,
            tenant_id=tenant_id,
        )
        return await self.compiler.handle_event(event)

    async def on_wiki_page_updated(
        self, page_id: str, new_version: int, tenant_id: str = DEFAULT_TENANT_ID
    ) -> dict:
        """Called when a human updates a wiki page.

        Actions:
          - Mark all compiled pages that reference this wiki page as STALE
          - Queue those pages for recompilation
        """
        logger.info(f"Event: wiki page updated — {page_id} (v{new_version})")
        event = KnowledgeEvent(
            event_type="wiki_page_updated",
            source_type="human_wiki",
            source_id=page_id,
            source_version=new_version,
            tenant_id=tenant_id,
        )
        return await self.compiler.handle_event(event)

    # ─── Flow 2: Raw Document Changed ─────────────────────────────────

    async def on_raw_doc_added(
        self, doc_id: str, supersedes: Optional[str] = None, tenant_id: str = DEFAULT_TENANT_ID
    ) -> dict:
        """Called when a new raw document is uploaded.

        Actions:
          - If supersedes an existing doc: mark compiled pages citing old doc as stale
          - Queue affected compiled pages for recompilation
        """
        logger.info(f"Event: raw doc added — {doc_id} (supersedes: {supersedes})")

        result = {"doc_id": doc_id, "stale_count": 0}

        # If this doc supersedes an old one, mark old citations stale
        if supersedes:
            event = KnowledgeEvent(
                event_type="raw_doc_superseded",
                source_type="raw_doc",
                source_id=supersedes,
                tenant_id=tenant_id,
                metadata={"new_doc_id": doc_id},
            )
            actions = await self.compiler.handle_event(event)
            result["stale_count"] = actions.get("stale_count", 0)

        # Also fire event for the new doc (allows proactive compilation)
        event = KnowledgeEvent(
            event_type="raw_doc_added",
            source_type="raw_doc",
            source_id=doc_id,
            tenant_id=tenant_id,
        )
        await self.compiler.handle_event(event)

        return result

    # ─── Flow 3: Scheduled Recompilation ──────────────────────────────

    async def run_recompilation_batch(
        self, limit: int = 10, tenant_id: str = DEFAULT_TENANT_ID
    ) -> dict:
        """Run a batch of recompilation jobs.

        Called by a scheduled background task (e.g., every 5 minutes).
        Processes the most stale pages first.
        """
        logger.info(f"Running recompilation batch (limit: {limit})")
        results = await self.compiler.recompile_stale_pages(limit, tenant_id)
        return {
            "processed": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [
                {"page_id": r.page_id, "title": r.title, "success": r.success, "error": r.error}
                for r in results
            ],
        }

    # ─── Flow 4: Staleness Audit ──────────────────────────────────────

    async def run_staleness_audit(self, max_age_days: int = 30, tenant_id: str = DEFAULT_TENANT_ID) -> dict:
        """Audit compiled pages for time-based staleness.

        Pages not revalidated in `max_age_days` are marked stale.
        Called by a scheduled daily task.
        """
        from datetime import datetime, timezone, timedelta
        from sqlmodel import select
        from db.models.knowledge.compiled_page import CompiledPageEntity

        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=max_age_days)

        result = await self.session.exec(
            select(CompiledPageEntity).where(
                CompiledPageEntity.tenant_id == tenant_id,
                CompiledPageEntity.stale.is_(False),
                CompiledPageEntity.compiled_at < cutoff,
            )
        )
        old_pages = list(result.all())

        count = 0
        for page in old_pages:
            page.stale = True
            page.stale_reason = f"Not revalidated in {max_age_days} days"
            page.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.session.add(page)
            count += 1

        if count > 0:
            await self.session.flush()

        logger.info(f"Staleness audit: marked {count} pages stale (older than {max_age_days} days)")
        return {"marked_stale": count, "max_age_days": max_age_days}
