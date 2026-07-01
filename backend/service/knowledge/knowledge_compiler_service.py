"""Knowledge Compiler Agent — The core engine of Layer 2.

This agent is responsible for:
1. Receiving events when source data changes (wiki edits, new docs)
2. Marking dependent compiled pages as stale
3. Recompiling stale pages by reading latest sources
4. Detecting contradictions and creating suggestions

Data Flow:
  - Human Wiki (L0) changes → mark stale → queue recompilation
  - Raw Doc (L1) added/superseded → mark stale → queue recompilation
  - Recompilation → read sources → LLM synthesis → update compiled page
  - Contradiction detected → create Suggestion (never modify wiki directly)

Ownership Rules:
  - This agent WRITES ONLY to Layer 2 (compiled pages) and Suggestions queue.
  - This agent READS from Layer 0 (wiki) and Layer 1 (raw docs).
  - This agent NEVER modifies Layer 0 or Layer 1.
"""

from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass

from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

from service.knowledge.wiki_page_service import WikiPageService
from service.knowledge.compiled_page_service import CompiledPageService
from service.knowledge.suggestion_service import SuggestionService
from db.models.knowledge.compiled_page import CompiledPageCreate
from db.models.knowledge.suggestion import SuggestionCreate
from common.system_constants import DEFAULT_TENANT_ID


@dataclass
class KnowledgeEvent:
    """Event representing a change in source data."""

    event_type: str  # "wiki_page_updated" | "wiki_page_created" | "raw_doc_added" | "raw_doc_superseded"
    source_type: str  # "human_wiki" | "raw_doc"
    source_id: str
    source_version: Optional[int] = None
    tenant_id: str = DEFAULT_TENANT_ID
    metadata: Optional[dict] = None


@dataclass
class CompilationResult:
    """Result of a compilation job."""

    page_id: str
    title: str
    success: bool
    error: Optional[str] = None
    suggestions_created: int = 0


class KnowledgeCompilerService:
    """Knowledge Compiler Agent — orchestrates compilation and staleness.

    This is the brain of the knowledge layer. It:
    1. Processes events from source layers
    2. Manages the staleness lifecycle
    3. Orchestrates recompilation
    4. Detects contradictions and raises suggestions
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.wiki_service = WikiPageService(session)
        self.compiled_service = CompiledPageService(session)
        self.suggestion_service = SuggestionService(session)

    # ─── Event Processing (Flow 1 & 2 from the plan) ──────────────────

    async def handle_event(self, event: KnowledgeEvent) -> dict:
        """Process a knowledge event and trigger appropriate actions.

        This is the main entry point for the data flow.
        Returns a summary of actions taken.
        """
        logger.info(f"Processing knowledge event: {event.event_type} ({event.source_type}:{event.source_id})")

        actions = {"event": event.event_type, "stale_count": 0, "recompile_queued": False}

        if event.event_type in ("wiki_page_updated", "wiki_page_created"):
            actions["stale_count"] = await self._handle_wiki_change(event)
        elif event.event_type in ("raw_doc_added", "raw_doc_superseded"):
            actions["stale_count"] = await self._handle_doc_change(event)
        else:
            logger.warning(f"Unknown event type: {event.event_type}")

        # If pages were marked stale, queue recompilation
        if actions["stale_count"] > 0:
            actions["recompile_queued"] = True

        return actions

    async def _handle_wiki_change(self, event: KnowledgeEvent) -> int:
        """Handle a wiki page update: mark dependent compiled pages stale."""
        reason = f"Wiki page '{event.source_id}' updated to version {event.source_version}"
        count = await self.compiled_service.mark_stale_by_source(
            source_type="human_wiki",
            source_id=event.source_id,
            reason=reason,
            tenant_id=event.tenant_id,
        )
        logger.info(f"Wiki change: marked {count} compiled pages stale")
        return count

    async def _handle_doc_change(self, event: KnowledgeEvent) -> int:
        """Handle a raw document change: mark dependent compiled pages stale."""
        reason = f"Raw doc '{event.source_id}' {event.event_type}"
        count = await self.compiled_service.mark_stale_by_source(
            source_type="raw_doc",
            source_id=event.source_id,
            reason=reason,
            tenant_id=event.tenant_id,
        )
        logger.info(f"Doc change: marked {count} compiled pages stale")
        return count

    # ─── Compilation (Flow 1 & 2 continuation) ────────────────────────

    async def compile_page(
        self,
        title: str,
        wiki_page_ids: List[str],
        raw_doc_ids: List[str],
        compiler_model: str = "",
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> CompilationResult:
        """Compile a new knowledge page from wiki pages and raw docs.

        This is the core compilation logic. In a full implementation,
        this would call an LLM to synthesize content. For now, it
        assembles the provenance and placeholder content.

        Args:
            title: Title for the compiled page
            wiki_page_ids: Wiki page IDs to compile from
            raw_doc_ids: Raw document IDs to compile from
            compiler_model: LLM model used for compilation
            tenant_id: Tenant ID
        """
        logger.info(f"Compiling page: '{title}' from {len(wiki_page_ids)} wiki + {len(raw_doc_ids)} docs")

        # Build provenance
        compiled_from = []
        wiki_contents = []

        for wiki_id in wiki_page_ids:
            wiki_page = await self.wiki_service.get_page(wiki_id, tenant_id)
            if wiki_page:
                compiled_from.append({
                    "source_type": "human_wiki",
                    "source_id": wiki_id,
                    "source_version": wiki_page.version,
                })
                wiki_contents.append(wiki_page.content)

        for doc_id in raw_doc_ids:
            compiled_from.append({
                "source_type": "raw_doc",
                "source_id": doc_id,
                "source_version": None,
            })

        # Check if page already exists — if so, recompile
        existing = await self.compiled_service.get_page_by_title(title, tenant_id)

        if existing:
            # Recompile existing page
            # TODO: Call LLM for actual synthesis
            content = self._synthesize_content(title, wiki_contents, raw_doc_ids)
            await self.compiled_service.update_page_content(
                page_id=existing.id,
                content=content,
                compiled_from=compiled_from,
                compiler_model=compiler_model,
                confidence=0.85,  # TODO: LLM self-assessed
                tenant_id=tenant_id,
            )
            return CompilationResult(page_id=existing.id, title=title, success=True)
        else:
            # Create new compiled page
            content = self._synthesize_content(title, wiki_contents, raw_doc_ids)
            data = CompiledPageCreate(
                title=title,
                page_type="topic_summary",
                content=content,
                compiled_from=compiled_from,
                compiler_model=compiler_model,
                confidence=0.85,
                tags=[],
            )
            entity = await self.compiled_service.create_page(data, tenant_id)
            return CompilationResult(page_id=entity.id, title=title, success=True)

    async def recompile_stale_pages(
        self, limit: int = 10, tenant_id: str = DEFAULT_TENANT_ID
    ) -> List[CompilationResult]:
        """Recompile stale pages (batch operation for scheduled runs).

        Processes the most stale pages first.
        """
        stale_pages = await self.compiled_service.get_stale_pages(tenant_id, limit)
        results = []

        for page in stale_pages:
            try:
                # Extract source IDs from provenance
                wiki_ids = [
                    s["source_id"] for s in (page.compiled_from or [])
                    if s.get("source_type") == "human_wiki"
                ]
                doc_ids = [
                    s["source_id"] for s in (page.compiled_from or [])
                    if s.get("source_type") == "raw_doc"
                ]

                result = await self.compile_page(
                    title=page.title,
                    wiki_page_ids=wiki_ids,
                    raw_doc_ids=doc_ids,
                    compiler_model=page.compiler_model or "recompile",
                    tenant_id=tenant_id,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to recompile page {page.id}: {e}")
                results.append(CompilationResult(
                    page_id=page.id, title=page.title, success=False, error=str(e)
                ))

        logger.info(f"Recompilation batch: {len(results)} pages processed, "
                    f"{sum(1 for r in results if r.success)} successful")
        return results

    # ─── Contradiction Detection (Flow 3 from the plan) ───────────────

    async def detect_contradictions(
        self, wiki_page_id: str, tenant_id: str = DEFAULT_TENANT_ID
    ) -> List[SuggestionCreate]:
        """Detect contradictions between a wiki page and raw docs.

        In a full implementation, this would use an LLM to compare
        wiki content against raw document content and identify conflicts.

        Returns list of suggestions to create.
        """
        wiki_page = await self.wiki_service.get_page(wiki_page_id, tenant_id)
        if not wiki_page:
            return []

        # TODO: Implement LLM-based contradiction detection
        # For now, this is a placeholder that returns empty list
        # Full implementation would:
        # 1. Get all raw docs referenced by this wiki page
        # 2. Compare key claims in wiki vs raw docs
        # 3. Identify contradictions, coverage gaps, quality issues
        # 4. Return structured suggestions

        logger.debug(f"Contradiction detection for wiki page '{wiki_page_id}' — placeholder (no LLM)")
        return []

    async def create_suggestion_from_detection(
        self,
        target_wiki_page_id: str,
        issue_type: str,
        description: str,
        evidence: List[dict],
        suggested_action: str,
        confidence: float,
        tenant_id: str = DEFAULT_TENANT_ID,
    ) -> None:
        """Create a suggestion when the agent detects an issue.

        This is the ONLY way the agent communicates "up" to humans.
        """
        data = SuggestionCreate(
            target_wiki_page_id=target_wiki_page_id,
            issue_type=issue_type,
            description=description,
            evidence=evidence,
            suggested_action=suggested_action,
            confidence=confidence,
        )
        await self.suggestion_service.create_suggestion(data, tenant_id)

    # ─── Internal Helpers ─────────────────────────────────────────────

    def _synthesize_content(self, title: str, wiki_contents: List[str], raw_doc_ids: List[str]) -> str:
        """Placeholder content synthesis.

        In production, this would call an LLM (e.g., qwen3-235b) to:
        1. Read all wiki content and raw doc extracts
        2. Synthesize a comprehensive page with inline citations
        3. Self-assess confidence

        For now, returns a structured placeholder.
        """
        parts = [f"# {title}\n"]
        parts.append(f"*Compiled at: {datetime.now(timezone.utc).isoformat()}Z*\n")

        if wiki_contents:
            parts.append("## From Human Wiki\n")
            for i, content in enumerate(wiki_contents, 1):
                # Truncate for placeholder
                snippet = content[:500] if content else "(empty)"
                parts.append(f"### Source {i}\n{snippet}\n")

        if raw_doc_ids:
            parts.append("## From Raw Documents\n")
            for doc_id in raw_doc_ids:
                parts.append(f"- Referenced: `{doc_id}`\n")

        parts.append("\n---\n*This page was auto-compiled. Content pending LLM synthesis.*")
        return "\n".join(parts)
