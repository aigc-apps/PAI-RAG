"""Knowledge Query Service — Layer 3: Read-only serving layer.

Implements Flow 4 from the plan: Query Execution.
Combines Human Wiki (L0) + Agent KB (L2) to serve answers.

MCP Tools exposed:
  - search(query, filters) → results from Agent KB with citations
  - read_wiki(page_id) → content from Human Wiki directly
  - get_context(query) → smart routing: Human Wiki + Agent KB combined
  - verify_claim(claim) → check claim against raw sources
  - list_topics() → catalog of all available knowledge

Priority Rules:
  1. Human Wiki ALWAYS wins over Agent KB
  2. Raw Documents ALWAYS win over Agent KB
  3. Human Wiki vs Raw Documents → flag as suggestion, human decides
"""

from typing import Optional, List
from dataclasses import dataclass, field

from sqlmodel.ext.asyncio.session import AsyncSession

from service.knowledge.wiki_page_service import WikiPageService
from service.knowledge.compiled_page_service import CompiledPageService
from common.system_constants import DEFAULT_TENANT_ID


@dataclass
class SearchResult:
    """A single search result with provenance."""

    source_layer: str  # "human_wiki" | "agent_kb"
    page_id: str
    title: str
    content_snippet: str
    confidence: float = 1.0
    stale: bool = False
    citations: List[dict] = field(default_factory=list)


@dataclass
class QueryResponse:
    """Response from the knowledge query service."""

    query: str
    results: List[SearchResult]
    priority_source: str  # "human_wiki" | "agent_kb" | "mixed"
    has_conflicts: bool = False
    conflict_details: Optional[str] = None


class KnowledgeQueryService:
    """Layer 3: Query Service — serves knowledge queries by combining layers.

    Routing logic:
      1. Check Human Wiki for exact page match (authoritative)
      2. Search Agent KB (compiled pages) for synthesis
      3. Apply priority rules if conflict detected
      4. Return combined result with full citations
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.wiki_service = WikiPageService(session)
        self.compiled_service = CompiledPageService(session)

    # ─── MCP Tool: search ─────────────────────────────────────────────

    async def search(
        self,
        query: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        page_type: Optional[str] = None,
        include_stale: bool = True,
    ) -> List[SearchResult]:
        """Search the Agent KB (Layer 2) for compiled knowledge.

        This is a text-based search over compiled pages.
        In production, this would use vector similarity + BM25 hybrid search.
        For now, implements basic title/content matching.
        """
        compiled_result = await self.compiled_service.list_pages(
            tenant_id=tenant_id,
            page_type=page_type,
            stale_only=False,
            page=1,
            size=50,
        )

        results = []
        query_lower = query.lower()

        for page in compiled_result["items"]:
            # Basic relevance: title or content contains query terms
            title_match = query_lower in page.title.lower()
            content_match = query_lower in page.content.lower()

            if title_match or content_match:
                if not include_stale and page.stale:
                    continue

                snippet = self._extract_snippet(page.content, query, max_len=300)
                results.append(SearchResult(
                    source_layer="agent_kb",
                    page_id=page.id,
                    title=page.title,
                    content_snippet=snippet,
                    confidence=page.confidence,
                    stale=page.stale,
                    citations=page.compiled_from or [],
                ))

        # Sort by confidence (non-stale first, then by score)
        results.sort(key=lambda r: (not r.stale, r.confidence), reverse=True)
        return results

    # ─── MCP Tool: read_wiki ──────────────────────────────────────────

    async def read_wiki(
        self, page_id: str, tenant_id: str = DEFAULT_TENANT_ID
    ) -> Optional[dict]:
        """Read a Human Wiki page directly (Layer 0).

        Returns the authoritative human-authored content.
        """
        page = await self.wiki_service.get_page(page_id, tenant_id)
        if not page:
            return None

        return {
            "source": "human_wiki",
            "id": page.id,
            "title": page.title,
            "category": page.category,
            "content": page.content,
            "version": page.version,
            "last_edited_by": page.last_edited_by,
            "tags": page.tags,
        }

    # ─── MCP Tool: get_context ────────────────────────────────────────

    async def get_context(
        self, query: str, tenant_id: str = DEFAULT_TENANT_ID
    ) -> QueryResponse:
        """Smart query routing: combine Human Wiki + Agent KB.

        Priority logic:
          - If Human Wiki has a direct answer → prefer it (authoritative)
          - If Agent KB has richer synthesis → use it, cite wiki as source
          - If conflict → show Agent KB answer + flag conflict
        """
        wiki_results = []
        kb_results = []

        # 1. Check Human Wiki for topic match
        all_wiki_pages = await self.wiki_service.list_all_pages(tenant_id)
        query_lower = query.lower()

        for page in all_wiki_pages:
            if query_lower in page.title.lower() or query_lower in page.content.lower():
                wiki_results.append(SearchResult(
                    source_layer="human_wiki",
                    page_id=page.id,
                    title=page.title,
                    content_snippet=self._extract_snippet(page.content, query, max_len=300),
                    confidence=1.0,  # Wiki is authoritative
                    stale=False,
                ))

        # 2. Search Agent KB
        kb_results = await self.search(query, tenant_id)

        # 3. Merge with priority rules
        all_results = []
        priority_source = "mixed"

        if wiki_results and not kb_results:
            all_results = wiki_results
            priority_source = "human_wiki"
        elif kb_results and not wiki_results:
            all_results = kb_results
            priority_source = "agent_kb"
        else:
            # Both have results — wiki wins, but include KB for synthesis
            all_results = wiki_results + kb_results
            priority_source = "human_wiki"

        # 4. Detect conflicts (simplified)
        has_conflicts = False
        conflict_details = None
        if wiki_results and kb_results:
            # In production: LLM would compare claims
            # For now: flag if both have content about same topic
            has_conflicts = False  # Placeholder

        return QueryResponse(
            query=query,
            results=all_results[:10],  # Limit results
            priority_source=priority_source,
            has_conflicts=has_conflicts,
            conflict_details=conflict_details,
        )

    # ─── MCP Tool: list_topics ────────────────────────────────────────

    async def list_topics(self, tenant_id: str = DEFAULT_TENANT_ID) -> dict:
        """List all available knowledge topics across both layers."""
        wiki_pages = await self.wiki_service.list_all_pages(tenant_id)
        compiled_result = await self.compiled_service.list_pages(
            tenant_id=tenant_id, page=1, size=100
        )

        wiki_topics = [
            {"id": p.id, "title": p.title, "category": p.category, "source": "human_wiki"}
            for p in wiki_pages
        ]
        kb_topics = [
            {
                "id": p.id,
                "title": p.title,
                "type": p.page_type,
                "source": "agent_kb",
                "stale": p.stale,
            }
            for p in compiled_result["items"]
        ]

        return {
            "wiki_topics": wiki_topics,
            "kb_topics": kb_topics,
            "total_wiki": len(wiki_topics),
            "total_kb": len(kb_topics),
        }

    # ─── MCP Tool: verify_claim ───────────────────────────────────────

    async def verify_claim(
        self, claim: str, tenant_id: str = DEFAULT_TENANT_ID
    ) -> dict:
        """Verify a claim against known sources.

        In production, this would:
        1. Search raw docs (Layer 1) for supporting evidence
        2. Check Human Wiki for authoritative statements
        3. Check Agent KB for compiled context

        Returns verification result with supporting/contradicting evidence.
        """
        # Search both layers for relevant content
        context = await self.get_context(claim, tenant_id)

        supporting = []
        for result in context.results:
            if result.confidence > 0.5:
                supporting.append({
                    "source": result.source_layer,
                    "page_id": result.page_id,
                    "title": result.title,
                    "snippet": result.content_snippet,
                })

        return {
            "claim": claim,
            "verified": len(supporting) > 0,
            "confidence": max((r.confidence for r in context.results), default=0.0),
            "supporting_sources": supporting,
            "note": "Full verification requires RAG search over raw documents (Layer 1)",
        }

    # ─── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_snippet(content: str, query: str, max_len: int = 300) -> str:
        """Extract a relevant snippet from content around the query match."""
        if not content:
            return ""

        query_lower = query.lower()
        content_lower = content.lower()
        pos = content_lower.find(query_lower)

        if pos == -1:
            # No exact match, return beginning
            return content[:max_len] + ("..." if len(content) > max_len else "")

        # Center snippet around match
        start = max(0, pos - max_len // 2)
        end = min(len(content), pos + max_len // 2)
        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet
