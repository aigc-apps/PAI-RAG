"""Knowledge layer services.

Provides CRUD and business logic for the knowledge architecture layers:
- WikiPageService: Layer 0 (human wiki) management
- CompiledPageService: Layer 2 (agent KB) management
- SuggestionService: Suggestions queue management
- KnowledgeCompilerService: Knowledge compilation orchestrator
- KnowledgeQueryService: Layer 3 query routing
"""

from service.knowledge.wiki_page_service import WikiPageService
from service.knowledge.compiled_page_service import CompiledPageService
from service.knowledge.suggestion_service import SuggestionService
from service.knowledge.knowledge_compiler_service import KnowledgeCompilerService
from service.knowledge.knowledge_event_handler import KnowledgeEventHandler
from service.knowledge.knowledge_query_service import KnowledgeQueryService

__all__ = [
    "WikiPageService",
    "CompiledPageService",
    "SuggestionService",
    "KnowledgeCompilerService",
    "KnowledgeEventHandler",
    "KnowledgeQueryService",
]
