"""Knowledge layer data models.

Implements the 4-layer knowledge architecture:
- Layer 0: Human Wiki (source of truth, human-editable)
- Layer 1: Raw Documents (immutable sources)  [reuses existing KbFileEntity]
- Layer 2: Agent KB (compiled knowledge, agent-owned)
- Suggestions Queue (agent -> human feedback channel)
"""

from db.models.knowledge.wiki_page import WikiPageEntity, WikiPageCreate, WikiPageUpdate
from db.models.knowledge.compiled_page import CompiledPageEntity, CompiledPageCreate
from db.models.knowledge.suggestion import SuggestionEntity, SuggestionCreate

__all__ = [
    "WikiPageEntity",
    "WikiPageCreate",
    "WikiPageUpdate",
    "CompiledPageEntity",
    "CompiledPageCreate",
    "SuggestionEntity",
    "SuggestionCreate",
]
