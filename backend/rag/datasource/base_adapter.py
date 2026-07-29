"""BaseAdapter: the only coupling point between a source type and the unified protocol.

A concrete adapter implements two abstract stages — ``discover()`` (list current
documents) and ``fetch(doc)`` (return one body as markdown). The third stage,
``emit()``, is source-agnostic and implemented here: it normalizes the body,
computes ``content_hash``/``doc_id`` and assembles a :class:`SourceDocument`.

Adapters are pure: config in, documents out. They do no DB or vector-store work.
"""

import re
import hashlib
from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional
from urllib.parse import urlparse, urljoin

from rag.datasource.schema import DiscoveredDoc, SourceDocument

# markdown inline link / image: [text](target) and ![alt](target)
_MD_LINK_RE = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")
# absolute / non-rewritable targets
_ABSOLUTE_RE = re.compile(r"^(https?:|//|#|mailto:|tel:|data:)", re.IGNORECASE)
# leading YAML frontmatter block
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
# first markdown ATX heading (# .. ######)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)


def first_markdown_heading(body: str) -> Optional[str]:
    """Return the text of the first markdown heading, or None."""
    m = _HEADING_RE.search(body or "")
    return m.group(1).strip() if m else None


class BaseAdapter(ABC):
    """Abstract data source adapter (discover / fetch / emit)."""

    #: short identifier matching DataSourceType, set by subclasses
    source_type: str = ""
    #: extraction method recorded in metadata (e.g. "aliyun-llms.txt")
    fetched_from: str = "unknown"

    def __init__(self, datasource_key: str, source_config: Optional[dict] = None):
        self.datasource_key = datasource_key
        self.source_config = source_config or {}
        # Set True by discover() when the listing is known to be incomplete (e.g. a
        # crawl page failed transiently). The sync worker then SKIPS deletions so a
        # transient fetch error can't be mistaken for source-side removal.
        self.discovery_partial = False

    # -- abstract stages ----------------------------------------------------
    @abstractmethod
    def discover(self) -> List[DiscoveredDoc]:
        """List every document currently in the source (no body fetch)."""

    @abstractmethod
    def fetch(self, doc: DiscoveredDoc) -> str:
        """Return the raw markdown body for a discovered document."""

    # -- source-agnostic emit ----------------------------------------------
    def emit(self, doc: DiscoveredDoc, body: str) -> SourceDocument:
        """Normalize a fetched body into a :class:`SourceDocument`."""
        normalized = self.normalize_body(body, base_url=doc.source_url or doc.fetch_url)
        encoded = normalized.encode("utf-8")
        content_hash = hashlib.sha256(encoded).hexdigest()
        source_site = None
        if doc.source_url:
            source_site = urlparse(doc.source_url).netloc or None

        # Always end up with a human-readable title: adapter-provided title,
        # else the body's first markdown heading, else the file basename.
        title = doc.title or first_markdown_heading(normalized) or doc.path.rsplit("/", 1)[-1]

        return SourceDocument(
            source_id=self.get_source_id(doc),
            datasource_key=self.datasource_key,
            path=doc.path,
            title=title,
            content=normalized,
            content_hash=content_hash,
            byte_size=len(encoded),
            fetched_from=self.fetched_from,
            fetched_at=date.today().isoformat(),
            source_url=doc.source_url,
            fetch_url=doc.fetch_url,
            source_site=source_site,
            section=doc.section,
            product=doc.product,
            summary=doc.summary,
            lang=doc.lang,
            source_meta=doc.source_meta or {},
        )

    # -- helpers ------------------------------------------------------------
    def get_source_id(self, doc: DiscoveredDoc) -> str:
        """Return the stable upstream identity used by the sync manifest."""
        source_id = (doc.source_id or doc.path.lstrip("/")).strip()
        if not source_id:
            raise ValueError(f"Document '{doc.path}' has no stable source_id.")
        return source_id

    @staticmethod
    def strip_frontmatter(body: str) -> str:
        """Drop a leading YAML frontmatter block if present."""
        return _FRONTMATTER_RE.sub("", body, count=1)

    @classmethod
    def normalize_body(cls, body: str, base_url: Optional[str] = None) -> str:
        """Normalize a body to markdown: strip frontmatter, rewrite relative links."""
        body = cls.strip_frontmatter(body)
        if base_url:
            body = cls.rewrite_relative_links(body, base_url)
        return body.strip() + "\n"

    @staticmethod
    def rewrite_relative_links(body: str, base_url: str) -> str:
        """Best-effort rewrite of relative markdown link/image targets to absolute URLs."""

        def _sub(m: re.Match) -> str:
            prefix, target, suffix = m.group(1), m.group(2), m.group(3)
            if _ABSOLUTE_RE.match(target):
                return m.group(0)
            try:
                absolute = urljoin(base_url, target)
            except Exception:
                return m.group(0)
            return f"{prefix}{absolute}{suffix}"

        return _MD_LINK_RE.sub(_sub, body)
