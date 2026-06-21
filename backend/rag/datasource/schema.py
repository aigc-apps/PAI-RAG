"""Source-agnostic data structures exchanged between adapters and the sync worker.

`DiscoveredDoc` is the lightweight per-document record produced by an adapter's
``discover()`` (available *before* fetching the body). `SourceDocument` is the
fully normalized document produced by ``emit()`` (after ``fetch()``), carrying
the unified metadata schema + body that the sync worker ingests.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiscoveredDoc:
    """One document as listed by an adapter's discover() stage."""

    path: str  # relative path within the data source, e.g. "zh/pai/billing-of-eas.md"
    title: str
    source_url: Optional[str] = None  # human-readable page URL (citation)
    fetch_url: Optional[str] = None  # the actual resource to fetch (e.g. the .md)
    section: Optional[str] = None
    summary: Optional[str] = None
    product: Optional[str] = None
    lang: Optional[str] = None
    # adapter-specific extras (manifest-only, not exposed to the agent)
    source_meta: dict = field(default_factory=dict)


@dataclass
class SourceDocument:
    """A fully normalized document, ready for ingestion (unified metadata + body)."""

    doc_id: str  # "{datasource_key}/{path}", stable across re-fetches
    datasource_key: str
    path: str
    title: str
    content: str  # normalized markdown body
    content_hash: str  # sha256 of the (frontmatter-stripped) body
    byte_size: int
    fetched_from: str  # extraction method, e.g. "aliyun-llms.txt"
    fetched_at: str  # ISO date
    source_url: Optional[str] = None
    fetch_url: Optional[str] = None
    source_site: Optional[str] = None
    section: Optional[str] = None
    product: Optional[str] = None
    summary: Optional[str] = None
    lang: Optional[str] = None
    source_meta: dict = field(default_factory=dict)
