"""Heading-aware chunking, contextual embedding, content cap/window, and locate.

- ``split_document`` splits markdown at heading boundaries (each chunk carries its
  ancestor heading_path), windows over-long sections, and keeps char offsets aligned
  with the stripped base text; plain text falls back to a flat window.
- ingestion embeds a title+heading-prefixed representation while STORING the raw
  body in ``chunk.text``;
- over-long documents store only a capped prefix flagged ``truncated``, and reads
  window the stored copy via SQL substr;
- ``view_file`` locate opens the full document at a chunk's position with context.

All offline — local hash embedder (router=None), in-memory SQLite.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.db import create_all, make_engine
from app import knowledge as kmod
from app.knowledge import KnowledgeService, split_document, _embed_input
from app.models import KnowledgeChunkRow, KnowledgeDocumentContentRow
from app.store.base import User
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

ADMIN = User(id="u_admin", email="a@x.io", role="admin")

MARKDOWN = (
    "# Runbook\n"
    "Intro prose before any section.\n\n"
    "## Deployment\n"
    "Deploy steps go here.\n\n"
    "### Rollback\n"
    "To roll back, run the ROLLBACK_x9 command carefully.\n\n"
    "## Monitoring\n"
    "Watch the dashboards.\n"
)


async def _svc():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    return KnowledgeService(engine, router=None)


# --------------------------------------------------------------------------- #
# split_document (pure)
# --------------------------------------------------------------------------- #
def test_split_markdown_tracks_heading_path():
    chunks = split_document(
        MARKDOWN, mime_type="text/markdown", chunk_size=1000, chunk_overlap=100
    )
    paths = [c["heading_path"] for c in chunks]
    assert ["Runbook"] in paths  # preamble under the H1
    assert ["Runbook", "Deployment"] in paths
    assert ["Runbook", "Deployment", "Rollback"] in paths  # nested H3
    assert ["Runbook", "Monitoring"] in paths  # H3 popped, back up to H2
    # The Rollback section owns the ROLLBACK_x9 term.
    roll = next(c for c in chunks if "ROLLBACK_x9" in c["text"])
    assert roll["heading_path"] == ["Runbook", "Deployment", "Rollback"]


def test_split_offsets_align_with_stripped_base():
    base = MARKDOWN.strip()
    chunks = split_document(
        MARKDOWN, mime_type="text/markdown", chunk_size=1000, chunk_overlap=100
    )
    # Every chunk's recorded span reproduces its body from the stored base text.
    for c in chunks:
        assert base[c["char_start"]: c["char_end"]] == c["text"]


def test_split_windows_long_section():
    body = "word " * 400  # ~2000 chars, well over chunk_size below
    md = f"# Big\n{body}\n"
    chunks = split_document(md, mime_type="text/markdown", chunk_size=300, chunk_overlap=50)
    big = [c for c in chunks if c["heading_path"] == ["Big"]]
    assert len(big) > 1  # the long section was windowed into multiple chunks
    assert all(len(c["text"]) <= 300 for c in big)


def test_split_plain_text_has_empty_heading_path():
    chunks = split_document(
        "Just some plain prose with no markdown headings at all.",
        mime_type="text/plain",
        chunk_size=1000,
        chunk_overlap=100,
    )
    assert chunks and all(c["heading_path"] == [] for c in chunks)


def test_embed_input_prefixes_title_and_heading():
    out = _embed_input("Runbook", ["Deployment", "Rollback"], "body text")
    assert out == "Runbook\nDeployment > Rollback\n\nbody text"
    # No title/heading → just the body (no stray separators).
    assert _embed_input("", [], "body text") == "body text"


# --------------------------------------------------------------------------- #
# ingestion: contextual embedding vs raw stored body + heading_path persisted
# --------------------------------------------------------------------------- #
def test_ingest_embeds_context_but_stores_raw_body(monkeypatch):
    recorded: list[str] = []

    class RecordingEmbedder:
        async def embed(self, texts, *, text_type="document"):
            recorded.extend(texts)
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(kmod, "build_embedder", lambda *a, **k: RecordingEmbedder())

    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        doc, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="Runbook", content=MARKDOWN,
            uri="ops/rb", mime_type="text/markdown",
        )
        async with AsyncSession(svc._engine) as s:
            rows = (
                await s.exec(
                    select(KnowledgeChunkRow).where(
                        KnowledgeChunkRow.document_id == doc.id
                    ).order_by(KnowledgeChunkRow.chunk_index)
                )
            ).all()
        return rows

    rows = asyncio.run(scenario())
    # The stored body is raw — no title/heading prefix leaked into chunk.text.
    roll = next(r for r in rows if "ROLLBACK_x9" in r.text)
    assert not roll.text.startswith("Runbook")
    assert roll.heading_path == ["Runbook", "Deployment", "Rollback"]
    # …but the embedder saw the context-enriched representation for that chunk.
    enriched = next(t for t in recorded if "ROLLBACK_x9" in t)
    assert enriched.startswith("Runbook\nRunbook > Deployment > Rollback\n\n")


# --------------------------------------------------------------------------- #
# content cap + truncated flag + windowed read
# --------------------------------------------------------------------------- #
def test_over_long_document_is_capped_and_flagged(monkeypatch):
    monkeypatch.setattr(kmod, "MAX_STORED_CONTENT_CHARS", 500)

    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        big = "abcdefghij " * 200  # ~2200 chars, over the patched 500 cap
        doc, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="big", content=big, uri="d/big"
        )
        async with AsyncSession(svc._engine) as s:
            row = await s.get(KnowledgeDocumentContentRow, doc.id)
        return row

    row = asyncio.run(scenario())
    assert row.truncated is True
    assert row.char_len == 500
    assert len(row.text) == 500


def test_windowed_read_returns_the_requested_slice():
    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        content = "".join(f"[{i:04d}]" for i in range(500))  # 3000 deterministic chars
        doc, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="w", content=content, uri="d/w"
        )
        base = content.strip()
        mid = await svc.fetch_document_or_chunk(
            user=ADMIN, document_id=doc.id, mode="full_doc", max_chars=100, offset=300
        )
        return base, mid

    base, mid = asyncio.run(scenario())
    assert mid["text"] == base[300:400]  # exact SQL-substr window


# --------------------------------------------------------------------------- #
# view_file locate
# --------------------------------------------------------------------------- #
def test_locate_opens_document_at_chunk_position():
    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        doc, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="Runbook", content=MARKDOWN,
            uri="ops/rb", mime_type="text/markdown",
        )
        # Find the chunk owning the Rollback term via grep.
        hits = await svc.grep_chunks(user=ADMIN, query="ROLLBACK_x9")
        chunk_id = hits[0]["chunk_id"]
        located = await svc.fetch_document_or_chunk(
            user=ADMIN, chunk_id=chunk_id, mode="locate", max_chars=50000
        )
        return chunk_id, located

    chunk_id, located = asyncio.run(scenario())
    assert located["chunk_id"] == chunk_id
    # The window carries the hit itself plus preceding document context (the H1).
    assert "ROLLBACK_x9" in located["text"]
    assert "# Runbook" in located["text"]
