"""Verbatim document text storage (KnowledgeDocumentContentRow).

`import_text_document` persists the exact ingested `content` once, and
`fetch_document_or_chunk(mode="full_doc")` returns it byte-faithfully instead of
re-stitching overlapping chunks. Old documents (no content row) fall back to the
chunk stitch. All offline (local hash embedder, in-memory SQLite).
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import create_all, make_engine
from app.knowledge import KnowledgeService
from app.models import KnowledgeDocumentContentRow
from app.store.base import User

ADMIN = User(id="u_admin", email="a@x.io", role="admin")

# Long enough (and repetitive enough near boundaries) to span several overlapping
# chunks, so chunk-stitching would visibly duplicate the overlap regions.
ORIGINAL = (
    "Section one talks about the ingestion pipeline in careful detail. "
    "The tokenizer raises ERR_x7abc on overflow. "
) * 40


async def _svc():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    return KnowledgeService(engine, router=None)


def test_full_doc_read_is_byte_faithful():
    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        doc, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="d", content=ORIGINAL, uri="d/1"
        )
        # max_chars large enough to hold the whole thing.
        res = await svc.fetch_document_or_chunk(
            user=ADMIN, document_id=doc.id, mode="full_doc", max_chars=50000
        )
        return doc, res

    doc, res = asyncio.run(scenario())
    # Faithful: exactly the original (outer whitespace trimmed, since the stored
    # copy is content.strip() so its offsets align with chunk char_start), no
    # duplicated overlap, no injected "\n\n".
    assert res["text"] == ORIGINAL.strip()
    assert "\n\n" not in res["text"]


def test_content_row_is_stored_once_and_upserted_on_reimport():
    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        doc, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="d", content=ORIGINAL, uri="d/1"
        )
        # Re-import the SAME uri with new content → reuse doc, replace content row.
        updated = ORIGINAL + " APPENDED_TAIL_marker"
        doc2, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="d", content=updated, uri="d/1"
        )
        async with AsyncSession(svc._engine) as s:
            rows = (
                await s.exec(
                    select(KnowledgeDocumentContentRow).where(
                        KnowledgeDocumentContentRow.document_id == doc.id
                    )
                )
            ).all()
        return doc, doc2, rows, updated

    doc, doc2, rows, updated = asyncio.run(scenario())
    assert doc2.id == doc.id  # same doc reused
    assert len(rows) == 1  # upsert replaced, not appended
    assert rows[0].text == updated
    assert rows[0].char_len == len(updated)


def test_falls_back_to_chunk_stitch_when_no_content_row():
    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        doc, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="d", content=ORIGINAL, uri="d/1"
        )
        # Simulate a document ingested before content storage existed: drop its row.
        async with AsyncSession(svc._engine) as s:
            row = await s.get(KnowledgeDocumentContentRow, doc.id)
            await s.delete(row)
            await s.commit()
        res = await svc.fetch_document_or_chunk(
            user=ADMIN, document_id=doc.id, mode="full_doc", max_chars=50000
        )
        return res

    res = asyncio.run(scenario())
    # Fallback still returns readable text with the seeded content present…
    assert "ERR_x7abc" in res["text"]
    # …but it is the stitched form (chunks joined with blank lines), not verbatim.
    assert "\n\n" in res["text"]
