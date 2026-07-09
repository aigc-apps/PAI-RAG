"""End-to-end proof that a PK collision on a KB write is recovered by the
`with_id_retry` wrapper (and that exhausting the retries surfaces the error).

We monkeypatch ``app.knowledge._uuid`` to force the astronomically-rare event
that the Base58 entropy otherwise makes unobservable: the generator hands back an
id that already exists. The DB unique index rejects it; the wrapper regenerates
and the write succeeds with a distinct id.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import app.knowledge as knowledge_mod
from app.db import create_all, make_engine
from app.knowledge import KnowledgeService
from app.store.base import User, new_id

ADMIN = User(id="u_admin", email="a@x.io", role="admin")


async def _svc():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    return KnowledgeService(engine, router=None)


def _collide_once(monkeypatch, prefix: str, dup_id: str):
    """Patch ``_uuid`` so the first call for ``prefix`` returns ``dup_id`` (a
    guaranteed collision), and every other call delegates to the real generator."""
    fired = {"done": False}

    def fake(p: str) -> str:
        if p == prefix and not fired["done"]:
            fired["done"] = True
            return dup_id
        return new_id(p)

    monkeypatch.setattr(knowledge_mod, "_uuid", fake)
    return fired


def test_create_kb_retries_past_a_kb_id_collision(monkeypatch):
    async def scenario():
        svc = await _svc()
        first = await svc.create_kb(user=ADMIN, name="A", visibility="public")
        # Next create_kb's first "kb" id collides with `first`; retry must recover.
        _collide_once(monkeypatch, "kb", first.id)
        second = await svc.create_kb(user=ADMIN, name="B", visibility="public")
        return first, second

    first, second = asyncio.run(scenario())
    assert second.id != first.id
    assert second.name == "B"


def test_import_text_document_retries_past_a_chunk_id_collision(monkeypatch):
    # Seed a doc, force the next import's first "chk" id to duplicate one of the
    # seeded doc's chunk ids, and assert the retry regenerates it away.
    async def full():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        doc0, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="d0", content="seed body one", uri="d/0"
        )
        # Grab an existing chunk id straight from the store.
        from sqlmodel import select
        from sqlmodel.ext.asyncio.session import AsyncSession
        from app.models import KnowledgeChunkRow

        async with AsyncSession(svc._engine) as s:
            existing_chk = (
                await s.exec(select(KnowledgeChunkRow.id).where(KnowledgeChunkRow.document_id == doc0.id))
            ).first()

        _collide_once(monkeypatch, "chk", existing_chk)
        doc1, job1 = await svc.import_text_document(
            kb.id, user=ADMIN, title="d1", content="another distinct body here", uri="d/1"
        )
        # The new doc's chunks must all have distinct, non-colliding ids.
        async with AsyncSession(svc._engine) as s:
            new_ids = (
                await s.exec(select(KnowledgeChunkRow.id).where(KnowledgeChunkRow.document_id == doc1.id))
            ).all()
        return existing_chk, doc1, list(new_ids)

    existing_chk, doc1, new_ids = asyncio.run(full())
    assert doc1.chunk_count >= 1
    assert existing_chk not in new_ids  # collision was regenerated away
    assert len(new_ids) == len(set(new_ids))


def test_import_text_document_raises_when_collisions_never_clear(monkeypatch):
    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        doc0, _ = await svc.import_text_document(
            kb.id, user=ADMIN, title="d0", content="seed body one", uri="d/0"
        )
        from sqlmodel import select
        from sqlmodel.ext.asyncio.session import AsyncSession
        from app.models import KnowledgeChunkRow

        async with AsyncSession(svc._engine) as s:
            existing_chk = (
                await s.exec(select(KnowledgeChunkRow.id).where(KnowledgeChunkRow.document_id == doc0.id))
            ).first()

        # Always return the same duplicate chunk id → every attempt collides.
        def always_dup(p: str) -> str:
            return existing_chk if p == "chk" else new_id(p)

        monkeypatch.setattr(knowledge_mod, "_uuid", always_dup)

        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            await svc.import_text_document(
                kb.id, user=ADMIN, title="d1", content="body that will fail", uri="d/1"
            )

    asyncio.run(scenario())
