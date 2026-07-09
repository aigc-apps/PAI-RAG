"""view_file + grep_file agent tools (and their KnowledgeService seams).

- view_file reads a whole document by its short id, paginates, and fails softly;
- grep_file finds an EXACT literal substring — including one buried inside a larger
  token, which a BM25 tokenizer cannot match — proving the literal-match value;
- both enforce the same private/workspace/public permission scoping as the REST
  path (kb resolved internally still runs through get_kb);
- service seams: fetch_document_or_chunk works with kb_id=None; grep_chunks merges
  across accessible KBs and honors document_id scoping.

All offline — local hash embedder (router=None), in-memory SQLite.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent.tools.builtin.view_file import make_view_file_tool
from agent.tools.builtin.grep_file import make_grep_file_tool
from agent.tools.scope import ToolScope, set_current_tool_scope, reset_current_tool_scope
from app.db import create_all, make_engine
from app.knowledge import KnowledgeService
from app.store.base import User

ADMIN = User(id="u_admin", email="a@x.io", role="admin")
OTHER = User(id="u_other", email="b@x.io", role="user")

# A term with an exact code embedded in a larger token — grep must find the inner
# substring "x7abc", which no word/CJK tokenizer would ever surface.
DOC_BODY = (
    "Overview of the ingestion pipeline and its widgets.\n\n"
    "The tokenizer raises ERR_x7abc when a document exceeds the shard limit. "
    "Operators should retry with a smaller batch. "
    "Unrelated prose about gadgets, sprockets, and flanges follows here so the "
    "document spans multiple chunks when chunked at the default size. " * 4
)


async def _svc():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    return KnowledgeService(engine, router=None)


def _scope_for(user: User):
    return ToolScope(user_id=user.id, metadata={"role": user.role})


def _run(user: User, coro_factory):
    """Run an async tool call under a ToolScope for ``user``."""
    async def scenario():
        token = set_current_tool_scope(_scope_for(user))
        try:
            return await coro_factory()
        finally:
            reset_current_tool_scope(token)

    return asyncio.run(scenario())


def _seed():
    """Create a private KB owned by ADMIN + import DOC_BODY; return (svc, kb, doc)."""
    async def build():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="Ops KB", visibility="private")
        doc, _job = await svc.import_text_document(
            kb.id, user=ADMIN, title="Runbook", content=DOC_BODY, uri="ops/runbook"
        )
        return svc, kb, doc

    return asyncio.run(build())


# --------------------------------------------------------------------------- #
# view_file
# --------------------------------------------------------------------------- #
def test_view_file_returns_full_document():
    svc, kb, doc = _seed()
    view = make_view_file_tool(svc)
    out = _run(ADMIN, lambda: view.fn(document_id=doc.id))
    assert "Runbook" in out
    assert f"document_id: {doc.id}" in out
    assert "ERR_x7abc" in out  # body is present in full


def test_view_file_paginates_with_offset_and_max_chars():
    svc, kb, doc = _seed()
    view = make_view_file_tool(svc)
    page1 = _run(ADMIN, lambda: view.fn(document_id=doc.id, max_chars=200, offset=0))
    assert "truncated" in page1  # more to read → paging hint present
    page2 = _run(ADMIN, lambda: view.fn(document_id=doc.id, max_chars=200, offset=200))
    # The two windows are different slices of the document.
    assert page1 != page2


def test_view_file_bad_id_is_friendly():
    svc, kb, doc = _seed()
    view = make_view_file_tool(svc)
    out = _run(ADMIN, lambda: view.fn(document_id="doc_doesNotExist"))
    assert "view_file" in out and "Traceback" not in out


def test_view_file_requires_an_id():
    svc, kb, doc = _seed()
    view = make_view_file_tool(svc)
    out = _run(ADMIN, lambda: view.fn())
    assert "requires" in out


def test_view_file_denies_other_users_private_doc():
    svc, kb, doc = _seed()
    view = make_view_file_tool(svc)
    out = _run(OTHER, lambda: view.fn(document_id=doc.id))
    assert "do not have access" in out
    assert "ERR_x7abc" not in out  # nothing leaked


# --------------------------------------------------------------------------- #
# grep_file
# --------------------------------------------------------------------------- #
def test_grep_file_finds_exact_term():
    svc, kb, doc = _seed()
    grep = make_grep_file_tool(svc)
    out = _run(ADMIN, lambda: grep.fn(query="ERR_x7abc"))
    assert doc.id in out
    assert "ERR_x7abc" in out


def test_grep_file_finds_substring_inside_a_token():
    # "x7ab" is a substring of the token "ERR_x7abc" — impossible for a tokenizer
    # to match, trivial for a literal substring grep.
    svc, kb, doc = _seed()
    grep = make_grep_file_tool(svc)
    out = _run(ADMIN, lambda: grep.fn(query="x7ab"))
    assert doc.id in out


def test_grep_file_no_match_is_friendly():
    svc, kb, doc = _seed()
    grep = make_grep_file_tool(svc)
    out = _run(ADMIN, lambda: grep.fn(query="ZZ_not_present_anywhere"))
    assert "No chunk contains" in out


def test_grep_file_scoped_to_document_id():
    svc, kb, doc = _seed()
    grep = make_grep_file_tool(svc)
    out = _run(ADMIN, lambda: grep.fn(query="tokenizer", document_id=doc.id))
    assert doc.id in out


def test_grep_file_hides_other_users_private_kb():
    svc, kb, doc = _seed()
    grep = make_grep_file_tool(svc)
    # OTHER has no accessible KBs → nothing to grep, even for the seeded term.
    out = _run(OTHER, lambda: grep.fn(query="ERR_x7abc"))
    assert "No chunk contains" in out
    # Even naming the KB explicitly must not leak (get_kb rejects → skipped).
    out2 = _run(OTHER, lambda: grep.fn(query="ERR_x7abc", kb_ids=[kb.id]))
    assert "No chunk contains" in out2


# --------------------------------------------------------------------------- #
# service seams
# --------------------------------------------------------------------------- #
def test_fetch_document_or_chunk_resolves_kb_when_omitted():
    svc, kb, doc = _seed()

    async def call():
        return await svc.fetch_document_or_chunk(user=ADMIN, document_id=doc.id)

    result = asyncio.run(call())
    assert result["kb_id"] == kb.id
    assert result["document_id"] == doc.id
    assert "ERR_x7abc" in result["text"]


def test_grep_chunks_merges_and_scopes():
    svc, kb, doc = _seed()

    async def call():
        allkb = await svc.grep_chunks(user=ADMIN, query="tokenizer")
        scoped = await svc.grep_chunks(user=ADMIN, query="tokenizer", document_id=doc.id)
        miss = await svc.grep_chunks(user=ADMIN, query="tokenizer", document_id="doc_nope")
        return allkb, scoped, miss

    allkb, scoped, miss = asyncio.run(call())
    assert allkb and all(m["document_id"] == doc.id for m in allkb)
    assert scoped and scoped[0]["chunk_index"] >= 0
    assert miss == []
