# ruff: noqa: E402
"""Knowledge inspection agent tools (and their KnowledgeService seams).

- knowledge_read reads a whole document by its short id, paginates, and fails softly;
- knowledge_find finds an EXACT literal substring — including one buried inside a larger
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

from loguru import logger

from agent.tools.builtin.knowledge_read import make_knowledge_read_tool
from agent.tools.builtin.knowledge_find import make_knowledge_find_tool
from agent.tools.builtin.knowledge_list import make_knowledge_list_tool
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
# knowledge_read
# --------------------------------------------------------------------------- #
def test_knowledge_read_returns_full_document():
    svc, kb, doc = _seed()
    view = make_knowledge_read_tool(svc)
    assert view.name == "knowledge_read"
    out = _run(ADMIN, lambda: view.fn(document_id=doc.id))
    assert "Runbook" in out
    assert f"document_id: {doc.id}" in out
    assert "ERR_x7abc" in out  # body is present in full


def test_knowledge_read_paginates_with_offset_and_max_chars():
    svc, kb, doc = _seed()
    view = make_knowledge_read_tool(svc)
    page1 = _run(ADMIN, lambda: view.fn(document_id=doc.id, max_chars=200, offset=0))
    assert "truncated" in page1  # more to read → paging hint present
    page2 = _run(ADMIN, lambda: view.fn(document_id=doc.id, max_chars=200, offset=200))
    # The two windows are different slices of the document.
    assert page1 != page2


def test_knowledge_read_bad_id_is_friendly():
    svc, kb, doc = _seed()
    view = make_knowledge_read_tool(svc)
    out = _run(ADMIN, lambda: view.fn(document_id="doc_doesNotExist"))
    assert "knowledge_read" in out and "Traceback" not in out


def test_knowledge_read_not_found_redacts_exception_message():
    sentinel = "SENSITIVE_DOCUMENT_CONTENT_401d"

    class MissingDocumentService:
        async def fetch_document_or_chunk(self, **kwargs):
            raise LookupError(f"document contains {sentinel}")

    view = make_knowledge_read_tool(MissingDocumentService())
    out = _run(ADMIN, lambda: view.fn(document_id="doc_missing"))
    assert out == "knowledge_read: document or chunk was not found."
    assert sentinel not in out


def test_knowledge_read_failure_redacts_content_from_logs_and_output():
    sentinel = "SENSITIVE_DOCUMENT_CONTENT_53ba"

    class FailingReadService:
        async def fetch_document_or_chunk(self, **kwargs):
            raise RuntimeError(f"parser exposed {sentinel}")

    view = make_knowledge_read_tool(FailingReadService())
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        out = _run(ADMIN, lambda: view.fn(document_id="doc_failure"))
    finally:
        logger.remove(sink)
    assert out == "knowledge_read failed due to an internal error."
    assert sentinel not in out
    assert all(sentinel not in message for message in messages)
    assert any(
        "operation=knowledge_read" in message and "error_type=RuntimeError" in message
        for message in messages
    )


def test_knowledge_read_requires_an_id():
    svc, kb, doc = _seed()
    view = make_knowledge_read_tool(svc)
    out = _run(ADMIN, lambda: view.fn())
    assert "requires" in out


def test_knowledge_read_denies_other_users_private_doc():
    svc, kb, doc = _seed()
    view = make_knowledge_read_tool(svc)
    out = _run(OTHER, lambda: view.fn(document_id=doc.id))
    assert "do not have access" in out
    assert "ERR_x7abc" not in out  # nothing leaked


# --------------------------------------------------------------------------- #
# knowledge_find
# --------------------------------------------------------------------------- #
def test_knowledge_find_finds_exact_term():
    svc, kb, doc = _seed()
    grep = make_knowledge_find_tool(svc)
    assert grep.name == "knowledge_find"
    out = _run(ADMIN, lambda: grep.fn(query="ERR_x7abc"))
    assert doc.id in out
    assert "ERR_x7abc" in out


def test_knowledge_find_finds_substring_inside_a_token():
    # "x7ab" is a substring of the token "ERR_x7abc" — impossible for a tokenizer
    # to match, trivial for a literal substring grep.
    svc, kb, doc = _seed()
    grep = make_knowledge_find_tool(svc)
    out = _run(ADMIN, lambda: grep.fn(query="x7ab"))
    assert doc.id in out


def test_knowledge_find_no_match_is_friendly():
    svc, kb, doc = _seed()
    grep = make_knowledge_find_tool(svc)
    out = _run(ADMIN, lambda: grep.fn(query="ZZ_not_present_anywhere"))
    assert "No chunk contains" in out


def test_knowledge_find_scoped_to_document_id():
    svc, kb, doc = _seed()
    grep = make_knowledge_find_tool(svc)
    out = _run(ADMIN, lambda: grep.fn(query="tokenizer", document_id=doc.id))
    assert doc.id in out


def test_knowledge_find_hides_other_users_private_kb():
    svc, kb, doc = _seed()
    grep = make_knowledge_find_tool(svc)
    # OTHER has no accessible KBs → nothing to grep, even for the seeded term.
    out = _run(OTHER, lambda: grep.fn(query="ERR_x7abc"))
    assert "No chunk contains" in out
    # Even naming the KB explicitly must not leak (get_kb rejects → skipped).
    out2 = _run(OTHER, lambda: grep.fn(query="ERR_x7abc", kb_ids=[kb.id]))
    assert "No chunk contains" in out2


def test_knowledge_find_failure_redacts_query_from_logs_and_output():
    sentinel = "SENSITIVE_FIND_QUERY_ba72"

    class FailingFindService:
        async def grep_chunks(self, **kwargs):
            raise RuntimeError(f"database rejected {sentinel}")

    grep = make_knowledge_find_tool(FailingFindService())
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        out = _run(ADMIN, lambda: grep.fn(query=sentinel))
    finally:
        logger.remove(sink)
    assert out == "knowledge_find failed due to an internal error."
    assert sentinel not in out
    assert all(sentinel not in message for message in messages)
    assert any(
        "operation=knowledge_find" in message and "error_type=RuntimeError" in message
        for message in messages
    )


# --------------------------------------------------------------------------- #
# knowledge_list
# --------------------------------------------------------------------------- #
def test_knowledge_list_shows_accessible_bases_with_ids():
    svc, kb, doc = _seed()
    tool = make_knowledge_list_tool(svc)
    assert tool.name == "knowledge_list"
    out = _run(ADMIN, lambda: tool.fn())
    assert kb.id in out  # the id the model needs for kb_ids scoping
    assert "Ops KB" in out
    assert "1 docs" in out  # size surfaced so the model can skip empty bases


def test_knowledge_list_hides_other_users_private_base():
    svc, kb, doc = _seed()
    tool = make_knowledge_list_tool(svc)
    out = _run(OTHER, lambda: tool.fn())
    # OTHER can't see ADMIN's private base → the friendly "none" message, no leak.
    assert kb.id not in out
    assert "No knowledge bases" in out


def test_knowledge_list_failure_redacts_content_from_logs_and_output():
    sentinel = "SENSITIVE_KB_DESCRIPTION_cd19"

    class FailingListService:
        async def list_kbs(self, *, user):
            raise RuntimeError(f"bad knowledge base description {sentinel}")

    tool = make_knowledge_list_tool(FailingListService())
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        out = _run(ADMIN, lambda: tool.fn())
    finally:
        logger.remove(sink)
    assert out == "knowledge_list failed due to an internal error."
    assert sentinel not in out
    assert all(sentinel not in message for message in messages)
    assert any(
        "operation=knowledge_list" in message and "error_type=RuntimeError" in message
        for message in messages
    )


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
