"""Retrieval-engine tests: local pagination, the Elasticsearch DSL (against a
fake async client — no live server), and auto-mode fallback to local."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent_config import VectorDBConfig
from app.db import create_all, make_engine
from app.knowledge import KnowledgeService
from app.search_engine import ElasticsearchEngine, LocalSearchEngine, build_search_engine
from app.store.base import User

ADMIN = User(id="u_admin", email="a@x.io", role="admin")


# --------------------------------------------------------------------------- #
# Local engine — offset / limit / total
# --------------------------------------------------------------------------- #
async def _seed_many():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    svc = KnowledgeService(engine)
    kb = await svc.create_kb(user=ADMIN, name="Docs", visibility="public")
    for i in range(5):
        await svc.import_text_document(
            kb.id, user=ADMIN, title=f"doc {i}",
            content=f"共同关键词 alpha 段落 {i} 独有词{i}", uri=f"docs/{i}",
        )
    return svc, kb


def test_local_engine_pagination_offset_limit_total():
    async def scenario():
        svc, kb = await _seed_many()
        # every doc shares "alpha" so all 5 match
        page1, total1 = await svc.search(user=ADMIN, kb_ids=[kb.id], query="alpha", top_k=2, offset=0, mode="keyword")
        page2, total2 = await svc.search(user=ADMIN, kb_ids=[kb.id], query="alpha", top_k=2, offset=2, mode="keyword")
        page3, total3 = await svc.search(user=ADMIN, kb_ids=[kb.id], query="alpha", top_k=2, offset=4, mode="keyword")
        return (page1, total1), (page2, total2), (page3, total3)

    (p1, t1), (p2, t2), (p3, t3) = asyncio.run(scenario())
    assert t1 == t2 == t3 == 5
    assert len(p1) == 2 and len(p2) == 2 and len(p3) == 1
    # pages are disjoint (stable ordering by score)
    ids = {h.chunk_id for h in p1} | {h.chunk_id for h in p2} | {h.chunk_id for h in p3}
    assert len(ids) == 5


# --------------------------------------------------------------------------- #
# Elasticsearch engine — DSL shape against a fake client
# --------------------------------------------------------------------------- #
class _FakeIndices:
    def __init__(self, parent):
        self._p = parent

    async def exists(self, index):
        return index in self._p.created

    async def create(self, index, mappings=None, settings=None):
        self._p.created.add(index)
        self._p.create_calls.append({"index": index, "mappings": mappings, "settings": settings})

    async def delete(self, index, ignore_unavailable=False):
        self._p.deleted.append(index)
        self._p.created.discard(index)


class FakeES:
    def __init__(self):
        self.created: set[str] = set()
        self.create_calls: list[dict] = []
        self.deleted: list[str] = []
        self.bulk_calls: list[dict] = []
        self.delete_by_query_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.indices = _FakeIndices(self)
        self._search_response = {"hits": {"total": {"value": 0}, "hits": []}}

    def set_search_response(self, resp):
        self._search_response = resp

    async def bulk(self, operations, refresh=None):
        self.bulk_calls.append({"operations": operations, "refresh": refresh})
        return {"errors": False, "items": []}

    async def delete_by_query(self, index, query, **kw):
        self.delete_by_query_calls.append({"index": index, "query": query})
        return {"deleted": 0}

    async def search(self, index, **body):
        self.search_calls.append({"index": index, "body": body})
        return self._search_response

    async def ping(self):
        return True


class _FakeKB:
    id = "kb_test"
    embedding_config = {"dimension": 64}


class _FakeDoc:
    id = "doc_1"
    title = "安装指南"
    uri = "docs/install"
    source_type = "text"
    category = "guide"
    tags = ["pai"]


def _es_engine(fake):
    return ElasticsearchEngine("http://es:9200", index_prefix="kb", client_factory=lambda: fake)


def test_es_ensure_index_builds_dense_vector_and_cjk_mapping():
    fake = FakeES()
    eng = _es_engine(fake)
    asyncio.run(eng.ensure_index(_FakeKB()))
    assert fake.create_calls, "index should be created"
    m = fake.create_calls[0]["mappings"]["properties"]
    assert m["embedding"]["type"] == "dense_vector"
    assert m["embedding"]["dims"] == 64
    assert m["embedding"]["similarity"] == "cosine"
    assert m["text"]["analyzer"] == "kb_text"
    settings = fake.create_calls[0]["settings"]
    assert settings["analysis"]["analyzer"]["kb_text"]["type"] == "cjk"


def test_es_index_chunks_bulk_replaces_document():
    fake = FakeES()
    eng = _es_engine(fake)
    chunks = [
        {"chunk_id": "c1", "chunk_index": 0, "text": "第一段", "embedding": [0.1] * 64},
        {"chunk_id": "c2", "chunk_index": 1, "text": "第二段", "embedding": [0.2] * 64},
    ]
    asyncio.run(eng.index_chunks(_FakeKB(), _FakeDoc(), chunks))
    # old doc chunks removed before re-indexing
    assert fake.delete_by_query_calls
    assert fake.delete_by_query_calls[0]["query"] == {"term": {"document_id": "doc_1"}}
    ops = fake.bulk_calls[0]["operations"]
    # action/source pairs → 2 chunks = 4 entries
    assert len(ops) == 4
    assert ops[0]["index"]["_id"] == "c1"
    assert ops[1]["kb_id"] == "kb_test"
    assert ops[1]["title"] == "安装指南"
    assert ops[1]["status"] == "active"
    assert len(ops[1]["embedding"]) == 64


def test_es_hybrid_search_dsl_has_knn_and_bm25():
    fake = FakeES()
    fake.set_search_response({
        "hits": {
            "total": {"value": 7},
            "hits": [
                {"_id": "c1", "_score": 3.2, "_source": {
                    "kb_id": "kb_test", "document_id": "doc_1", "chunk_id": "c1",
                    "title": "安装指南", "source_uri": "docs/install", "source_type": "text",
                    "text": "安装步骤", "chunk_index": 0, "tags": ["pai"], "category": "guide",
                }},
            ],
        }
    })
    eng = _es_engine(fake)
    hits, total = asyncio.run(eng.search(
        kb_ids=["kb_test"], query="如何安装", mode="hybrid", offset=0, limit=5,
        filters={"source_type": "text", "tags": ["pai"]},
    ))
    body = fake.search_calls[0]["body"]
    # BM25 clause
    must = body["query"]["bool"]["must"]
    assert must[0]["multi_match"]["fields"] == ["text", "title^2", "heading^1.5"]
    # kNN clause present in the SAME body (native hybrid)
    assert body["knn"]["field"] == "embedding"
    assert len(body["knn"]["query_vector"]) == 64
    # filters pushed in-query (kb_id + status + source_type + tags)
    filt = body["query"]["bool"]["filter"]
    assert {"terms": {"kb_id": ["kb_test"]}} in filt
    assert {"term": {"status": "active"}} in filt
    assert {"term": {"source_type": "text"}} in filt
    assert {"terms": {"tags": ["pai"]}} in filt
    assert body["from"] == 0 and body["size"] == 5
    # result mapping + total
    assert total == 7
    assert hits[0].chunk_id == "c1"
    assert hits[0].title == "安装指南"
    assert hits[0].score == 3.2


def test_es_keyword_mode_omits_knn_and_vector_mode_omits_bm25():
    fake = FakeES()
    eng = _es_engine(fake)
    asyncio.run(eng.search(kb_ids=["kb_test"], query="安装", mode="keyword", limit=3))
    kw_body = fake.search_calls[-1]["body"]
    assert "knn" not in kw_body
    assert "must" in kw_body["query"]["bool"]

    asyncio.run(eng.search(kb_ids=["kb_test"], query="安装", mode="vector", limit=3))
    vec_body = fake.search_calls[-1]["body"]
    assert "knn" in vec_body
    assert "must" not in vec_body["query"]["bool"]


# --------------------------------------------------------------------------- #
# auto mode: ES transport error → local fallback
# --------------------------------------------------------------------------- #
class _BoomEngine:
    name = "elasticsearch"

    async def ensure_index(self, kb): ...
    async def index_chunks(self, kb, doc, chunks): raise RuntimeError("es down")
    async def delete_document(self, kb_id, document_id): ...
    async def delete_kb(self, kb_id): ...
    async def healthy(self): return False

    async def search(self, **kw):
        raise RuntimeError("es unreachable")


def test_auto_mode_falls_back_to_local_on_es_error():
    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        # primary ES engine is broken; fallback enabled (auto)
        svc = KnowledgeService(engine, search_engine=_BoomEngine(), fallback_to_local=True)
        kb = await svc.create_kb(user=ADMIN, name="Docs", visibility="public")
        # import must succeed even though the ES index hook raises (best-effort)
        await svc.import_text_document(kb.id, user=ADMIN, title="安装", content="安装 PAI 需要配置环境变量。", uri="docs/x")
        hits, total = await svc.search(user=ADMIN, kb_ids=[kb.id], query="安装 PAI", mode="keyword")
        return hits, total

    hits, total = asyncio.run(scenario())
    assert total >= 1
    assert any("安装" in h.text for h in hits)


def test_forced_es_mode_raises_without_fallback():
    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine, search_engine=_BoomEngine(), fallback_to_local=False)
        kb = await svc.create_kb(user=ADMIN, name="Docs", visibility="public")
        try:
            await svc.search(user=ADMIN, kb_ids=[kb.id], query="安装")
        except RuntimeError as ex:
            return str(ex)
        return None

    err = asyncio.run(scenario())
    assert err == "es unreachable"


# --------------------------------------------------------------------------- #
# build_search_engine: driven by the global knowledgebase.vectordb section
# --------------------------------------------------------------------------- #
def test_build_search_engine_from_vectordb_elasticsearch():
    cfg = VectorDBConfig(engine="elasticsearch", url="http://es:9200", api_key="k")
    eng = build_search_engine(None, "sql_engine_sentinel", vectordb=cfg)
    assert isinstance(eng, ElasticsearchEngine)
    assert eng._url == "http://es:9200"
    assert eng._api_key == "k"


def test_build_search_engine_from_vectordb_local():
    cfg = VectorDBConfig(engine="local")
    eng = build_search_engine(None, "sql_engine_sentinel", vectordb=cfg)
    assert isinstance(eng, LocalSearchEngine)


def test_build_search_engine_elasticsearch_without_url_falls_back_local():
    cfg = VectorDBConfig(engine="elasticsearch", url="")  # misconfigured
    eng = build_search_engine(None, "sql_engine_sentinel", vectordb=cfg)
    assert isinstance(eng, LocalSearchEngine)


def test_build_search_engine_vectordb_none_uses_env_settings():
    # Legacy path: no section supplied → read Settings.search_engine / elasticsearch_url.
    settings = type("S", (), {"search_engine": "auto", "elasticsearch_url": ""})()
    eng = build_search_engine(settings, "sql_engine_sentinel", vectordb=None)
    assert isinstance(eng, LocalSearchEngine)


def test_from_vectordb_config_resolves_password_from_env():
    os.environ["_TEST_ES_PW"] = "sekret"
    try:
        cfg = VectorDBConfig(
            engine="elasticsearch", url="http://es:9200",
            username="elastic", password_env="_TEST_ES_PW",
        )
        eng = ElasticsearchEngine.from_vectordb_config(cfg)
        assert eng._username == "elastic"
        assert eng._password == "sekret"
    finally:
        os.environ.pop("_TEST_ES_PW", None)


def test_rebuild_search_engine_swaps_primary():
    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine)  # starts local
        assert svc._search.name == "local"
        svc.rebuild_search_engine(
            VectorDBConfig(engine="elasticsearch", url="http://es:9200", api_key="k")
        )
        return svc._search.name, svc._fallback_to_local

    name, fallback = asyncio.run(scenario())
    assert name == "elasticsearch"
    assert fallback is True  # ES degrades to local on outage
