"""The KB embedder/reranker default now lives in the model catalog
(default_embedding_model / default_rerank_model); the cosmetic embedding.default
/ rerank.default provider slots are gone. These tests pin that removal and that
apply_runtime_status still grades cleanly without them."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent_config import DEFAULT_DOCUMENT, apply_runtime_status, mask_secrets
from app.providers import ModelCatalog, ModelSpec, ProviderConfig, ProviderRouter


def _settings():
    return type("S", (), {
        "openai_api_key": "", "default_model": "dashscope/chat",
        "search_provider": "none", "search_api_key": "", "search_endpoint": "",
    })()


def _catalog():
    return ModelCatalog(
        default_model="dashscope/chat",
        default_embedding_model="dashscope/text-embedding-v4",
        default_rerank_model="dashscope/qwen3-rerank",
        providers=[
            ProviderConfig(name="dashscope", base_url="https://ds/compatible-mode/v1", api_key="k", models=[
                ModelSpec(id="chat"),
                ModelSpec(id="text-embedding-v4", type="embedding", dimension=1024, base_url="https://ds/emb"),
                ModelSpec(id="qwen3-rerank", type="rerank", base_url="https://ds/rr"),
            ]),
        ],
    )


def test_default_document_has_no_embedding_rerank_slots():
    ids = {p.id for p in DEFAULT_DOCUMENT.providers}
    assert "embedding.default" not in ids
    assert "rerank.default" not in ids
    # The vector engine is now a global typed section, not a provider slot.
    assert "vectordb.default" not in ids


def test_vectordb_lives_in_knowledgebase_section():
    vectordb = DEFAULT_DOCUMENT.knowledgebase.vectordb
    assert vectordb.engine == "local"


def test_knowledge_capability_refs_cleared():
    knowledge = next(c for c in DEFAULT_DOCUMENT.capabilities if c.id == "knowledge")
    assert knowledge.provider_refs == []


def test_apply_runtime_status_clean_without_slots():
    # No embedding.default/rerank.default to reconcile — and in local mode the
    # knowledge capability grades from `enabled`, not provider_refs.
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    out = apply_runtime_status(doc, _settings(), ProviderRouter(_catalog()))
    ids = {p.id for p in out.providers}
    assert "embedding.default" not in ids and "rerank.default" not in ids
    knowledge = next(c for c in out.capabilities if c.id == "knowledge")
    assert knowledge.status == "ready"


def test_apply_runtime_status_clean_without_router():
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    out = apply_runtime_status(doc, _settings(), None)
    knowledge = next(c for c in out.capabilities if c.id == "knowledge")
    assert knowledge.status == "ready"


# --------------------------------------------------------------------------- #
# knowledgebase.vectordb grading + masking
# --------------------------------------------------------------------------- #
def test_vectordb_local_graded_healthy():
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    out = apply_runtime_status(doc, _settings(), None)
    assert out.knowledgebase.vectordb.status == "healthy"
    assert out.knowledgebase.vectordb.error is None


def test_vectordb_elasticsearch_missing_url_or_secret():
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    doc.knowledgebase.vectordb.engine = "elasticsearch"  # no url/secret
    out = apply_runtime_status(doc, _settings(), None)
    assert out.knowledgebase.vectordb.status == "missing_config"
    assert out.knowledgebase.vectordb.error


def test_vectordb_elasticsearch_with_url_and_api_key_healthy():
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    vdb = doc.knowledgebase.vectordb
    vdb.engine, vdb.url, vdb.api_key = "elasticsearch", "http://es:9200", "k"
    out = apply_runtime_status(doc, _settings(), None)
    assert out.knowledgebase.vectordb.status == "healthy"
    assert out.knowledgebase.vectordb.secret_configured is True


def test_vectordb_elasticsearch_basic_auth_needs_username():
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    vdb = doc.knowledgebase.vectordb
    vdb.engine, vdb.url, vdb.password = "elasticsearch", "http://es:9200", "pw"
    # password without username → the ES engine won't use basic_auth → missing.
    out = apply_runtime_status(doc, _settings(), None)
    assert out.knowledgebase.vectordb.status == "missing_config"
    vdb.username = "elastic"
    out2 = apply_runtime_status(doc, _settings(), None)
    assert out2.knowledgebase.vectordb.status == "healthy"


def test_mask_secrets_masks_vectordb_credentials():
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    vdb = doc.knowledgebase.vectordb
    vdb.api_key, vdb.password = "real-key", "real-pw"
    masked = mask_secrets(doc)
    assert masked.knowledgebase.vectordb.api_key == "********"
    assert masked.knowledgebase.vectordb.password == "********"
    # original untouched
    assert doc.knowledgebase.vectordb.api_key == "real-key"
