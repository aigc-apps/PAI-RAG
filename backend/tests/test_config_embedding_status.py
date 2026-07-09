"""apply_runtime_status grades embedding.default / rerank.default from the model
they reference in the catalogue (healthy when the model resolves to the right
type and its key is ready; missing_config otherwise)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent_config import DEFAULT_DOCUMENT, apply_runtime_status
from app.providers import ModelCatalog, ModelSpec, ProviderConfig, ProviderRouter


def _settings():
    return type("S", (), {
        "openai_api_key": "", "default_model": "dashscope/chat",
        "search_provider": "none", "search_api_key": "", "search_endpoint": "",
    })()


def _catalog(*, emb_key="k"):
    return ModelCatalog(default_model="dashscope/chat", providers=[
        ProviderConfig(name="dashscope", base_url="https://ds/compatible-mode/v1", api_key=emb_key, models=[
            ModelSpec(id="chat"),
            ModelSpec(id="text-embedding-v4", type="embedding", dimension=1024, base_url="https://ds/emb"),
            ModelSpec(id="qwen3-rerank", type="rerank", base_url="https://ds/rr"),
        ]),
    ])


def test_embedding_and_rerank_healthy_when_model_resolves():
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    out = apply_runtime_status(doc, _settings(), ProviderRouter(_catalog()))
    providers = {p.id: p for p in out.providers}
    emb = providers["embedding.default"]
    rr = providers["rerank.default"]
    assert emb.status == "healthy" and emb.secret_configured is True
    assert emb.settings["provider"] == "dashscope"
    assert emb.settings["base_url"] == "https://ds/emb"
    assert emb.settings["dimension"] == 1024
    assert rr.status == "healthy" and rr.secret_configured is True


def test_embedding_missing_config_without_router():
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    out = apply_runtime_status(doc, _settings(), None)
    providers = {p.id: p for p in out.providers}
    assert providers["embedding.default"].status == "missing_config"
    assert providers["rerank.default"].status == "missing_config"


def test_embedding_missing_config_when_key_absent(monkeypatch):
    monkeypatch.delenv("DS_ABSENT", raising=False)
    cat = ModelCatalog(default_model="dashscope/chat", providers=[
        ProviderConfig(name="dashscope", base_url="u", api_key_env="DS_ABSENT", models=[
            ModelSpec(id="chat", base_url="u2"),  # chat carries its own key path via provider; keyless-less
            ModelSpec(id="text-embedding-v4", type="embedding", dimension=1024),
            ModelSpec(id="qwen3-rerank", type="rerank"),
        ]),
    ])
    # default_model 'chat' would also need the key, but load doesn't validate keys.
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    out = apply_runtime_status(doc, _settings(), ProviderRouter(cat))
    providers = {p.id: p for p in out.providers}
    assert providers["embedding.default"].status == "missing_config"
    assert providers["embedding.default"].secret_configured is False


def test_wrong_type_reference_is_missing_config_with_error():
    # Point embedding.default at a chat model id -> type mismatch -> missing_config.
    doc = DEFAULT_DOCUMENT.model_copy(deep=True)
    for p in doc.providers:
        if p.id == "embedding.default":
            p.settings = {"model": "dashscope/chat"}
    out = apply_runtime_status(doc, _settings(), ProviderRouter(_catalog()))
    emb = {p.id: p for p in out.providers}["embedding.default"]
    assert emb.status == "missing_config"
    assert emb.error and "not a embedding model" in emb.error
