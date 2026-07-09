import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.providers import (
    ModelConfig, ModelSpec, ProviderConfig, ModelCatalog, load_catalog, ProviderRouter,
)


class _Settings:
    openai_base_url = "https://api.openai.com/v1"
    openai_api_key = "sk-legacy"
    default_model = "openai/gpt-4o-mini"


def _catalog():
    return ModelCatalog(
        default_model="openai/fast",
        providers=[
            ProviderConfig(name="openai", base_url="https://fast/v1", api_key="k1", models=[
                ModelSpec(id="fast", context_window=128000, max_output_tokens=8192, supports_tools=True),
            ]),
            ProviderConfig(name="anthropic", base_url="https://smart/v1", api_key="k2", models=[
                ModelSpec(id="smart", context_window=200000, max_output_tokens=16384, supports_reasoning=True),
            ]),
        ],
    )


def test_get_config_and_unknown_raises():
    r = ProviderRouter(_catalog())
    assert r.get_config("anthropic/smart").base_url == "https://smart/v1"
    assert r.default_model_id == "openai/fast"
    with pytest.raises(KeyError):
        r.get_config("nope/nope")


def test_get_llm_builds_client_with_model_and_reasoning():
    r = ProviderRouter(_catalog())
    fast = r.get_llm("openai/fast")
    smart = r.get_llm("anthropic/smart")
    assert fast.model == "fast" and str(fast.client.base_url).startswith("https://fast")
    assert smart.enable_thinking is True and fast.enable_thinking is False
    # cached
    assert r.get_llm("openai/fast") is fast


def test_register_llm_overrides_client():
    r = ProviderRouter(_catalog())
    sentinel = object()
    r.register_llm("openai/fast", sentinel)
    assert r.get_llm("openai/fast") is sentinel


def test_list_models():
    r = ProviderRouter(_catalog())
    assert {m.qualified_id for m in r.list_models()} == {"openai/fast", "anthropic/smart"}


def test_key_resolution_direct_then_env(monkeypatch):
    monkeypatch.setenv("MYKEY", "from-env")
    cat = ModelCatalog(default_model="x/a", providers=[
        ProviderConfig(name="x", base_url="u", api_key_env="MYKEY", models=[ModelSpec(id="a")]),
    ])
    r = ProviderRouter(cat)
    assert r.get_llm("x/a").client.api_key == "from-env"


def test_provider_with_missing_key_is_kept_and_validated_on_use(monkeypatch):
    # Keys are not checked at load: a provider with an unset api_key_env stays
    # in the catalog (so the user can switch to it after setting the env var,
    # without re-editing the catalog). It only fails when actually used.
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    cat = ModelCatalog(default_model="x/present", providers=[
        ProviderConfig(name="x", base_url="u", api_key="k", models=[ModelSpec(id="present")]),
        ProviderConfig(name="y", base_url="u", api_key_env="ABSENT_KEY", models=[ModelSpec(id="absent")]),
    ])
    r = ProviderRouter(cat)
    assert "y/absent" in {m.qualified_id for m in r.list_models()}
    assert "x/present" in {m.qualified_id for m in r.list_models()}
    # using the configured provider works
    assert r.get_llm("x/present").model == "present"
    # using the unconfigured provider raises a clear, actionable error
    with pytest.raises(RuntimeError, match="requires env var 'ABSENT_KEY'"):
        r.get_llm("y/absent")


def test_keyless_provider_builds_without_error():
    # A provider with api_key_env="" (local/keyless) is used directly, no raise.
    cat = ModelCatalog(default_model="ollama/llama", providers=[
        ProviderConfig(name="ollama", base_url="http://localhost:11434/v1", api_key_env="", models=[
            ModelSpec(id="llama"),
        ]),
    ])
    r = ProviderRouter(cat)
    llm = r.get_llm("ollama/llama")
    assert llm.model == "llama"
    assert str(llm.client.api_key) == "EMPTY"


def test_reload_swaps_configs_and_preserves_unchanged_warm_client():
    r = ProviderRouter(_catalog())
    fast_before = r.get_llm("openai/fast")   # warm
    r.get_llm("anthropic/smart")
    # reload: 'openai/fast' unchanged, 'anthropic/smart' changed (new base_url), 'x/extra' added
    r.reload(ModelCatalog(default_model="openai/fast", providers=[
        ProviderConfig(name="openai", base_url="https://fast/v1", api_key="k1", models=[
            ModelSpec(id="fast", context_window=128000, max_output_tokens=8192, supports_tools=True),
        ]),
        ProviderConfig(name="anthropic", base_url="https://smart2/v1", api_key="k2", models=[
            ModelSpec(id="smart"),
        ]),
        ProviderConfig(name="x", base_url="https://extra/v1", api_key="k3", models=[
            ModelSpec(id="extra"),
        ]),
    ]))
    assert r.get_llm("openai/fast") is fast_before          # unchanged -> warm client preserved
    assert str(r.get_llm("anthropic/smart").client.base_url).startswith("https://smart2")  # changed -> rebuilt
    assert "x/extra" in {m.qualified_id for m in r.list_models()}


def test_duplicate_provider_name_raises():
    cat = ModelCatalog(default_model="openai/a", providers=[
        ProviderConfig(name="openai", base_url="u", api_key="k", models=[ModelSpec(id="a")]),
        ProviderConfig(name="openai", base_url="u", api_key="k", models=[ModelSpec(id="b")]),
    ])
    with pytest.raises(ValueError, match="duplicate provider"):
        ProviderRouter(cat)


def test_duplicate_model_id_within_provider_raises():
    cat = ModelCatalog(default_model="openai/a", providers=[
        ProviderConfig(name="openai", base_url="u", api_key="k", models=[
            ModelSpec(id="a"), ModelSpec(id="a"),
        ]),
    ])
    with pytest.raises(ValueError, match="duplicate model id"):
        ProviderRouter(cat)


def test_two_models_under_one_provider():
    cat = ModelCatalog(default_model="openai/big", providers=[
        ProviderConfig(name="openai", base_url="https://api.openai.com/v1", api_key="k", models=[
            ModelSpec(id="small", max_output_tokens=4096),
            ModelSpec(id="big", max_output_tokens=16384),
        ]),
    ])
    r = ProviderRouter(cat)
    assert {m.qualified_id for m in r.list_models()} == {"openai/small", "openai/big"}
    assert r.get_config("openai/big").base_url == "https://api.openai.com/v1"
    assert r.get_config("openai/small").max_output_tokens == 4096


def test_missing_default_model_raises():
    cat = ModelCatalog(default_model="openai/missing", providers=[
        ProviderConfig(name="openai", base_url="u", api_key="k", models=[
            ModelSpec(id="present"),
        ]),
    ])
    with pytest.raises(ValueError, match="default_model 'openai/missing' is not defined"):
        ProviderRouter(cat)


def test_empty_model_catalog_raises():
    cat = ModelCatalog(default_model="openai/missing", providers=[])
    with pytest.raises(ValueError, match="at least one model"):
        ProviderRouter(cat)


def test_load_catalog_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "models:\n"
        "  default_model: openai/m1\n"
        "  providers:\n"
        "    - name: openai\n"
        "      base_url: https://x/v1\n"
        "      api_key_env: OPENAI_API_KEY\n"
        "      models:\n"
        "        - id: m1\n"
        "          context_window: 8000\n"
    )
    cat = load_catalog(str(p), _Settings())
    assert cat.default_model == "openai/m1"
    assert cat.providers[0].base_url == "https://x/v1"
    assert cat.providers[0].models[0].id == "m1"
    assert cat.providers[0].models[0].context_window == 8000


def test_load_catalog_rejects_missing_models_section(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("default_model: openai/m1\nproviders: []\n")
    with pytest.raises(ValueError, match="must contain a models section"):
        load_catalog(str(p), _Settings())


def test_load_catalog_fallback_uses_default_unified_config():
    cat = load_catalog("/no/such/config.yaml", _Settings())
    assert cat.default_model == "openai/gpt-4o-mini"
    p0 = cat.providers[0]
    assert p0.name == "openai"
    assert p0.base_url == "https://api.openai.com/v1"
    assert p0.models[0].id == "gpt-4o-mini"


# --------------------------------------------------------------------------- #
# Typed models: embedding + rerank in the same provider (Dify/RAGFlow shape)
# --------------------------------------------------------------------------- #
def _typed_catalog(**overrides):
    emb_key = overrides.get("emb_api_key", "k")
    return ModelCatalog(
        default_model="dashscope/chat",
        providers=[
            ProviderConfig(
                name="dashscope",
                base_url="https://ds/compatible-mode/v1",
                api_key=emb_key,
                models=[
                    ModelSpec(id="chat"),
                    ModelSpec(
                        id="text-embedding-v4", type="embedding", protocol="dashscope",
                        dimension=1024, base_url="https://ds/emb",
                    ),
                    ModelSpec(id="qwen3-rerank", type="rerank", protocol="dashscope",
                              base_url="https://ds/rr"),
                ],
            )
        ],
    )


def test_typed_model_config_carries_type_dimension_and_base_url_override():
    r = ProviderRouter(_typed_catalog())
    emb = r.get_config("dashscope/text-embedding-v4")
    assert emb.type == "embedding" and emb.dimension == 1024
    assert emb.base_url == "https://ds/emb"          # model-level override
    chat = r.get_config("dashscope/chat")
    assert chat.type == "chat" and chat.base_url == "https://ds/compatible-mode/v1"


def test_get_embedder_builds_caches_and_type_checks():
    r = ProviderRouter(_typed_catalog())
    emb = r.get_embedder("dashscope/text-embedding-v4")
    assert emb.model == "text-embedding-v4" and emb.dimension == 1024
    assert emb.base_url == "https://ds/emb" and emb.api_key == "k"
    assert r.get_embedder("dashscope/text-embedding-v4") is emb  # cached
    with pytest.raises(ValueError, match="not an embedding model"):
        r.get_embedder("dashscope/chat")


def test_get_reranker_builds_and_type_checks():
    r = ProviderRouter(_typed_catalog())
    rr = r.get_reranker("dashscope/qwen3-rerank")
    assert rr.model == "qwen3-rerank" and rr.base_url == "https://ds/rr"
    with pytest.raises(ValueError, match="not a rerank model"):
        r.get_reranker("dashscope/chat")


def test_get_embedder_missing_key_raises(monkeypatch):
    monkeypatch.delenv("DS_ABSENT", raising=False)
    cat = ModelCatalog(default_model="dashscope/chat", providers=[
        ProviderConfig(name="dashscope", base_url="u", api_key_env="DS_ABSENT", models=[
            ModelSpec(id="chat"),
            ModelSpec(id="emb", type="embedding", dimension=8),
        ]),
    ])
    r = ProviderRouter(cat)
    with pytest.raises(RuntimeError, match="requires env var 'DS_ABSENT'"):
        r.get_embedder("dashscope/emb")


def test_get_embedder_and_reranker_pick_protocol():
    from app.retrieval_models import (
        DashScopeEmbedder, DashScopeReranker,
        OpenAICompatibleEmbedder, OpenAICompatibleReranker,
    )
    cat = ModelCatalog(default_model="p/chat", providers=[
        ProviderConfig(name="dashscope", base_url="https://ds/compatible-mode/v1", api_key="k", models=[
            ModelSpec(id="emb-native", type="embedding", protocol="dashscope", dimension=1024,
                      base_url="https://ds/native/emb"),
            ModelSpec(id="rr-native", type="rerank", protocol="dashscope", base_url="https://ds/native/rr"),
        ]),
        ProviderConfig(name="vendor", base_url="https://api.vendor.com/v1", api_key="k2", models=[
            ModelSpec(id="emb-compat", type="embedding", dimension=512),   # protocol defaults openai
            ModelSpec(id="rr-compat", type="rerank"),
        ]),
        ProviderConfig(name="p", base_url="u", api_key="k3", models=[ModelSpec(id="chat")]),
    ])
    r = ProviderRouter(cat)
    native_emb = r.get_embedder("dashscope/emb-native")
    compat_emb = r.get_embedder("vendor/emb-compat")
    assert isinstance(native_emb, DashScopeEmbedder)
    assert isinstance(compat_emb, OpenAICompatibleEmbedder)
    assert compat_emb.url == "https://api.vendor.com/v1/embeddings" and compat_emb.dimension == 512
    assert isinstance(r.get_reranker("dashscope/rr-native"), DashScopeReranker)
    compat_rr = r.get_reranker("vendor/rr-compat")
    assert isinstance(compat_rr, OpenAICompatibleReranker)
    assert compat_rr.url == "https://api.vendor.com/v1/rerank"


def test_default_model_id_of_type():
    r = ProviderRouter(_typed_catalog())
    assert r.default_model_id_of_type("embedding") == "dashscope/text-embedding-v4"
    assert r.default_model_id_of_type("rerank") == "dashscope/qwen3-rerank"
    assert r.default_model_id_of_type("chat") == "dashscope/chat"


def test_get_llm_rejects_non_chat_model():
    r = ProviderRouter(_typed_catalog())
    with pytest.raises(ValueError, match="not a chat model"):
        r.get_llm("dashscope/text-embedding-v4")


def test_default_model_must_be_chat_type():
    cat = ModelCatalog(default_model="dashscope/emb", providers=[
        ProviderConfig(name="dashscope", base_url="u", api_key="k", models=[
            ModelSpec(id="emb", type="embedding", dimension=8),
        ]),
    ])
    with pytest.raises(ValueError, match="must be a chat model"):
        ProviderRouter(cat)


# --------------------------------------------------------------------------- #
# base_url resolution in ModelConfig.from_provider
# --------------------------------------------------------------------------- #
_DASH_EMBED = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
_DASH_RERANK = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


def test_dashscope_native_embedding_defaults_base_url():
    # protocol=dashscope embedding with no base_url → the fixed native endpoint,
    # NOT the provider's compatible-mode root.
    p = ProviderConfig(name="dashscope", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="k",
                       models=[ModelSpec(id="text-embedding-v4", type="embedding", protocol="dashscope", dimension=1024)])
    cfg = ModelConfig.from_provider(p, p.models[0])
    assert cfg.base_url == _DASH_EMBED


def test_dashscope_native_rerank_defaults_base_url():
    p = ProviderConfig(name="dashscope", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="k",
                       models=[ModelSpec(id="qwen3-rerank", type="rerank", protocol="dashscope")])
    cfg = ModelConfig.from_provider(p, p.models[0])
    assert cfg.base_url == _DASH_RERANK


def test_explicit_base_url_overrides_dashscope_default():
    # An explicit URL (e.g. a proxied endpoint) always wins over the native default.
    p = ProviderConfig(name="dashscope", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="k",
                       models=[ModelSpec(id="emb", type="embedding", protocol="dashscope", dimension=1024,
                                         base_url="https://proxy.internal/embed")])
    cfg = ModelConfig.from_provider(p, p.models[0])
    assert cfg.base_url == "https://proxy.internal/embed"


def test_openai_compatible_embedding_inherits_provider_base_url():
    # No native default for openai-protocol models — inherit the provider /v1 root
    # (the OpenAI-compatible client appends /embeddings itself).
    p = ProviderConfig(name="siliconflow", base_url="https://api.siliconflow.cn/v1", api_key="k",
                       models=[ModelSpec(id="bge-m3", type="embedding", dimension=1024)])
    cfg = ModelConfig.from_provider(p, p.models[0])
    assert cfg.protocol == "openai"
    assert cfg.base_url == "https://api.siliconflow.cn/v1"


def test_dashscope_chat_has_no_native_default():
    # Chat models have no native embedding/rerank endpoint — they inherit the
    # provider's (openai-compatible) root even under protocol=dashscope.
    p = ProviderConfig(name="dashscope", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="k",
                       models=[ModelSpec(id="qwen3.7-plus", type="chat", protocol="dashscope")])
    cfg = ModelConfig.from_provider(p, p.models[0])
    assert cfg.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ---- Catalog-level embedding/rerank defaults (default_embedding_model / _rerank) ----

def _defaults_catalog(**kw):
    return ModelCatalog(
        default_model="dashscope/chat",
        providers=[
            ProviderConfig(name="dashscope", base_url="https://ds/v1", api_key="k", models=[
                ModelSpec(id="chat"),
                ModelSpec(id="emb-a", type="embedding", dimension=1024),
                ModelSpec(id="emb-b", type="embedding", dimension=512),
                ModelSpec(id="rr-a", type="rerank"),
            ]),
        ],
        **kw,
    )


def test_explicit_default_embedding_and_rerank_win():
    r = ProviderRouter(_defaults_catalog(
        default_embedding_model="dashscope/emb-b",
        default_rerank_model="dashscope/rr-a",
    ))
    assert r.default_model_id_of_type("embedding") == "dashscope/emb-b"
    assert r.default_rerank_model_id == "dashscope/rr-a"
    assert r.default_embedding_model_id == "dashscope/emb-b"


def test_default_of_type_falls_back_to_first_when_unset():
    # No explicit defaults -> first catalogued model of that type (insertion order).
    r = ProviderRouter(_defaults_catalog())
    assert r.default_model_id_of_type("embedding") == "dashscope/emb-a"
    assert r.default_model_id_of_type("rerank") == "dashscope/rr-a"


def test_default_embedding_absent_id_raises():
    with pytest.raises(ValueError, match="default_embedding_model"):
        ProviderRouter(_defaults_catalog(default_embedding_model="dashscope/nope"))


def test_default_embedding_wrong_type_raises():
    with pytest.raises(ValueError, match="must be a embedding model"):
        ProviderRouter(_defaults_catalog(default_embedding_model="dashscope/chat"))


def test_default_rerank_wrong_type_raises():
    with pytest.raises(ValueError, match="must be a rerank model"):
        ProviderRouter(_defaults_catalog(default_rerank_model="dashscope/emb-a"))
