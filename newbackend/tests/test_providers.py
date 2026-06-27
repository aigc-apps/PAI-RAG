import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.providers import ModelConfig, ModelCatalog, load_catalog, ProviderRouter


class _Settings:
    openai_base_url = "https://api.openai.com/v1"
    openai_api_key = "sk-legacy"
    default_model = "gpt-4o-mini"


def _catalog():
    return ModelCatalog(
        default_model="fast",
        models=[
            ModelConfig(id="fast", provider="openai", base_url="https://fast/v1", api_key="k1",
                        context_window=128000, max_output_tokens=8192, supports_tools=True),
            ModelConfig(id="smart", provider="anthropic", base_url="https://smart/v1", api_key="k2",
                        context_window=200000, max_output_tokens=16384, supports_reasoning=True),
        ],
    )


def test_get_config_and_unknown_raises():
    r = ProviderRouter(_catalog())
    assert r.get_config("smart").base_url == "https://smart/v1"
    assert r.default_model_id == "fast"
    with pytest.raises(KeyError):
        r.get_config("nope")


def test_get_llm_builds_client_with_model_and_reasoning():
    r = ProviderRouter(_catalog())
    fast = r.get_llm("fast")
    smart = r.get_llm("smart")
    assert fast.model == "fast" and str(fast.client.base_url).startswith("https://fast")
    assert smart.enable_thinking is True and fast.enable_thinking is False
    # cached
    assert r.get_llm("fast") is fast


def test_register_llm_overrides_client():
    r = ProviderRouter(_catalog())
    sentinel = object()
    r.register_llm("fast", sentinel)
    assert r.get_llm("fast") is sentinel


def test_list_models():
    r = ProviderRouter(_catalog())
    assert {m.id for m in r.list_models()} == {"fast", "smart"}


def test_key_resolution_direct_then_env(monkeypatch):
    monkeypatch.setenv("MYKEY", "from-env")
    cat = ModelCatalog(default_model="a", models=[
        ModelConfig(id="a", provider="x", base_url="u", api_key_env="MYKEY"),
    ])
    r = ProviderRouter(cat)
    assert r.get_llm("a").client.api_key == "from-env"


def test_model_with_missing_required_key_is_omitted(monkeypatch):
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    cat = ModelCatalog(default_model="present", models=[
        ModelConfig(id="present", provider="x", base_url="u", api_key="k"),
        ModelConfig(id="absent", provider="x", base_url="u", api_key_env="ABSENT_KEY"),
    ])
    r = ProviderRouter(cat)
    assert "absent" not in {m.id for m in r.list_models()}
    assert "present" in {m.id for m in r.list_models()}


def test_reload_swaps_configs_and_preserves_unchanged_warm_client():
    r = ProviderRouter(_catalog())
    fast_before = r.get_llm("fast")   # warm
    r.get_llm("smart")
    # reload: 'fast' unchanged, 'smart' changed (new base_url), 'extra' added
    r.reload(ModelCatalog(default_model="fast", models=[
        ModelConfig(id="fast", provider="openai", base_url="https://fast/v1", api_key="k1",
                    context_window=128000, max_output_tokens=8192, supports_tools=True),
        ModelConfig(id="smart", provider="anthropic", base_url="https://smart2/v1", api_key="k2"),
        ModelConfig(id="extra", provider="x", base_url="https://extra/v1", api_key="k3"),
    ]))
    assert r.get_llm("fast") is fast_before          # unchanged -> warm client preserved
    assert str(r.get_llm("smart").client.base_url).startswith("https://smart2")  # changed -> rebuilt
    assert "extra" in {m.id for m in r.list_models()}


def test_load_catalog_from_yaml(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(
        "default_model: m1\n"
        "models:\n"
        "  - id: m1\n"
        "    provider: openai\n"
        "    base_url: https://x/v1\n"
        "    api_key_env: OPENAI_API_KEY\n"
    )
    cat = load_catalog(str(p), _Settings())
    assert cat.default_model == "m1" and cat.models[0].base_url == "https://x/v1"


def test_load_catalog_fallback_synthesizes_from_settings():
    cat = load_catalog("/no/such/models.yaml", _Settings())
    assert cat.default_model == "gpt-4o-mini"
    m = cat.models[0]
    assert m.id == "gpt-4o-mini" and m.base_url == "https://api.openai.com/v1" and m.api_key == "sk-legacy"
