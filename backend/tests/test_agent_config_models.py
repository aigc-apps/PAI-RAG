from app.agent_config import DEFAULT_DOCUMENT, _merge_default


def test_default_model_provider_is_environment_managed():
    provider = DEFAULT_DOCUMENT.models["providers"][0]

    assert provider["name"] == "openai"
    assert provider["type"] == "openai_compatible"
    assert provider["use_default_env"] is True
    assert provider["base_url"] == ""
    assert provider["api_key"] == ""
    assert "api_key_env" not in provider


def test_merge_upgrades_legacy_shipped_default_provider():
    doc = _merge_default(
        {
            "models": {
                "default_model": "openai/custom-chat",
                "providers": [
                    {
                        "name": "openai",
                        "base_url": "https://manual.example/v1",
                        "api_key_env": "OPENAI_API_KEY",
                        "models": [{"id": "custom-chat"}],
                    }
                ],
            }
        }
    )

    provider = doc.models["providers"][0]
    assert provider["type"] == "openai_compatible"
    assert provider["use_default_env"] is True
    assert provider["base_url"] == "https://manual.example/v1"
    assert provider["api_key_env"] == "OPENAI_API_KEY"
    assert provider["models"] == [{"id": "custom-chat"}]


def test_merge_does_not_upgrade_custom_provider_named_openai():
    doc = _merge_default(
        {
            "models": {
                "default_model": "custom/chat",
                "providers": [
                    {
                        "name": "openai",
                        "base_url": "https://custom.example/v1",
                        "api_key_env": "CUSTOM_KEY",
                        "models": [{"id": "other"}],
                    },
                    {
                        "name": "custom",
                        "base_url": "https://default.example/v1",
                        "api_key": "key",
                        "models": [{"id": "chat"}],
                    },
                ],
            }
        }
    )

    provider = doc.models["providers"][0]
    assert "use_default_env" not in provider
