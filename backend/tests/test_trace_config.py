from extensions.trace.config import TraceConfig


def test_langfuse_base_url_builds_self_hosted_otlp_endpoint():
    cfg = TraceConfig.from_env(
        {
            "LANGFUSE_BASE_URL": " https://langfuse.example.com/ ",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
    )

    assert cfg.endpoint == "https://langfuse.example.com/api/public/otel"
    assert cfg.enabled is True
    assert cfg.protocol == "http/protobuf"
    assert cfg.headers["Authorization"].startswith("Basic ")


def test_explicit_otlp_endpoint_wins_over_langfuse_base_url():
    cfg = TraceConfig.from_env(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example.com",
            "LANGFUSE_BASE_URL": "https://langfuse.example.com",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
    )

    assert cfg.endpoint == "https://collector.example.com"


def test_langfuse_host_is_ignored():
    cfg = TraceConfig.from_env(
        {
            "LANGFUSE_HOST": "https://legacy.example.com",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }
    )

    assert cfg.endpoint == "https://cloud.langfuse.com/api/public/otel"
