"""Wire the OpenTelemetry TracerProvider + OTLP exporter from the environment.

``init_tracing`` is idempotent and fully defensive: any missing package,
malformed config, or unreachable collector degrades to "tracing off" with a
single warning — it never blocks or breaks application startup. The heavy
opentelemetry imports live inside the function body so merely importing this
module (e.g. from a guarded try/except in the core) does not require the SDK.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from .config import TraceConfig
from .tracer import _set_enabled, is_enabled

_initialized = False
_provider = None


def _build_exporter(cfg: TraceConfig):
    """Construct the OTLP span exporter for the configured protocol.

    For http/protobuf the OTel spec has the SDK append ``/v1/traces`` to the base
    endpoint, so we do that ourselves (Langfuse's ``…/api/public/otel`` base
    likewise expects the ``/v1/traces`` suffix). gRPC takes the bare host:port.
    """
    if cfg.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(endpoint=cfg.endpoint, headers=cfg.headers or None)

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    endpoint = cfg.endpoint
    if not endpoint.rstrip("/").endswith("/v1/traces"):
        endpoint = endpoint.rstrip("/") + "/v1/traces"
    return OTLPSpanExporter(endpoint=endpoint, headers=cfg.headers or None)


def init_tracing(settings: Optional[object] = None) -> bool:
    """Initialise tracing once. Returns True iff an exporter was wired.

    ``settings`` is accepted for future use (e.g. deriving a service name) but the
    configuration is driven entirely by environment variables today.
    """
    global _initialized, _provider
    if _initialized:
        return is_enabled()
    _initialized = True

    cfg = TraceConfig.from_env()
    if not cfg.enabled:
        logger.info("[trace] tracing disabled (no OTLP endpoint / LANGFUSE_* configured)")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        resource = Resource.create({"service.name": cfg.service_name})
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(cfg.sample_ratio)),
        )
        provider.add_span_processor(BatchSpanProcessor(_build_exporter(cfg)))
        trace.set_tracer_provider(provider)
        _provider = provider
        _set_enabled(True)
    except Exception as e:  # missing SDK/exporter, bad endpoint, etc.
        logger.warning("[trace] failed to initialise OTLP tracing, continuing without it: {}", e)
        _set_enabled(False)
        return False

    # Auto-instrument the OpenAI client so every chat.completions.create() (incl.
    # streaming + tool calls + token usage) becomes an OpenInference LLM span that
    # nests under the active agent span. Best-effort: absence of openinference must
    # not disable the manual agent/tool spans.
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=_provider)
        logger.info("[trace] OpenAI auto-instrumentation enabled")
    except Exception as e:
        logger.warning("[trace] OpenAI auto-instrumentation unavailable ({}); "
                       "LLM generations will not be traced automatically", e)

    logger.info("[trace] tracing enabled → {} ({}), service={} sample={}",
                cfg.endpoint, cfg.protocol, cfg.service_name, cfg.sample_ratio)
    return True


def instrument_fastapi(app) -> None:
    """Instrument a FastAPI app so HTTP requests become root server spans.

    No-op when tracing is disabled or the instrumentation package is absent.
    """
    if not is_enabled():
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("[trace] FastAPI instrumentation enabled")
    except Exception as e:
        logger.warning("[trace] FastAPI instrumentation unavailable: {}", e)


def shutdown_tracing() -> None:
    """Flush and shut down the provider so buffered spans are exported on exit."""
    global _provider
    if _provider is None:
        return
    try:
        _provider.force_flush()
    except Exception:
        pass
    try:
        _provider.shutdown()
    except Exception:
        pass
    _provider = None
    _set_enabled(False)
