import os
import socket
from functools import wraps
from typing import Callable, AsyncGenerator
from loguru import logger

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.resources import (
    HOST_NAME,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.trace import Span
from opentelemetry.context import attach, detach
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes

from extensions.trace.reloadable_exporter import ReloadableOTLPSpanExporter
from extensions.trace import context as trace_context
from extensions.trace.trace_config import TraceConfig


# trace_provider为singleton, 不支持覆盖，故修改trace配置时，默认覆盖exporter和resource
# 这样如果用户填错密码，还可以成功刷新
trace_config: TraceConfig = None
exporter: ReloadableOTLPSpanExporter = None
resource: Resource = None
trace_provider: TracerProvider = None


def init_instrument(config: TraceConfig):
    global trace_config
    if config == trace_config:
        logger.info("Trace config not changed.")
        return

    if not config.is_enabled():
        os.environ["TRACING_ENABLED"] = "false"
        OpenAIInstrumentor().uninstrument()
        trace_config = config
        logger.info("Tracing is DISABLED.")
        return

    if config.user_args:
        trace_context.init_custom_context(config.user_args.values())

    grpc_endpoint = config.endpoint
    token = config.token
    service_name = config.service_name
    service_app_name = config.service_name

    attributes = {SERVICE_NAME: service_name, HOST_NAME: socket.gethostname()}

    if not token:
        logger.error("token not provided in trace config.")
        raise ValueError("token must be provided!")

    attributes["service.app.name"] = service_app_name

    # ToDo: change to adaptive versioning
    attributes[SERVICE_VERSION] = "1.1.0"

    global resource
    if resource is None:
        resource = Resource(attributes=attributes)
    else:
        resource._attributes = attributes

    global exporter
    if exporter is None:
        exporter = ReloadableOTLPSpanExporter(
            endpoint=grpc_endpoint, headers=(f"Authentication={token}")
        )
    else:
        exporter.reload(endpoint=grpc_endpoint, headers=(f"Authentication={token}"))

    global trace_provider
    if trace_provider is None:
        span_processor = BatchSpanProcessor(exporter)
        trace_provider = TracerProvider(
            resource=resource, active_span_processor=span_processor
        )

        trace.set_tracer_provider(trace_provider)

    OpenAIInstrumentor().instrument()
    os.environ["TRACING_ENABLED"] = "true"
    logger.info("Init trace successfully.")

    trace_config = config


def use_current_span(span: Span):
    """use current span, connect to span in async call"""

    def decorator(func: Callable[..., AsyncGenerator]):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncGenerator:
            if span and span.is_recording():
                ctx = trace.set_span_in_context(span)
                token = attach(ctx)
                try:
                    async for item in func(*args, **kwargs):
                        yield item
                finally:
                    detach(token)
            else:
                async for item in func(*args, **kwargs):
                    yield item

        return wrapper

    return decorator


def gen_ai_semantic_conversion():
    """
    semantic conversion:
    ref: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
    ref: https://www.alibabacloud.com/help/zh/arms/application-monitoring/developer-reference/llm-trace-field-definition-description
    """
    for attr_name in dir(SpanAttributes):
        if attr_name.startswith("__"):
            continue

        value = getattr(SpanAttributes, attr_name)
        if isinstance(value, str) and value.startswith("llm."):
            if value == "llm.invocation_parameters":
                new_value = "gen_ai.request.parameters"
            elif value.startswith("llm.token_count.prompt"):
                new_value = "gen_ai.usage.input_tokens"
            elif value.startswith("llm.token_count.completion"):
                new_value = "gen_ai.usage.output_tokens"
            elif value.startswith("llm.token_count.total"):
                new_value = "gen_ai.usage.total_tokens"
            elif value.startswith("llm.input_messages"):
                new_value = "gen_ai.prompts"
            elif value.startswith("llm.output_messages"):
                new_value = "gen_ai.completions"
            else:
                new_value = value.replace("llm.", "gen_ai.", 1)
            setattr(SpanAttributes, attr_name, new_value)


gen_ai_semantic_conversion()
