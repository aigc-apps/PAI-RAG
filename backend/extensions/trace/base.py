import os
import socket
from functools import wraps
from typing import Callable, AsyncGenerator, Union
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
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes, MessageAttributes, MessageContentAttributes

from extensions.trace.grpc_exporter import ReloadableGrpcOTLPSpanExporter
from extensions.trace.http_exporter import ReloadableHttpOTLPSpanExporter
from extensions.trace import context as trace_context
from extensions.trace.trace_config import TraceConfig
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from extensions.trace.baggage_processor import LoongSuiteBaggageSpanProcessor
from extensions.trace.trace_context_middleware import TraceContextMiddleware


ENABLE_TRACE_DEBUG = os.getenv("ENABLE_TRACE_DEBUG", "false").lower() in ["true", "1", "yes", "y"]


def setup_propagator(app):
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),  # 处理 traceparent
        W3CBaggagePropagator()           # 处理 baggage
    ]))
    app.add_middleware(TraceContextMiddleware)


# trace_provider为singleton, 不支持覆盖，故修改trace配置时，默认覆盖exporter和resource
# 这样如果用户填错密码，还可以成功刷新
trace_config: TraceConfig = None
exporter: Union[ReloadableGrpcOTLPSpanExporter, ReloadableHttpOTLPSpanExporter] = None
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

    trace_endpoint = config.endpoint
    token = config.token
    service_name = config.service_name
    service_app_name = config.service_name

    attributes = {SERVICE_NAME: service_name, HOST_NAME: socket.gethostname()}

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
        if config.exporter_type == "grpc":
            logger.info(f"Use grpc exporter: {trace_endpoint}")
            exporter = ReloadableGrpcOTLPSpanExporter(
                endpoint=trace_endpoint, headers=(f"Authentication={token}")
            )
        elif config.exporter_type == "http":
            logger.info(f"Use http exporter: {trace_endpoint}")
            exporter = ReloadableHttpOTLPSpanExporter(
                endpoint=trace_endpoint, headers=(f"Authentication={token}")
            )
        else:
            raise ValueError(f"Invalid exporter type: {config.exporter_type}")
    else:
        exporter.reload(endpoint=trace_endpoint, headers=(f"Authentication={token}"))

    global trace_provider
    if trace_provider is None:
        span_processor = BatchSpanProcessor(exporter)
        trace_provider = TracerProvider(
            resource=resource
        )
        trace_provider.add_span_processor(span_processor)
        trace_provider.add_span_processor(LoongSuiteBaggageSpanProcessor())
        if ENABLE_TRACE_DEBUG:
            trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
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
                trace_id = format(span.get_span_context().trace_id, '032x')
                with trace.use_span(span, end_on_exit=False):
                    async for item in func(*args, **kwargs):
                        if hasattr(item, 'trace_id'):
                            item.trace_id = trace_id
                        yield item
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

    # Message & MessageContent attributes key
    for attr_name in dir(MessageAttributes):
        if attr_name.startswith("__"):
            continue

        value = getattr(MessageAttributes, attr_name)
        if isinstance(value, str) and value == "message.contents":
            new_value = "message.content"
            setattr(MessageAttributes, attr_name, new_value)

    for attr_name in dir(MessageContentAttributes):
        if attr_name.startswith("__"):
            continue

        value = getattr(MessageContentAttributes, attr_name)
        if isinstance(value, str) and value.startswith("message_content."):
            new_value = value.replace("message_content.", "")
            setattr(MessageContentAttributes, attr_name, new_value)


gen_ai_semantic_conversion()
