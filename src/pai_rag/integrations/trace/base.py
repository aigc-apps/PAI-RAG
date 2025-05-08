import socket
from functools import wraps
from typing import Callable, AsyncGenerator
from loguru import logger
from pydantic.v1 import json as pydantic_v1_json
from pydantic import json as pydantic_json

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.resources import (
    HOST_NAME,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.trace import Span, use_span

from pai_rag.integrations.trace.trace_config import TraceConfig
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


def init_trace(config: TraceConfig):
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

    resource = Resource(attributes=attributes)
    exporter = GRPCExporter(endpoint=grpc_endpoint, headers=(f"Authentication={token}"))

    span_processor = BatchSpanProcessor(exporter)
    trace_provider = TracerProvider(
        resource=resource, active_span_processor=span_processor
    )
    trace.set_tracer_provider(trace_provider)

    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    logger.info("Init trace successfully.")


def use_current_span(span: Span):
    """use current span, connect to span in async call"""

    def decorator(func: Callable[..., AsyncGenerator]):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncGenerator:
            with use_span(span, end_on_exit=False):
                async for item in func(*args, **kwargs):
                    yield item

        return wrapper

    return decorator


# arize instrumentation uses: pydantic.v1.json.pydantic_encoder
# but pydantic.v1.json.pydantic_encoder explicitly check v1
# this caused llama-index obj fails pydantic/v1/json.py#L77
# from pydantic.v1.main import BaseModel
# if isinstance(obj, BaseModel):
pydantic_v1_json.pydantic_encoder = pydantic_json.pydantic_encoder
