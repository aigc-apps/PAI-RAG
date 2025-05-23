import socket
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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

from trace.tracing_config import TracingConfig
from trace.reloadable_exporter import ReloadableOTLPSpanExporter
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

tracing_config: TracingConfig = None
exporter: ReloadableOTLPSpanExporter = None
resource: Resource = None
trace_provider: TracerProvider = None


def init_instrument(config: TracingConfig):
    global tracing_config
    if config == tracing_config:
        logger.info("Trace config not changed.")
        return

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

        LlamaIndexInstrumentor().instrument(tracer_provider=trace_provider)
        OpenAIInstrumentor().instrument(tracer_provider=trace_provider)
        logger.info("Init trace successfully.")
    else:
        logger.info("Reload trace successfully.")

    tracing_config = config


# arize instrumentation uses: pydantic.v1.json.pydantic_encoder
# but pydantic.v1.json.pydantic_encoder explicitly check v1
# this caused llama-index obj fails pydantic/v1/json.py#L77
# from pydantic.v1.main import BaseModel
# if isinstance(obj, BaseModel):
pydantic_v1_json.pydantic_encoder = pydantic_json.pydantic_encoder
