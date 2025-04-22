import atexit
import logging
import os
import socket

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    HOST_NAME,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCExporter,
)
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

from pai_rag.integrations.trace.trace_config import TraceConfig
from loguru import logger
from pai_rag.integrations.trace.llama_index import LlamaIndexInstrumentor



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
    exporter = GRPCExporter(
        endpoint=grpc_endpoint, headers=(f"Authentication={token}")
    )

    span_processor = BatchSpanProcessor(exporter)
    trace_provider = TracerProvider(
        resource=resource, active_span_processor=span_processor
    )
    trace.set_tracer_provider(trace_provider)

    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    logger.info("Init trace successfully.")
