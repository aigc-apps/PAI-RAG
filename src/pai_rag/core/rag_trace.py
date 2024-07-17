import socket
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import (
    Resource,
    HOST_NAME,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPSpanGrpcExporter,
)

from aliyun.instrumentation.llama_index import AliyunLlamaIndexInstrumentor


def start_trace(rag_config):
    # set up instrumentation
    resource = Resource(
        attributes={
            # 设置您的应用名称，版本等基础信息。
            SERVICE_NAME: rag_config.name,
            SERVICE_VERSION: rag_config.version,
            HOST_NAME: socket.gethostname(),
        }
    )
    # 使用GRPC协议上报
    span_processor = BatchSpanProcessor(
        OTLPSpanGrpcExporter(
            endpoint=rag_config.trace.endpoint,
            headers=(f"Authentication={rag_config.trace.token}"),
        )
    )
    trace_provider = TracerProvider(
        resource=resource, active_span_processor=span_processor
    )
    trace.set_tracer_provider(trace_provider)
    # aliyun llama-index instrumentor
    AliyunLlamaIndexInstrumentor().instrument()
