from pai.llm_trace.instrumentation import init_opentelemetry
from pai.llm_trace.instrumentation.llama_index import LlamaIndexInstrumentor
import logging

logger = logging.getLogger(__name__)


def init_trace(trace_config):
    if trace_config is None or trace_config.token is None:
        logger.info("Trace disabled.")
    else:
        init_opentelemetry(
            LlamaIndexInstrumentor,
            grpc_endpoint=trace_config.endpoint,
            token=trace_config.token,
            service_name="PAI-RAG",
            service_version="0.1.0",
            service_id="",
            deployment_environment="",
            service_owner_id="",
            service_owner_sub_id="",
        )
        logger.info(f"Trace enabled to endpoint: '{trace_config.endpoint}'.")
