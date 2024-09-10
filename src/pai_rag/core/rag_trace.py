from pai.llm_trace.instrumentation import init_opentelemetry
from pai.llm_trace.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core import set_global_handler
import logging

logger = logging.getLogger(__name__)


def init_trace(trace_config):
    if (
        trace_config is not None
        and trace_config.type == "pai-llm-trace"
        and trace_config.token
    ):
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
        logger.info(f"Pai-LLM-Trace enabled with endpoint: '{trace_config.endpoint}'.")
    elif trace_config is not None and trace_config.type == "arize_phoenix":
        set_global_handler("arize_phoenix")
        logger.info("Arize trace enabled.")
    else:
        logger.warning("No trace used.")
