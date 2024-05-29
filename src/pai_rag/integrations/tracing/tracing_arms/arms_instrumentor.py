from typing import Any, Collection

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from openinference.instrumentation.llama_index._callback import (
    OpenInferenceTraceCallbackHandler as _OpenInferenceTraceCallbackHandler,
)

import llama_index.core


class ArmsCallbackHandler(_OpenInferenceTraceCallbackHandler):
    def __init__(self, tracer: trace.Tracer) -> None:
        super().__init__(tracer=tracer)


class ArmsLlamaIndexInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return ["llama-index >= 0.10.0"]

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace.get_tracer_provider()
        tracer = trace.get_tracer(__name__, tracer_provider=tracer_provider)

        self._original_global_handler = llama_index.core.global_handler
        llama_index.core.global_handler = ArmsCallbackHandler(tracer=tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        llama_index.core.global_handler = self._original_global_handler
        self._original_global_handler = None
