from typing import Any
from pai.llm_trace.instrument.instrumentor import LlamaIndexInstrumentor
from pai.llm_trace.instrument.pop_exporter import PopExporter


class RagTrace:
    @staticmethod
    def initialize(trace_config: Any):
        print("token:", trace_config.token)
        LlamaIndexInstrumentor(
            PopExporter(
                llm_app_name="PAI-RAG",
                llm_app_version="0.0.1",  # 应用对应的版本
                llm_app_user="pai",  # 改应用的开发者（用户）
                # 填入你的token
                token=trace_config.token,
                region=trace_config.region,  # 注意，这里填cn-hangzhou
                endpoint=trace_config.endpoint,
            )
        ).instrument()
