from typing import Dict, List, Any
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.evaluation.pai_evaluator import PaiEvaluator
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG


class EvaluationModule(ConfigurableModule):
    """Class for managing indices.

    RagIndex to manage vector indices for RagApplication.
    When initializing, the index is empty or load from existing index.
    User can add nodes to index when needed.
    """

    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule", "IndexModule", "RetrieverModule", "QueryEngineModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        llm = new_params["LlmModule"]
        index = new_params["IndexModule"]
        retriever = new_params["RetrieverModule"]
        query_engine = new_params["QueryEngineModule"]

        retrieval_metrics = config.get("retrieval", None)
        response_metrics = config.get("response", None)

        return PaiEvaluator(
            llm=llm,
            index=index,
            query_engine=query_engine,
            retriever=retriever,
            retrieval_metrics=retrieval_metrics,
            response_metrics=response_metrics,
        )
