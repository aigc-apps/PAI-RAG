"""query engine factory based on config"""

import logging
from typing import Dict, List, Any

from pai_rag.integrations.query_engine.pai_retriever_query_engine import (
    PaiRetrieverQueryEngine,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule


logger = logging.getLogger(__name__)


class QueryEngineModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return [
            "QueryTransformModule",
            "RetrieverModule",
            "SynthesizerModule",
            "PostprocessorModule",
            "LlmModule",
            "MultiModalLlmModule",
        ]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        query_transform = new_params["QueryTransformModule"]
        retriever = new_params["RetrieverModule"]
        synthesizer = new_params["SynthesizerModule"]
        postprocessor = new_params["PostprocessorModule"]

        if not postprocessor:
            logger.info("Query_engine without postprocess created")
            my_query_engine = PaiRetrieverQueryEngine(
                query_transform=query_transform,
                retriever=retriever,
                response_synthesizer=synthesizer,
            )
        elif isinstance(postprocessor, List):
            my_query_engine = PaiRetrieverQueryEngine(
                query_transform=query_transform,
                retriever=retriever,
                response_synthesizer=synthesizer,
                node_postprocessors=postprocessor,
            )
        else:
            my_query_engine = PaiRetrieverQueryEngine(
                query_transform=query_transform,
                retriever=retriever,
                response_synthesizer=synthesizer,
                node_postprocessors=[postprocessor],
            )
        logger.info("Query_engine instance created")
        return my_query_engine
