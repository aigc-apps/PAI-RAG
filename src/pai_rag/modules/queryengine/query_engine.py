"""query engine factory based on config"""

import logging
from typing import Dict, List, Any

# from llama_index.core.query_engine import RetrieverQueryEngine
from pai_rag.modules.queryengine.my_retriever_query_engine import MyRetrieverQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG


logger = logging.getLogger(__name__)


class QueryEngineModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["RetrieverModule", "SynthesizerModule", "PostprocessorModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        retriever = new_params["RetrieverModule"]
        synthesizer = new_params["SynthesizerModule"]
        postprocessor = new_params["PostprocessorModule"]

        if config["type"] == "RetrieverQueryEngine":
            if not postprocessor:
                my_query_engine = MyRetrieverQueryEngine(
                    retriever=retriever, response_synthesizer=synthesizer
                )
            elif isinstance(postprocessor, List):
                my_query_engine = MyRetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=synthesizer,
                    node_postprocessors=postprocessor,
                )
            else:
                my_query_engine = MyRetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=synthesizer,
                    node_postprocessors=[postprocessor],
                )
            logger.info("Query_engine instance created")
            return my_query_engine

        if config["type"] == "TransformQueryEngine":
            hyde = HyDEQueryTransform(include_original=True)
            if not postprocessor:
                my_query_engine = MyRetrieverQueryEngine(
                    retriever=retriever, response_synthesizer=synthesizer
                )
                hyde_query_engine = TransformQueryEngine(my_query_engine, hyde)
            elif isinstance(postprocessor, List):
                my_query_engine = MyRetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=synthesizer,
                    node_postprocessors=postprocessor,
                )
                hyde_query_engine = TransformQueryEngine(my_query_engine, hyde)
            else:
                my_query_engine = MyRetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=synthesizer,
                    node_postprocessors=[postprocessor],
                )
                hyde_query_engine = TransformQueryEngine(my_query_engine, hyde)

            logger.info("HyDE_query_engine instance created")
            return hyde_query_engine

        else:
            raise ValueError(
                "Supports RetrieverQueryEngine & TransformQueryEngine only."
            )
