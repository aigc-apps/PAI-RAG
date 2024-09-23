"""retriever factory, used to generate retriever instance based on customer config and index"""

import logging
from typing import Dict, List, Any


from pai_rag.integrations.index.pai.pai_vector_index import (
    retrieval_type_to_search_mode,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTED_RANK_VECTOR_WEIGHT = 0.7
DEFAULT_WEIGHTED_RANK_KEYWORD_WEIGHT = 0.3


class RetrieverModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["IndexModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        pai_index = new_params["IndexModule"]
        vector_weight = config.get("vector_weight", DEFAULT_WEIGHTED_RANK_VECTOR_WEIGHT)
        keyword_weight = config.get(
            "keyword_weight", DEFAULT_WEIGHTED_RANK_KEYWORD_WEIGHT
        )

        retrieval_args = {
            "similarity_top_k": config.get("similarity_top_k"),
            "image_similarity_top_k": config.get("image_similarity_top_k"),
            "search_image": config.get("need_image", False),
            "hybrid_fusion_weights": [vector_weight, keyword_weight],
        }
        if config.get("retrieval_mode"):
            retrieval_args["vector_store_query_mode"] = retrieval_type_to_search_mode(
                config.get("retrieval_mode")
            )

        return pai_index.as_retriever(**retrieval_args)
