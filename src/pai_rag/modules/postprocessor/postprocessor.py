"""postprocessor factory, used to generate postprocessor instance based on customer config"""

import os
import logging
from typing import Dict, List, Any

# from modules.query.postprocessor.base import BaseNodePostprocessor
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.integrations.postprocessor.my_simple_weighted_rerank import (
    MySimpleWeightedRerank,
)
from pai_rag.integrations.postprocessor.my_model_based_reranker import (
    MyModelBasedReranker,
)

DEFAULT_RANK_MODEL = "bge-reranker-base"
DEFAULT_WEIGHTED_RANK_VECTOR_WEIGHT = 0.7
DEFAULT_WEIGHTED_RANK_KEYWORD_WEIGHT = 0.3
DEFAULT_RANK_SIMILARITY_THRES = None
DEFAULT_RANK_TOP_N = 2

logger = logging.getLogger(__name__)


class PostprocessorModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        post_processors = []

        reranker_type = config.get("reranker_type", "")

        if reranker_type == "simple-weighted-reranker":
            vector_weight = config.get(
                "vector_weight", DEFAULT_WEIGHTED_RANK_VECTOR_WEIGHT
            )
            keyword_weight = config.get(
                "keyword_weight", DEFAULT_WEIGHTED_RANK_KEYWORD_WEIGHT
            )
            top_n = config.get("top_n", DEFAULT_RANK_TOP_N)
            similarity_threshold = config.get(
                "similarity_threshold", DEFAULT_RANK_SIMILARITY_THRES
            )
            logger.info(
                f"[PostProcessor]: Simple weighted reranker used with top_n: {top_n}, keyword_weight: {keyword_weight}, vector_weight: {vector_weight}, and similarity_threshold: {similarity_threshold}."
            )
            post_processors.append(
                MySimpleWeightedRerank(
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    top_n=top_n,
                    similarity_threshold=similarity_threshold,
                )
            )
        elif reranker_type == "model-based-reranker":
            top_n = config.get("top_n", DEFAULT_RANK_TOP_N)
            similarity_threshold = config.get(
                "similarity_threshold", DEFAULT_RANK_TOP_N
            )
            reranker_model = config.get("reranker_model", DEFAULT_RANK_TOP_N)
            if (
                reranker_model == "bge-reranker-base"
                or reranker_model == "bge-reranker-large"
            ):
                model_dir = config.get("rerank_model_dir", DEFAULT_MODEL_DIR)
                model = os.path.join(model_dir, reranker_model)
                logger.info(
                    f"[PostProcessor]: Reranker model used with model-based-reranker: {reranker_model}, top_n: {top_n}, and similarity_threshold: {similarity_threshold}."
                )
                post_processors.append(
                    MyModelBasedReranker(
                        model=model,
                        top_n=top_n,
                        similarity_threshold=similarity_threshold,
                        use_fp16=True,
                    ),
                )
            else:
                raise ValueError(f"Not supported reranker_model: {reranker_model}")
        else:
            logger.info("[PostProcessor]: No Reranker used.")

        return post_processors
