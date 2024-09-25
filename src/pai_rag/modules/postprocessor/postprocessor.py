"""postprocessor factory, used to generate postprocessor instance based on customer config"""

import os
import logging
from typing import Dict, List, Any

# from modules.query.postprocessor.base import BaseNodePostprocessor
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.integrations.postprocessor.my_model_based_reranker import (
    MyModelBasedReranker,
)
from llama_index.core.postprocessor import SimilarityPostprocessor

DEFAULT_RANK_MODEL = "bge-reranker-base"
DEFAULT_RANK_SIMILARITY_THRESHOLD = None
DEFAULT_RERANK_SIMILARITY_THRESHOLD = 0
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

        if reranker_type == "no-reranker":
            similarity_threshold = config.get(
                "similarity_threshold", DEFAULT_RANK_SIMILARITY_THRESHOLD
            )
            logger.info(
                f"[PostProcessor]: Simple weighted reranker used with and similarity_threshold: {similarity_threshold}."
            )
            post_processors.append(
                SimilarityPostprocessor(
                    similarity_cutoff=similarity_threshold,
                )
            )
        elif reranker_type == "model-based-reranker":
            top_n = config.get("top_n", DEFAULT_RANK_TOP_N)
            similarity_threshold = config.get(
                "reranker_similarity_threshold", DEFAULT_RERANK_SIMILARITY_THRESHOLD
            )
            reranker_model = config.get("reranker_model", DEFAULT_RANK_TOP_N)
            if (
                reranker_model == "bge-reranker-base"
                or reranker_model == "bge-reranker-large"
            ):
                model_dir = config.get("rerank_model_dir", DEFAULT_MODEL_DIR)
                model = os.path.join(model_dir, reranker_model)
                logger.info(
                    f"""[PostProcessor]: Reranker model inited
                        model_name: {reranker_model}
                        top_n: {top_n},
                        similarity_threshold: {similarity_threshold}"""
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
