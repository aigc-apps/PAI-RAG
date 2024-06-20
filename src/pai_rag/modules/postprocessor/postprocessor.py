"""postprocessor factory, used to generate postprocessor instance based on customer config"""

import os
import logging
from typing import Dict, List, Any

# from modules.query.postprocessor.base import BaseNodePostprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.postprocessor.my_llm_rerank import MyLLMRerank
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

DEFAULT_RANK_MODEL = "bge-reranker-base"
DEFAULT_RANK_TOP_N = 2

logger = logging.getLogger(__name__)


class PostprocessorModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        llm = new_params["LlmModule"]
        post_processors = []

        if "similarity_cutoff" in config:
            similarity_cutoff = config["similarity_cutoff"]
            logger.info(
                f"[PostProcessor]: SimilarityPostprocessor used with threshold {similarity_cutoff}."
            )
            post_processors.append(
                SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
            )

        rerank_model = config.get("rerank_model", "")
        if rerank_model == "llm-reranker":
            top_n = config.get("top_n", DEFAULT_RANK_TOP_N)
            logger.info(f"[PostProcessor]: Llm reranker used with top_n {top_n}.")
            post_processors.append(MyLLMRerank(top_n=top_n, llm=llm))

        elif (
            rerank_model == "bge-reranker-base" or rerank_model == "bge-reranker-large"
        ):
            model_dir = config.get("rerank_model_dir", DEFAULT_MODEL_DIR)
            model_name = config.get("rerank_model_name", rerank_model)
            model = os.path.join(model_dir, model_name)
            top_n = config.get("top_n", DEFAULT_RANK_TOP_N)
            logger.info(
                f"[PostProcessor]: Reranker model used with top_n {top_n}, model {model_name}."
            )
            post_processors.append(
                FlagEmbeddingReranker(model=model, top_n=top_n, use_fp16=True),
            )

        else:
            logger.info("[PostProcessor]: No Reranker used.")

        return post_processors
