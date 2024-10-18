from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel
from llama_index.core.bridge.pydantic import PrivateAttr
import os
from pai_rag.integrations.postprocessor.my_model_based_reranker import (
    MyModelBasedReranker,
)
from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
import logging

logger = logging.getLogger(__name__)

# rerank constants
DEFAULT_RERANK_MODEL = "bge-reranker-base"
DEFAULT_SIMILARITY_THRESHOLD = 0
DEFAULT_RERANK_SIMILARITY_THRESHOLD = 0
DEFAULT_RERANK_TOP_N = 2


class PostProcessorType(str, Enum):
    no_reranker = "no-reranker"
    reranker_model = "model-based-reranker"


class BasePostProcessorConfig(BaseModel):
    reranker_type: Literal[
        PostProcessorType.no_reranker
    ] = PostProcessorType.no_reranker


class SimilarityPostProcessorConfig(BasePostProcessorConfig):
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD


class RerankModelPostProcessorConfig(BasePostProcessorConfig):
    reranker_type: Literal[
        PostProcessorType.reranker_model
    ] = PostProcessorType.reranker_model
    reranker_model: str = DEFAULT_RERANK_MODEL
    top_n: int = DEFAULT_RERANK_TOP_N
    similarity_threshold: float = DEFAULT_RERANK_SIMILARITY_THRESHOLD


def create_postprocessors(config: BasePostProcessorConfig):
    postprocessors = []
    if isinstance(config, RerankModelPostProcessorConfig):
        pai_model_dir = os.getenv("PAI_RAG_MODEL_DIR", "./model_repository")
        model = os.path.join(pai_model_dir, config.reranker_model)
        postprocessors.append(
            MyModelBasedReranker(
                model=model,
                top_n=config.top_n,
                similarity_threshold=config.similarity_threshold,
                callback_manager=Settings.callback_manager,
            )
        )
        logger.info(
            f"""[PostProcessor]: Reranker model inited
                model_name: {model}
                top_n: {config.top_n},
                similarity_threshold: {config.similarity_threshold}"""
        )
    elif isinstance(config, SimilarityPostProcessorConfig):
        postprocessors.append(
            SimilarityPostprocessor(
                similarity_cutoff=config.similarity_threshold,
                callback_manager=Settings.callback_manager,
            )
        )

    return postprocessors


class PaiPostProcessor(BaseNodePostprocessor):
    _postprocessors: List[BaseNodePostprocessor] = PrivateAttr()

    def __init__(
        self,
        postprocessor_config: BasePostProcessorConfig,
    ):
        self._postprocessors = create_postprocessors(postprocessor_config)
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "PaiPostProcessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        for postprocessor in self._postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle)
        return nodes
