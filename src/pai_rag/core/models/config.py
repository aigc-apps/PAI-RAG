from enum import Enum
from typing import List, Literal
from pydantic import BaseModel
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pai_rag.integrations.synthesizer.pai_synthesizer import (
    DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
)


DEFAULT_WEIGHTED_RANK_VECTOR_WEIGHT = 0.7
DEFAULT_WEIGHTED_RANK_KEYWORD_WEIGHT = 0.3


class NodeEnhancementConfig(BaseModel):
    tree_depth: int = 3
    max_clusters: int = 52
    proba_threshold: float = 0.10


class OssStoreConfig(BaseModel):
    bucket: str | None = None
    endpoint: str = "oss-cn-hangzhou.aliyuncs.com"
    ak: str | None = None
    sk: str | None = None


class RetrieverConfig(BaseModel):
    vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT
    similarity_top_k: int = 5
    image_similarity_top_k: int = 2
    search_image: bool = False
    hybrid_fusion_weights: List[float] = [
        DEFAULT_WEIGHTED_RANK_VECTOR_WEIGHT,
        DEFAULT_WEIGHTED_RANK_KEYWORD_WEIGHT,
    ]


class SearchWebConfig(BaseModel):
    search_api_key: str | None = None
    search_count: int = 10
    search_lang: str = "zh-CN"


class SynthesizerConfig(BaseModel):
    use_multimodal_llm: bool = False
    text_qa_template: str = DEFAULT_TEXT_QA_PROMPT_SEL
    multimodal_qa_template: str = DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL


class TraceType(str, Enum):
    """Trace types."""

    arize_pheonix = "arize_phoenix"
    pai_trace = "pai_trace"
    default = "no_trace"


class BaseTraceConfig(BaseModel):
    type: str = TraceType.default

    @classmethod
    def get_subclasses(cls):
        return tuple(cls.__subclasses__())


class PaiTraceConfig(BaseTraceConfig):
    type: Literal[TraceType.pai_trace] = TraceType.pai_trace
    endpoint: str | None = None
    token: str | None = None
    app_name: str = "PAI-RAG"


class ArizeTraceConfig(BaseTraceConfig):
    type: Literal[TraceType.arize_pheonix] = TraceType.arize_pheonix
