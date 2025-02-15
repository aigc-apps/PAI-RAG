from typing import List
from pydantic import BaseModel
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pai_rag.integrations.synthesizer.prompt_templates import (
    DEFAULT_SYSTEM_ROLE_TEMPLATE,
    DEFAULT_CUSTOM_PROMPT_TEMPLATE,
)


DEFAULT_WEIGHTED_RANK_VECTOR_WEIGHT = 0.7
DEFAULT_WEIGHTED_RANK_KEYWORD_WEIGHT = 0.3


class AliyunTextModerationPlusConfig(BaseModel):
    endpoint: str = "green-cip.cn-hangzhou.aliyuncs.com"
    region: str = "cn-hangzhou"
    access_key_id: str | None = None
    access_key_secret: str | None = None
    custom_advice: str | None = None

    def is_enabled(self) -> bool:
        return self.access_key_id is not None and self.access_key_secret is not None


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


class SynthesizerConfig(BaseModel):
    use_multimodal_llm: bool = False
    system_role_template: str = DEFAULT_SYSTEM_ROLE_TEMPLATE
    custom_prompt_template: str = DEFAULT_CUSTOM_PROMPT_TEMPLATE
