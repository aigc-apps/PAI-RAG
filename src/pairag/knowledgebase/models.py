from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Dict, Union

from pairag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
)
from pairag.knowledgebase.index.pai.vector_store_config import BaseVectorStoreConfig
from pairag.file.nodeparsers.pai.pai_node_parser import NodeParserConfig
from pairag.integrations.synthesizer.prompt_templates import (
    DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_ROLE_TEMPLATE,
)
from pairag.utils.constants import DEFAULT_KNOWLEDGEBASE_NAME

from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from pairag.integrations.postprocessor.pai.pai_postprocessor import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_SIMILARITY_THRESHOLD,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_SIMILARITY_THRESHOLD,
)


class KnowledgeBase(BaseModel):
    name: str = Field(
        default=DEFAULT_KNOWLEDGEBASE_NAME,
        description="Knowledgebase name.",
        pattern=r"^[0-9a-zA-Z_-]{3, 20}$",
    )

    vector_store_config: Annotated[
        Union[BaseVectorStoreConfig.get_subclasses()], Field(discriminator="type")
    ]
    node_parser_config: NodeParserConfig = Field(default_factory=NodeParserConfig)
    embedding_config: Annotated[
        Union[PaiBaseEmbeddingConfig.get_subclasses()], Field(discriminator="source")
    ]
    retrieval_settings: Dict = Field(default_factory=dict)
    qa_prompt_templates: Dict = {
        "system_prompt_template": DEFAULT_SYSTEM_ROLE_TEMPLATE,
        "task_prompt_template": DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    }

    @model_validator(mode="before")
    def preprocess(cls, values: Dict) -> Dict:
        if "index_name" in values:
            values["name"] = values["index_name"]
        return values

    def model_post_init(self, context):
        # 修改retrieval_settings的key
        default_retrieval_settings = {
            "retrieval_mode": "default",
            "similarity_top_k": DEFAULT_SIMILARITY_TOP_K,
            "reranker_type": "no-reranker",
            "reranker_model": DEFAULT_RERANK_MODEL,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "reranker_similarity_threshold": DEFAULT_RERANK_SIMILARITY_THRESHOLD,
            "reranker_similarity_top_k": DEFAULT_RERANK_TOP_N,
        }
        default_retrieval_settings.update(self.retrieval_settings)
        self.retrieval_settings = default_retrieval_settings
