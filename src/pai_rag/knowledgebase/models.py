from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Dict, Union

from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
)
from pai_rag.knowledgebase.index.pai.vector_store_config import BaseVectorStoreConfig
from pai_rag.file.nodeparsers.pai.pai_node_parser import NodeParserConfig
from pai_rag.integrations.synthesizer.prompt_templates import (
    DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_ROLE_TEMPLATE,
)
from pai_rag.utils.constants import DEFAULT_KNOWLEDGEBASE_NAME


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
