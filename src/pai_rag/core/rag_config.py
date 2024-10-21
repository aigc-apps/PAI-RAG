from typing import Annotated, Dict, Union
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator
from pai_rag.core.models.config import (
    BaseTraceConfig,
    NodeEnhancementConfig,
    OssStoreConfig,
    RetrieverConfig,
    SearchWebConfig,
    SynthesizerConfig,
)
from pai_rag.integrations.agent.pai.pai_agent import AgentConfig
from pai_rag.integrations.chat_store.pai.pai_chat_store import (
    LocalChatStoreConfig,
    RedisChatStoreConfig,
)
from pai_rag.integrations.data_analysis.data_analysis_config import (
    MysqlAnalysisConfig,
    PandasAnalysisConfig,
    SqliteAnalysisConfig,
)
from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
)
from pai_rag.integrations.index.pai.vector_store_config import PaiVectorIndexConfig
from pai_rag.integrations.llms.pai.llm_config import PaiBaseLlmConfig
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import NodeParserConfig
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import (
    RerankModelPostProcessorConfig,
    SimilarityPostProcessorConfig,
)
from pai_rag.integrations.readers.pai.pai_data_reader import BaseDataReaderConfig
from pai_rag.integrations.router.pai.pai_router import IntentConfig


def validate_case_insensitive(value: Dict) -> Dict:
    if value is None:
        return value

    keys = ["type", "source", "reranker_type"]
    for key in keys:
        if key in value:
            value[key] = value[key].lower()
            # fix old config
            if value[key] == "pai-llm-trace":
                value[key] = "pai_trace"
    return value


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # reader, parser
    data_reader: BaseDataReaderConfig
    node_parser: NodeParserConfig

    # index
    index: PaiVectorIndexConfig

    # embedding
    embedding: Annotated[
        Union[PaiBaseEmbeddingConfig.get_subclasses()],
        Field(discriminator="source"),
        BeforeValidator(validate_case_insensitive),
    ]
    multimodal_embedding: Annotated[
        Union[PaiBaseEmbeddingConfig.get_subclasses()],
        Field(discriminator="source"),
        BeforeValidator(validate_case_insensitive),
    ] | None = None

    # llm
    llm: Annotated[
        Union[PaiBaseLlmConfig.get_subclasses()],
        Field(discriminator="source"),
        BeforeValidator(validate_case_insensitive),
    ]
    multimodal_llm: Annotated[
        Union[PaiBaseLlmConfig.get_subclasses()],
        Field(discriminator="source"),
        BeforeValidator(validate_case_insensitive),
    ] | None = None

    # currently not used
    functioncalling_llm: Annotated[
        Union[PaiBaseLlmConfig.get_subclasses()],
        Field(discriminator="source"),
        BeforeValidator(validate_case_insensitive),
    ] | None = None

    # agent
    agent: AgentConfig

    # chat_store
    chat_store: Annotated[
        Union[LocalChatStoreConfig, RedisChatStoreConfig],
        Field(discriminator="type"),
        BeforeValidator(validate_case_insensitive),
    ]

    # data_analysis
    data_analysis: Annotated[
        Union[PandasAnalysisConfig, SqliteAnalysisConfig, MysqlAnalysisConfig],
        Field(discriminator="type"),
        BeforeValidator(validate_case_insensitive),
    ]

    intent: IntentConfig | None = None

    # node_enhancement
    node_enhancement: NodeEnhancementConfig

    # oss_store
    oss_store: OssStoreConfig

    # postprocessor
    postprocessor: Annotated[
        Union[SimilarityPostProcessorConfig, RerankModelPostProcessorConfig],
        Field(discriminator="reranker_type"),
        BeforeValidator(validate_case_insensitive),
    ]

    retriever: RetrieverConfig

    # search web
    search: SearchWebConfig

    # synthesizer
    synthesizer: SynthesizerConfig

    # trace
    trace: Annotated[
        Union[BaseTraceConfig.get_subclasses()],
        Field(discriminator="type"),
        BeforeValidator(validate_case_insensitive),
    ] | None = None
