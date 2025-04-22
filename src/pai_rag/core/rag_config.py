from typing import Annotated, Dict, Union, List
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator
from pai_rag.core.models.config import (
    AliyunTextModerationPlusConfig,
    NodeEnhancementConfig,
    OssStoreConfig,
    QueryRewriteConfig,
    RetrieverConfig,
    SynthesizerConfig,
    ChatConfig,
)
from pai_rag.extensions.news.news_config import MiaobiNewsConfig
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
from pai_rag.integrations.llms.pai.llm_config import (
    DashScopeMultiModalLlmConfig,
    OpenAILlmConfig,
    PaiBaseLlmConfig,
    PaiEasLlmConfig,
    OpenAICompatibleLlmConfig,
)
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import NodeParserConfig
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import (
    RerankModelPostProcessorConfig,
    SimilarityPostProcessorConfig,
)
from pai_rag.integrations.readers.pai.pai_data_reader import BaseDataReaderConfig
from pai_rag.integrations.router.pai.pai_router import IntentConfig
from pai_rag.integrations.search.search_config import (
    BingSearchConfig,
    QuarkSearchConfig,
    AliyunSearchConfig,
    GoogleSearchConfig,
)
from pai_rag.integrations.trace.trace_config import TraceConfig


def validate_case_insensitive(value: Dict) -> Dict:
    if value is None:
        return value

    if isinstance(value, PaiBaseLlmConfig):
        value = value.model_dump()

    keys = ["type", "source", "reranker_type"]
    for key in keys:
        if key in value:
            value[key] = value[key].lower()
            # fix old config
            if value[key] == "simple-weighted-reranker":
                value[key] = "no-reranker"

    if value.get("source") == "paieas":
        value["source"] = "openai_compatible"
        value["base_url"] = value["endpoint"]
        value["api_key"] = str(value["token"])
    elif value.get("source") == "dashscope" and "embed_batch_size" not in value:
        value["source"] = "openai_compatible"
    return value


class SystemConfig(BaseModel):
    default_web_search: bool = False
    query_type: str | None = "rag"


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # system
    system: SystemConfig = SystemConfig()

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

    llms: Annotated[
        List[Union[PaiBaseLlmConfig.get_subclasses()]],
        Field(default_factory=list),
        BeforeValidator(lambda x: [validate_case_insensitive(item) for item in x]),
    ]

    chat: ChatConfig = ChatConfig()

    # llm
    llm: Annotated[
        Union[PaiBaseLlmConfig.get_subclasses()],
        Field(discriminator="source"),
        BeforeValidator(validate_case_insensitive),
    ]
    multimodal_llm: Annotated[
        Union[
            DashScopeMultiModalLlmConfig,
            PaiEasLlmConfig,
            OpenAILlmConfig,
            OpenAICompatibleLlmConfig,
        ],
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

    intent: IntentConfig = IntentConfig()

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
    search: Annotated[
        Union[
            BingSearchConfig, QuarkSearchConfig, AliyunSearchConfig, GoogleSearchConfig
        ],
        Field(discriminator="source"),
        BeforeValidator(validate_case_insensitive),
    ]

    # synthesizer
    synthesizer: SynthesizerConfig

    query_rewrite: QueryRewriteConfig = QueryRewriteConfig()

    guardrail: AliyunTextModerationPlusConfig = AliyunTextModerationPlusConfig()

    news_extension: MiaobiNewsConfig = MiaobiNewsConfig()

    trace: TraceConfig = TraceConfig()
