from typing import Annotated, Dict, Union, List
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator
from pairag.core.models.config import (
    OssStoreConfig,
    QueryRewriteConfig,
    SynthesizerConfig,
    ChatConfig,
)
from pairag.extensions.news.news_config import MiaobiNewsConfig
from pairag.integrations.guardrail.config import AliyunTextModerationPlusConfig
from pairag.integrations.data_analysis.data_analysis_config import (
    MysqlAnalysisConfig,
    PandasAnalysisConfig,
    SqliteAnalysisConfig,
)
from pairag.integrations.llms.pai.llm_config import (
    OpenAICompatibleLlmConfig,
)
from pairag.integrations.postprocessor.pai.pai_postprocessor import (
    RerankModelPostProcessorConfig,
    SimilarityPostProcessorConfig,
)
from pairag.integrations.search.search_config import (
    BingSearchConfig,
    AliyunSearchConfig,
    GoogleSearchConfig,
)
from pairag.integrations.trace.trace_config import TraceConfig
from pairag.knowledgebase.index.pai.vector_store_config import PaiVectorIndexConfig


def validate_case_insensitive(value: Dict) -> Dict:
    if value is None:
        return value

    keys = ["type", "source", "reranker_type"]
    for key in keys:
        if key in value:
            value[key] = value[key].lower()

    return value


class SystemConfig(BaseModel):
    query_type: str | None = "rag"
    query_types: List | None = ["chat_llm"]


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # system
    system: SystemConfig = SystemConfig()

    llms: List[OpenAICompatibleLlmConfig] = []

    chat: ChatConfig = ChatConfig()

    # data_analysis
    data_analysis: Annotated[
        Union[PandasAnalysisConfig, SqliteAnalysisConfig, MysqlAnalysisConfig],
        Field(discriminator="type"),
        BeforeValidator(validate_case_insensitive),
    ]

    # vector_index connection
    index: PaiVectorIndexConfig

    # oss_store
    oss_store: OssStoreConfig

    # postprocessor
    postprocessor: Annotated[
        Union[SimilarityPostProcessorConfig, RerankModelPostProcessorConfig],
        Field(discriminator="reranker_type"),
        BeforeValidator(validate_case_insensitive),
    ]

    # search web
    search: Annotated[
        Union[BingSearchConfig, AliyunSearchConfig, GoogleSearchConfig],
        Field(discriminator="source"),
        BeforeValidator(validate_case_insensitive),
    ]

    # synthesizer
    synthesizer: SynthesizerConfig

    query_rewrite: QueryRewriteConfig = QueryRewriteConfig()

    guardrail: AliyunTextModerationPlusConfig = AliyunTextModerationPlusConfig()

    news_extension: MiaobiNewsConfig = MiaobiNewsConfig()

    trace: TraceConfig = TraceConfig()
