from typing import Annotated, Dict, Union, List
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator
from pairag.core.models.config import (
    AliyunTextModerationPlusConfig,
    OssStoreConfig,
    QueryRewriteConfig,
    SynthesizerConfig,
    ChatConfig,
)
from pairag.extensions.news.news_config import MiaobiNewsConfig
from pairag.integrations.data_analysis.data_analysis_config import (
    MysqlAnalysisConfig,
    PandasAnalysisConfig,
    SqliteAnalysisConfig,
)
from pairag.integrations.llms.pai.llm_config import (
    PaiBaseLlmConfig,
)
from pairag.integrations.postprocessor.pai.pai_postprocessor import (
    RerankModelPostProcessorConfig,
    SimilarityPostProcessorConfig,
)
from pairag.integrations.search.search_config import (
    BingSearchConfig,
    QuarkSearchConfig,
    AliyunSearchConfig,
    GoogleSearchConfig,
)
from pairag.integrations.trace.trace_config import TraceConfig


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
    query_type: str | None = "rag"
    query_types: List | None = ["chat_llm"]


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # system
    system: SystemConfig = SystemConfig()

    llms: Annotated[
        List[Union[PaiBaseLlmConfig.get_subclasses()]],
        Field(default_factory=list),
        BeforeValidator(lambda x: [validate_case_insensitive(item) for item in x]),
    ]

    chat: ChatConfig = ChatConfig()

    # data_analysis
    data_analysis: Annotated[
        Union[PandasAnalysisConfig, SqliteAnalysisConfig, MysqlAnalysisConfig],
        Field(discriminator="type"),
        BeforeValidator(validate_case_insensitive),
    ]

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
