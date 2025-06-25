from pydantic import BaseModel, ConfigDict
from pairag.integrations.llms.pai.llm_config import OpenAICompatibleLlmConfig
from pairag.integrations.synthesizer.prompt_templates import (
    DEFAULT_SYSTEM_ROLE_TEMPLATE,
)
from pairag.utils.prompt_template import (
    CHAT_LLM_REWRITE_PROMPT_ZH,
    KNOWLEDGEBASE_REWRITE_PROMPT_ZH,
    NEWS_REWRITE_PROMPT_ZH,
    NL2SQL_REWRITE_PROMPT_ZH,
    REWRITE_PROMPT_ROLE_ZH,
    WEBSEARCH_REWRITE_PROMPT_ZH,
)
from pairag.utils.prompt_template import (
    DEFALT_LLM_CHAT_PROMPT_TEMPL,
)


class ChatConfig(BaseModel):
    model_id: str | None = None
    model_config = ConfigDict(coerce_numbers_to_str=True)


class QueryRewriteConfig(BaseModel):
    enabled: bool = True
    base_prompt_template_str: str = REWRITE_PROMPT_ROLE_ZH
    llm_tool_prompt_str: str = CHAT_LLM_REWRITE_PROMPT_ZH
    knowledge_tool_prompt_str: str = KNOWLEDGEBASE_REWRITE_PROMPT_ZH
    websearch_tool_prompt_str: str = WEBSEARCH_REWRITE_PROMPT_ZH
    db_tool_prompt_str: str = NL2SQL_REWRITE_PROMPT_ZH
    news_tool_prompt_str: str = NEWS_REWRITE_PROMPT_ZH

    model_id: str | None = None
    llm: OpenAICompatibleLlmConfig | None = OpenAICompatibleLlmConfig()
    model_config = ConfigDict(coerce_numbers_to_str=True)


class NodeEnhancementConfig(BaseModel):
    tree_depth: int = 3
    max_clusters: int = 52
    proba_threshold: float = 0.10


class OssStoreConfig(BaseModel):
    bucket: str | None = None
    endpoint: str = "oss-cn-hangzhou.aliyuncs.com"
    ak: str | None = None
    sk: str | None = None
    model_config = ConfigDict(coerce_numbers_to_str=True)


class SynthesizerConfig(BaseModel):
    use_multimodal_llm: bool = False
    system_role_template: str = DEFAULT_SYSTEM_ROLE_TEMPLATE
    custom_prompt_template: str = DEFALT_LLM_CHAT_PROMPT_TEMPL
