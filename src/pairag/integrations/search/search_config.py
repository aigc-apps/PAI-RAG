from pydantic import BaseModel
from enum import Enum
from typing import Literal

DEFAULT_ALIYUN_SEARCH_ENDPOINT = "iqs.cn-zhangjiakou.aliyuncs.com"
DEFAULT_GOOGLE_SEARCH_ENDPOINT = "https://serpapi.com/search"
DEFAULT_SEARCH_COUNT = 10
DEFAULT_SEARCH_QA_PROMPT_TEMPLATE = """
你的目标是根据搜索结果提供准确、有用且易于理解的信息。
# 任务要求：
- 请严格根据提供的参考内容回答问题，并非所有参考内容都与用户的问题密切相关，你需要结合问题，对参考内容进行甄别、筛选。仅参考与问题相关的内容并忽略所有不相关的信息。
- 如果参考内容中没有相关信息或与问题无关，请基于你的已有知识进行回答。
- 确保答案准确、简洁，并且使用与用户提问相同的语种。
- 在回答过程中，请避免使用“从参考内容得出”、“从材料得出”、“根据参考内容”等措辞。
- 保持回答的专业性和友好性。
- 如果需要更多信息来更好地回答问题，请礼貌地询问。
- 对于复杂的问题，尽量简化解释，使信息易于理解。如果回答很长，请尽量结构化、分段落总结。如果需要分点作答，尽量控制在5个点以内，并合并相关的内容。
- 对于客观类的问答，如果问题的答案非常简短，可以适当补充一到两句相关信息，以丰富内容。
- 除非用户要求，否则请保持输出语种与用户输入问题语种的一致性。
- 对于涉及不安全/不道德/敏感/色情/暴力/赌博/违法等行为的问题，请明确拒绝提供所要求的信息，并简单解释为什么这样的请求不能被满足。
- 你知道今天的日期是{current_datetime}，但你不会主动在回复开头提到日期信息。

# 以下内容是基于用户发送的消息的搜索/查询结果:
{context_str}

# 以下内容是用户问答历史记录:
{history_str}

# 以下内容是用户消息:
{query_str}
"""


class SupportedSearchType(str, Enum):
    bing = "bing"
    aliyun = "aliyun"
    google = "google"


class BaseSearchConfig(BaseModel):
    source: SupportedSearchType
    search_count: int = DEFAULT_SEARCH_COUNT
    search_qa_prompt_template: str = DEFAULT_SEARCH_QA_PROMPT_TEMPLATE

    class Config:
        frozen = True

    @classmethod
    def get_subclasses(cls):
        return tuple(cls.__subclasses__())

    @classmethod
    def get_type(cls):
        return cls.model_fields["source"].default


class BingSearchConfig(BaseSearchConfig):
    source: Literal[SupportedSearchType.bing] = SupportedSearchType.bing
    search_api_key: str | None = None
    search_lang: str = "zh-CN"


class AliyunSearchConfig(BaseSearchConfig):
    source: Literal[SupportedSearchType.aliyun] = SupportedSearchType.aliyun
    endpoint: str = DEFAULT_ALIYUN_SEARCH_ENDPOINT
    access_key_id: str | None = None
    access_key_secret: str | None = None


class GoogleSearchConfig(BaseSearchConfig):
    source: Literal[SupportedSearchType.google] = SupportedSearchType.google
    serpapi_key: str | None = None
    search_lang: str = "zh-CN"
