from pydantic import BaseModel
from enum import Enum
from typing import Literal

DEFAULT_ALIYUN_SEARCH_ENDPOINT = "iqs.cn-zhangjiakou.aliyuncs.com"
DEFAULT_QUARK_SEARCH_ENDPOINT = "https://zx-dsc.sm.cn/"
DEFAULT_SEARCH_COUNT = 10


class SupportedSearchType(str, Enum):
    bing = "bing"
    quark = "quark"
    aliyun = "aliyun"


class BaseSearchConfig(BaseModel):
    source: SupportedSearchType
    search_count: int = DEFAULT_SEARCH_COUNT

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


class QuarkSearchConfig(BaseSearchConfig):
    source: Literal[SupportedSearchType.quark] = SupportedSearchType.quark
    host: str = DEFAULT_QUARK_SEARCH_ENDPOINT
    user: str | None = None
    secret: str | None = None


class AliyunSearchConfig(BaseSearchConfig):
    source: Literal[SupportedSearchType.aliyun] = SupportedSearchType.aliyun
    endpoint: str = DEFAULT_ALIYUN_SEARCH_ENDPOINT
    access_key_id: str | None = None
    access_key_secret: str | None = None
