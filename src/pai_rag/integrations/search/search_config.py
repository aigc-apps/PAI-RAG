from pydantic import BaseModel
from enum import Enum
from typing import Literal


class SupportedSearchType(str, Enum):
    bing = "bing"
    quark = "quark"


class BaseSearchConfig(BaseModel):
    source: SupportedSearchType
    search_count: int = 30

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
    host: str = "https://zx-dsc.sm.cn/"
    user: str | None = None
    secret: str | None = None
