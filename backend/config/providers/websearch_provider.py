import json
from typing import Any, Type

from pydantic import Field
from sqlmodel import SQLModel
from db.encrypt_utils import decrypt_key
from db.models.websearch import WebSearchConfigEntity
from config.providers.base_provider import BaseConfigProvider
from tools.search.aliyun_search_tool import AliyunSearchTool
from llama_index.core.tools import FunctionTool


async def aget_aliyun_search_result(query: str):
    if websearch_provider.searcher is None:
        raise ValueError("搜索尚未配置.")

    res = await websearch_provider.searcher.aquery(query)
    return json.dumps(res, ensure_ascii=False)



class WebSearchProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = WebSearchConfigEntity
    searcher: Any = Field(default=None)

    def _refresh(self, search_entity: WebSearchConfigEntity):
        self.searcher = AliyunSearchTool(
            access_key_id=decrypt_key(
                search_entity.encrypted_access_key_id
            ),
            access_key_secret=decrypt_key(
                search_entity.encrypted_access_key_secret
            ),
            endpoint=search_entity.endpoint,
        )


    def _load_entries(self, entries):
        super()._load_entries(entries)
        if len(entries) > 0:
            self._refresh(entries[0])

    def add(self, entry):
        super().add(entry)
        self._refresh(entry)

    def update(self, entry):
        super().update(entry)
        self._refresh(entry)

    def get_search_tools(self):
        assert len(self.config_map) > 0, "There is no available websearch configs."
        aliyun_search_tool = FunctionTool.from_defaults(
            async_fn=aget_aliyun_search_result,
            name="search-web",
            description="从阿里云搜索引擎中搜索给定查询的最新内容。",
        )
        return [aliyun_search_tool]


websearch_provider = WebSearchProvider()
