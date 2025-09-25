import json
from typing import Any, Type

from pydantic import Field
from sqlmodel import SQLModel
from db.encrypt_utils import decrypt_key
from db.models.websearch import WebSearchConfigEntity
from config.providers.base_provider import BaseConfigProvider
from tools.search.aliyun_search_tool import AliyunSearchTool
from llama_index.core.tools import FunctionTool
from chat.tools.tavily_search import TavilySearchTool
import traceback
from loguru import logger


async def aget_search_result(query: str):
    if websearch_provider.searcher is None:
        raise ValueError("搜索尚未配置.")

    res = await websearch_provider.searcher.aquery(query)
    return json.dumps(res, ensure_ascii=False)



class WebSearchProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = WebSearchConfigEntity
    searcher: Any = Field(default=None)
    searcher_type: str = Field(default=None)

    def _refresh(self, search_entity: WebSearchConfigEntity):
        try:
            if search_entity.type == "aliyun":
                self.searcher_type = search_entity.type
                self.searcher = AliyunSearchTool(
                    access_key_id=decrypt_key(
                        search_entity.encrypted_access_key_id
                    ),
                    access_key_secret=decrypt_key(
                        search_entity.encrypted_access_key_secret
                    ),
                    endpoint=search_entity.endpoint,
                    search_count=search_entity.search_count,
                )
            elif search_entity.type == "tavily":
                self.searcher_type = search_entity.type

                self.searcher = TavilySearchTool(
                    api_key=decrypt_key(search_entity.encrypted_tavily_api_key),
                    search_count=search_entity.search_count,
                )
            else:
                self.searcher = None
        except Exception:
            logger.error(f"创建搜索引擎失败: {traceback.format_exc()}")
            self.searcher = None


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

        if self.searcher_type == "tavily":
            search_tool = FunctionTool.from_defaults(
                async_fn=aget_search_result,
                name="tavily-websearch",
                description="从 Tavily 搜索引擎中搜索给定查询的最新内容。",
            )
        else:
            search_tool = FunctionTool.from_defaults(
                async_fn=aget_search_result,
                name="aliyun-websearch",
                description="从阿里云搜索引擎中搜索给定查询的最新内容。",
            )
        return [search_tool]


websearch_provider = WebSearchProvider()
