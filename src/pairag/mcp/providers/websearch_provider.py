import os
from pairag.db.encrypt_utils import decrypt_key
from pairag.mcp.providers.base_provider import BaseConfigProvider
from pairag.mcp.tools.search.aliyun_search_tool import aget_aliyun_search_result
from llama_index.core.tools import FunctionTool


class WebSearchProvider(BaseConfigProvider):
    def _load_entries(self, entries):
        super()._load_entries(entries)
        if len(entries) > 0:
            search_entity = entries[0]
            os.environ["WEBSEARCH_ACCESS_KEY_ID"] = decrypt_key(
                search_entity.encrypted_access_key_id
            )
            os.environ["WEBSEARCH_ACCESS_KEY_SECRET"] = decrypt_key(
                search_entity.encrypted_access_key_secret
            )

    def get_search_tools(self):
        assert len(self.config_map) > 0, "There is no available websearch configs."
        aliyun_search_tool = FunctionTool.from_defaults(
            async_fn=aget_aliyun_search_result,
            name="search-web",
            description="从阿里云搜索引擎中搜索给定查询的最新内容。",
        )
        return [aliyun_search_tool]


websearch_provider = WebSearchProvider()
