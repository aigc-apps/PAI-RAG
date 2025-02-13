import asyncio
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import QueryBundle
from loguru import logger

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_iqs20241111 import models
from alibabacloud_iqs20241111.client import Client

from pai_rag.app.api.models import PaiQueryBundle
from pai_rag.integrations.search.bing_search import DEFAULT_SEARCH_COUNT
from pai_rag.integrations.search.bs4_reader import ParallelBeautifulSoupWebReader
from pai_rag.integrations.search.search_config import DEFAULT_ALIYUN_SEARCH_ENDPOINT


DEFAULT_LANG = "zh-CN"
DEFAULT_TIMERANGE = "OneMonth"  # OneMonth, OneWeek, OneDay, OneYear, NoLimit


class AliyunSearchTool(BaseQueryEngine):
    def __init__(
        self,
        accessid: str,
        accesskey: str,
        synthesizer: BaseSynthesizer = None,
        endpoint: str = DEFAULT_ALIYUN_SEARCH_ENDPOINT,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG,
        time_range: str = DEFAULT_TIMERANGE,
    ):
        self.accessid = accessid
        self.accesskey = accesskey

        config = open_api_models.Config(
            access_key_id=accessid,
            access_key_secret=accesskey,
        )
        self.synthesizer = synthesizer

        self.search_count = search_count
        self.search_lang = search_lang

        config.endpoint = endpoint
        self.Client = Client(config)
        self.time_range = time_range
        self.html_reader = ParallelBeautifulSoupWebReader()

    async def _search_aliyun_single_page(self, query: str, page: int = 1):
        request = models.GenericSearchRequest(
            query=query,
            time_range=self.time_range,
            page=page,
        )
        response = await self.Client.generic_search_async(request)
        if response.status_code != 200:
            logger.warning(
                f"Aliyun Search API failed, status code {response.status_code}, detail {response}"
            )
            return []
        logger.info(f"Finished searching query {request}.")
        return response.body.to_map()

    async def _asearch(
        self,
        query: str,
    ):
        search_tasks = []
        for i in range(0, 1 + int((self.search_count - 1) / 10), 1):
            search_tasks.append(
                self._search_aliyun_single_page(query=query, page=i + 1)
            )

        search_results = await asyncio.gather(*search_tasks)

        nodes = []
        for result in search_results:
            items = result["pageItems"]
            for item in items:
                text = item.get("mainText") or item.get("markdownText")
                if not text:
                    continue

                score = 0.1
                node = TextNode(
                    text=text[:800],
                    metadata={
                        "file_url": item["link"],
                        "file_name": item.get("htmlTitle") or item.get("title"),
                    },
                )
                if item.get("publishTime"):
                    node.metadata["publish_time"] = item["publishTime"]
                if item.get("source"):
                    node.metadata["source"] = item["source"]
                if item.get("score"):
                    score = item["score"]
                nodes.append(NodeWithScore(node=node, score=score))
                if len(nodes) >= self.search_count:
                    break
        return nodes

    async def aquery(
        self,
        query: QueryBundle,
        system_role_str: str = None,
        prompt_template_str: str = None,
    ):
        if not query.need_web_search:
            no_search_query = PaiQueryBundle(
                query_str=query.query_str,
                no_retrieval=True,
                stream=query.stream,
                chat_messages_str=query.chat_messages_str,
            )
            return await self.synthesizer.asynthesize(
                query=no_search_query,
                nodes=[],
                system_role_str=system_role_str,
                prompt_template_str=prompt_template_str,
            )

        logger.info(f"Search with query {query.query_str,}.")
        nodes = await self._asearch(query=query.query_str)
        logger.info(f"Get {len(nodes)} docs from url.")

        return await self.synthesizer.asynthesize(
            query=query,
            nodes=nodes,
            system_role_str=system_role_str,
            prompt_template_str=prompt_template_str,
        )

    def _get_prompt_modules(self):
        raise NotImplementedError

    def _query(self, query_bundle):
        raise NotImplementedError

    async def _aquery(self, query_bundle):
        raise NotImplementedError
