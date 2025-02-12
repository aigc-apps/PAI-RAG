import asyncio
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import QueryBundle
from loguru import logger

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_iqs20241111 import models
from alibabacloud_iqs20241111.client import Client

from pai_rag.integrations.search.bs4_reader import ParallelBeautifulSoupWebReader


DEFAULT_ENDPOINT_BASE_URL = "iqs.cn-zhangjiakou.aliyuncs.com"
DEFAULT_SEARCH_COUNT = 30
DEFAULT_LANG = "zh-CN"
DEFAULT_TIMERANGE = "OneMonth"  # OneMonth, OneWeek, OneDay, OneYear, NoLimit


class AliyunSearchTool(BaseQueryEngine):
    def __init__(
        self,
        accessid: str,
        accesskey: str,
        synthesizer: BaseSynthesizer = None,
        endpoint: str = DEFAULT_ENDPOINT_BASE_URL,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG,
        time_range: str = DEFAULT_TIMERANGE,
    ):
        self.accessid = accessid
        self.accesskey = accesskey

        config = open_api_models.Config(
            access_key_id=accessid,
            access_key_secret=accessid,
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
        logger.info(f"Finished searching query {request}. {response.json()}")
        return response.body.to_map()

    async def _asearch(
        self,
        query: str,
    ):
        search_tasks = []
        for i in range(0, 1 + int(self.search_count / 10), 1):
            search_tasks.append(
                self._search_aliyun_single_page(query=query, page=i + 1)
            )

        search_results = await asyncio.gather(*search_tasks)

        nodes = []
        for result in search_results:
            items = result["pageItems"]
            for item in items:
                score = 0.1
                node = TextNode(
                    text=item["mainText"][:800],
                    metadata={"file_url": item["link"], "file_name": item["title"]},
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
    ):
        nodes = await self.asearch(query=query.query_str)
        logger.info(f"Get {len(nodes)} docs from url.")

        return await self.synthesizer.asynthesize(query=query, nodes=nodes)

    def _get_prompt_modules(self):
        raise NotImplementedError

    def _query(self, query_bundle):
        raise NotImplementedError

    async def _aquery(self, query_bundle):
        raise NotImplementedError
