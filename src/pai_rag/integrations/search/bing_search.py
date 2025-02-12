from typing import Optional
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import QueryBundle
import httpx
from loguru import logger

from pai_rag.integrations.search.bs4_reader import ParallelBeautifulSoupWebReader


DEFAULT_ENDPOINT_BASE_URL = "https://api.bing.microsoft.com/v7.0/search"
DEFAULT_SEARCH_COUNT = 30
DEFAULT_LANG = "zh-CN"


class BingSearchTool(BaseQueryEngine):
    def __init__(
        self,
        api_key: str,
        synthesizer: BaseSynthesizer = None,
        endpoint: str = DEFAULT_ENDPOINT_BASE_URL,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG,
    ):
        self.api_key = api_key
        self.synthesizer = synthesizer

        self.search_count = search_count
        self.search_lang = search_lang

        self.endpoint = endpoint
        self.html_reader = ParallelBeautifulSoupWebReader()

    async def _asearch(
        self,
        query: str,
    ):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.endpoint,
                headers={"Ocp-Apim-Subscription-Key": self.api_key},
                params={
                    "q": query,
                    "mkt": self.search_lang,
                    "count": self.search_count,
                    "responseFilter": "webpages",
                },
                timeout=5,
            )
            response_json = response.json()
            if "webPages" not in response_json:
                logger.warning(f"Bing Search API response: {response_json}")
                return []
            titles = [value["name"] for value in response_json["webPages"]["value"]]
            urls = [value["url"] for value in response_json["webPages"]["value"]]
            url2titles = dict(zip(urls, titles))
            logger.info(f"Get {len(urls)} url links using Bing Search.")

            docs = self.html_reader.load_data(urls, include_url_in_text=False)
            for doc in docs:
                doc.metadata["file_url"] = doc.metadata["URL"]
                doc.metadata["file_name"] = url2titles[doc.metadata["URL"]]

            return docs

    async def aquery(
        self,
        query: QueryBundle,
        lang: str = None,
        search_top_k: Optional[int] = None,
    ):
        if lang:
            self.search_lang = lang
        if search_top_k:
            self.search_count = search_top_k

        docs = await self._asearch(query=query.query_str)
        logger.info(f"Get {len(docs)} docs from url.")

        nodes = []
        for doc in docs:
            doc_node = TextNode(text=doc.text[:800], metadata=doc.metadata)
            nodes.append(NodeWithScore(node=doc_node, score=1))

        return await self.synthesizer.asynthesize(query=query, nodes=nodes)

    def _get_prompt_modules(self):
        raise NotImplementedError

    def _query(self, query_bundle):
        raise NotImplementedError

    async def _aquery(self, query_bundle):
        raise NotImplementedError
