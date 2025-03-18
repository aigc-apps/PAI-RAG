from typing import Optional
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import QueryBundle
from pai_rag.integrations.search.bs4_reader import ParallelBeautifulSoupWebReader
import httpx
import time
from loguru import logger

from pai_rag.integrations.search.search_config import DEFAULT_SEARCH_COUNT

DEFAULT_ENDPOINT_BASE_URL = "https://api.bing.microsoft.com/v7.0/search"
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

            urls = []
            url2titles = {}
            url2snippets = {}
            url2dates = {}
            for value in response_json["webPages"]["value"]:
                url = value.get("url")
                if url:
                    urls.append(url)
                    url2titles[url] = value.get("name")
                    url2snippets[url] = value.get("snippet")
                    url2dates[url] = value.get("dateLastCrawled")

            logger.info(f"Get {len(urls)} url links using Bing Search.")

            docs = self.html_reader.load_data(urls, include_url_in_text=False)
            for doc in docs:
                if doc.text_resource.text is None or len(doc.text_resource.text) < len(
                    url2snippets[doc.metadata["URL"]]
                ):
                    doc.text_resource.text = url2snippets[doc.metadata["URL"]]
                doc.text_resource.text = doc.text_resource.text[:800]
                doc.metadata["source"] = "web_search"
                doc.metadata["file_url"] = doc.metadata["URL"]
                doc.metadata["file_name"] = url2titles[doc.metadata["URL"]]
                doc.metadata["publish_time"] = url2dates[doc.metadata["URL"]]

            return docs

    async def aquery(
        self,
        query: QueryBundle,
        lang: str = None,
        system_role_str: Optional[str] = None,
        prompt_template_str: Optional[str] = None,
        search_top_k: Optional[int] = None,
    ):
        start = time.time()

        if lang:
            self.search_lang = lang
        if search_top_k:
            self.search_count = search_top_k

        logger.info(f"Bing Search with query {query.query_str}.")
        docs = await self._asearch(
            query=query.query_str,
        )

        nodes = []
        for doc in docs:
            doc_node = TextNode(text=doc.text[:800], metadata=doc.metadata)
            nodes.append(NodeWithScore(node=doc_node, score=1))

        logger.info(
            f"[WebSearch]-Bing Get {len(docs)} docs from url. Elapsed time: {time.time() - start}seconds."
        )

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
