from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle
from pairag.integrations.search.bs4_reader import ParallelBeautifulSoupWebReader
import time
from loguru import logger

from pairag.integrations.search.search_config import (
    DEFAULT_SEARCH_COUNT,
)
import serpapi


DEFAULT_LANG = "zh-CN"


class GoogleSearchTool(BaseRetriever):
    def __init__(
        self,
        api_key: str,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG,
    ):
        self.api_key = api_key

        self.search_count = search_count
        self.search_lang = search_lang

        self.html_reader = ParallelBeautifulSoupWebReader()

    async def _asearch(
        self,
        query: str,
    ):
        params = {
            "engine": "google",
            "q": query,
            "hl": self.search_lang,
            "num": self.search_count,
            "api_key": self.api_key,
        }

        search = serpapi.search(params)
        results = search.as_dict()

        if "organic_results" not in results:
            logger.warning(f"Google Search API response: {results}")
            return []

        urls = []
        url2titles = {}
        url2snippets = {}
        url2dates = {}
        url2sources = {}
        url2favicons = {}

        for result in results["organic_results"]:
            url = result.get("link")
            if url:
                urls.append(url)
                url2titles[url] = result.get("title")
                url2snippets[url] = result.get("snippet")
                url2dates[url] = result.get("date")
                url2sources[url] = result.get("source")
                url2favicons[url] = result.get("favicon")

        logger.info(f"Get {len(urls)} url links using Google Search.")

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
            doc.metadata["host_name"] = url2sources[doc.metadata["URL"]]
            doc.metadata["host_logo"] = url2favicons[doc.metadata["URL"]]
            doc.metadata["publish_time"] = url2dates[doc.metadata["URL"]]

        return docs

    async def _aretrieve(
        self,
        query_bundle: QueryBundle,
    ):
        start = time.time()

        logger.info(f"Google Search with query {query_bundle.query_str}.")
        docs = await self._asearch(
            query=query_bundle.query_str,
        )

        nodes = []
        for doc in docs:
            doc_node = TextNode(text=doc.text[:800], metadata=doc.metadata)
            nodes.append(NodeWithScore(node=doc_node, score=1))

        logger.info(
            f"[WebSearch]-Google Get {len(docs)} docs from url. Elapsed time: {time.time() - start} seconds."
        )

        return nodes

    def _retrieve(self, query: QueryBundle):
        raise NotImplementedError
