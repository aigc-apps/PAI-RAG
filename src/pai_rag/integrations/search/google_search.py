from typing import Optional
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import QueryBundle
from pai_rag.app.api.models import PaiQueryBundle
from pai_rag.integrations.search.bs4_reader import ParallelBeautifulSoupWebReader
import time
from loguru import logger

from pai_rag.integrations.search.search_config import DEFAULT_SEARCH_COUNT
import serpapi

DEFAULT_LANG = "zh-CN"


class GoogleSearchTool(BaseQueryEngine):
    def __init__(
        self,
        api_key: str,
        synthesizer: BaseSynthesizer = None,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG,
    ):
        self.api_key = api_key
        self.synthesizer = synthesizer

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

        logger.info(f"Google Search with query {query.query_str}.")
        docs = await self._asearch(
            query=query.query_str,
        )

        nodes = []
        for doc in docs:
            doc_node = TextNode(text=doc.text[:800], metadata=doc.metadata)
            nodes.append(NodeWithScore(node=doc_node, score=1))

        logger.info(
            f"[WebSearch]-Google Get {len(docs)} docs from url. Elapsed time: {time.time() - start} seconds."
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
