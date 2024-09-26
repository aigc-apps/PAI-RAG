import numpy as np
import requests
from typing import Any, List, Optional, cast
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import QueryBundle
import faiss
import logging

from pai_rag.integrations.search.bs4_reader import ParallelBeautifulSoupWebReader

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT_BASE_URL = "https://api.bing.microsoft.com/v7.0/search"
DEFAULT_SEARCH_COUNT = 10
DEFAULT_LANG = "zh-CN"


class BingSearchTool:
    def __init__(
        self,
        api_key: str,
        embed_model: BaseEmbedding = None,
        synthesizer: BaseSynthesizer = None,
        endpoint: str = DEFAULT_ENDPOINT_BASE_URL,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG,
    ):
        self.api_key = api_key
        self.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
        self.embed_model = embed_model
        self.embed_dims = len(embed_model.get_text_embedding("0"))
        self.synthesizer = synthesizer

        self.search_count = search_count
        self.search_lang = search_lang

        self.endpoint = endpoint
        self.html_reader = ParallelBeautifulSoupWebReader()

    def _search(
        self,
        query: str,
    ):
        response = requests.get(
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
        titles = [value["name"] for value in response_json["webPages"]["value"]]
        urls = [value["url"] for value in response_json["webPages"]["value"]]
        url2titles = dict(zip(urls, titles))
        logger.info(f"Get {len(urls)} url links using Bing Search.")

        docs = self.html_reader.load_data(urls, include_url_in_text=False)
        for doc in docs:
            doc.metadata["file_url"] = doc.metadata["URL"]
            doc.metadata["file_name"] = url2titles[doc.metadata["URL"]]

        return docs

    def _rank_nodes(
        self,
        nodes: List[BaseNode],
        query_embedding: Any,
    ) -> List[NodeWithScore]:
        faiss_index = faiss.IndexFlatIP(self.embed_dims)
        logger.info("Calculate doc embedding.")
        embeddings = self.embed_model.get_text_embedding_batch(
            [node.text for node in nodes]
        )
        logger.info("Insert to index.")
        for embedding in embeddings:
            text_embedding_np = np.array(embedding, dtype="float32")[np.newaxis, :]
            faiss_index.add(text_embedding_np)

        logger.info("Start search.")
        query_embedding = cast(List[float], query_embedding)
        query_embedding_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        dists, indices = faiss_index.search(query_embedding_np, self.search_count)
        faiss_index.reset()
        dists = list(dists[0])
        # if empty, then return an empty response
        if len(indices) == 0:
            return "Empty Response"

        # returned dimension is 1 x k
        node_idxs = indices[0]

        nodes_result = []
        for dist, idx in zip(dists, node_idxs):
            if idx < 0:
                continue
            nodes_result.append(NodeWithScore(node=nodes[idx], score=dist))
        return nodes_result

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

        query_embedding = self.embed_model.get_query_embedding(query.query_str)

        docs = self._search(query=query.query_str)
        logger.info(f"Get {len(docs)} docs from url.")

        nodes = self.node_parser.get_nodes_from_documents(docs)
        logger.info(f"Parsed {len(docs)} nodes from doc.")

        nodes_result = self._rank_nodes(nodes, query_embedding)
        logger.info(f"Searched {len(nodes_result)} nodes from web pages.")

        return await self.synthesizer.asynthesize(query=query, nodes=nodes_result)
