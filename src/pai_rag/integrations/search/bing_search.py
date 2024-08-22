import numpy as np
import requests
from typing import Any, List, Optional, cast
from llama_index.readers.web import ReadabilityWebPageReader
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import BaseSynthesizer
import faiss
import logging

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT_BASE_URL = "https://api.bing.microsoft.com/v7.0/search"
DEFAULT_SEARCH_COUNT = 5
DEFAULT_LANG = "zh-CN"


class BingSearchTool:
    def __init__(
        self,
        api_key: str,
        embed_model: BaseEmbedding = None,
        synthesizer: BaseSynthesizer = None,
        endpoint: str = DEFAULT_ENDPOINT_BASE_URL,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG
    ):
        self.api_key = api_key
        self.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
        self.embed_model = embed_model
        self.embed_dims = len(embed_model.get_text_embedding("0"))
        self.synthesizer = synthesizer

        self.search_count = search_count
        self.search_lang = search_lang

        self.endpoint = endpoint
        self.html_reader = ReadabilityWebPageReader(wait_until="domcontentloaded")
    
    def _search(
        self,
        query: str,
        lang: str = None,
        count: Optional[int] = None,
    ):
        response = requests.get(
            self.endpoint,
            headers={"Ocp-Apim-Subscription-Key":self.api_key},
            params={
                "q": query,
                "mkt": lang or self.search_lang,
                "count": count or self.search_count,
                "responseFilter": "webpages"
            },
        )
        return response.json()

    async def _aload_urls(
        self,
        urls: List[str]
    ):
        if not urls:
            return []

        # 并行可能会有点问题
        docs = []
        for url in urls:
            temp_docs = await self.html_reader.async_load_data(url)
            for doc in temp_docs:
                doc.metadata["file_name"] = url
            docs.extend(temp_docs)
        
        return docs
    
    def _rank_nodes(
        self,
        nodes: List[BaseNode],
        similarity_top_k: int,
        query_embedding: Any,
    ) -> List[NodeWithScore]:
        faiss_index = faiss.IndexFlatIP(self.embed_dims)
        embeddings = self.embed_model.get_text_embedding_batch([node.text for node in nodes])
        for embedding in embeddings:
            text_embedding_np = np.array(embedding, dtype="float32")[np.newaxis, :]
            faiss_index.add(text_embedding_np)
        
        query_embedding = cast(List[float], query_embedding)
        query_embedding_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        dists, indices = faiss_index.search(
            query_embedding_np, similarity_top_k
        )
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
        query: str,
        similarity_top_k: int = 10,
        lang: str = DEFAULT_LANG,
        search_top_k: int = DEFAULT_SEARCH_COUNT,
    ):
        query_embedding = self.embed_model.get_query_embedding(query)

        response_json = self._search(query=query, count=search_top_k, lang=lang)
        urls = [value["url"] for value in response_json["webPages"]["value"]]
        logger.info(f"Get {len(urls)} url links using Bing Search.")

        docs = await self._aload_urls(urls)
        logger.info(f"Get {len(docs)} docs from url.")

        nodes = self.node_parser.get_nodes_from_documents(docs)
        logger.info(f"Parsed {len(docs)} nodes from doc.")

        nodes_result = self._rank_nodes(nodes, similarity_top_k, query_embedding)

        return await self.synthesizer.asynthesize(query=query, nodes=nodes_result)
    
    async def astream_query(
        self,
        query: str,
        similarity_top_k: int = 10,
        lang: str = DEFAULT_LANG,
        search_top_k: int = DEFAULT_SEARCH_COUNT,
    ):
        streaming = self.synthesizer._streaming
        self.synthesizer._streaming = True

        query_embedding = self.embed_model.get_query_embedding(query)

        response_json = self._search(query=query, count=search_top_k, lang=lang)
        urls = [value["url"] for value in response_json["webPages"]["value"]]
        logger.info(f"Get {len(urls)} url links using Bing Search.")

        docs = await self._aload_urls(urls)
        logger.info(f"Get {len(docs)} docs from url.")

        nodes = self.node_parser.get_nodes_from_documents(docs)
        logger.info(f"Parsed {len(docs)} nodes from doc.")

        nodes_result = self._rank_nodes(nodes, similarity_top_k, query_embedding)

        stream_response = await self.synthesizer.asynthesize(query=query, nodes=nodes_result)
        self.synthesizer._streaming = streaming
        
        return stream_response