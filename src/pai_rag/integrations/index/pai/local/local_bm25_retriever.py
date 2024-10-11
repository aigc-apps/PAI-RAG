import logging
from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle
from pai_rag.integrations.index.pai.local.local_bm25_index import LocalBm25IndexStore

logger = logging.getLogger(__name__)


class LocalBM25Retriever(BaseRetriever):
    def __init__(
        self,
        bm25_index: LocalBm25IndexStore,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        self._similarity_top_k = similarity_top_k
        self.bm25_index = bm25_index
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        bm25_index: LocalBm25IndexStore = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
    ) -> "LocalBM25Retriever":
        return cls(
            bm25_index=bm25_index,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def _get_scored_nodes(self, query: str) -> List[NodeWithScore]:
        return self.bm25_index.query(query_str=query, top_n=self._similarity_top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not query_bundle.query_str:
            return []

        scored_nodes = self._get_scored_nodes(query_bundle.query_str)

        # Sort and get top_k nodes, score range => 0..1, closer to 1 means more relevant
        nodes = sorted(scored_nodes, key=lambda x: x.score or 0.0, reverse=True)
        return nodes[: self._similarity_top_k]
