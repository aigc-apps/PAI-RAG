import logging
from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.keyword_table.utils import simple_extract_keywords
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle
from nltk.stem import PorterStemmer
from pai_rag.modules.index.pai_bm25_index import PaiBm25Index

logger = logging.getLogger(__name__)


def tokenize_remove_stopwords(text: str) -> List[str]:
    # lowercase and stem words
    text = text.lower()
    stemmer = PorterStemmer()
    words = list(simple_extract_keywords(text))
    return [stemmer.stem(word) for word in words]


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        bm25_index: PaiBm25Index,
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
        bm25_index: PaiBm25Index = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
    ) -> "BM25Retriever":
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
