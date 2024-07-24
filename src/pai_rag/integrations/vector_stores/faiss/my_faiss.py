"""Faiss Vector store index.

An index that is built on top of an existing vector store.

"""

import logging
import os
from typing import Any, List, cast

import numpy as np
from llama_index.core.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.core.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from pai_rag.utils.score_utils import normalize_cosine_similarity_score

logger = logging.getLogger()

DEFAULT_PERSIST_PATH = os.path.join(
    DEFAULT_PERSIST_DIR, f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
)


class MyFaissVectorStore(FaissVectorStore):
    """My Faiss Vector Store.

    Embeddings are stored within a Faiss index.

    During query time, the index uses Faiss to query for the top
    k embeddings, and returns the corresponding indices.

    Args:
        faiss_index (faiss.Index): Faiss index instance

    """

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for Faiss yet.")

        query_embedding = cast(List[float], query.query_embedding)
        query_embedding_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        dists, indices = self._faiss_index.search(
            query_embedding_np, query.similarity_top_k
        )
        dists = list(dists[0])
        # if empty, then return an empty response
        if len(indices) == 0:
            return VectorStoreQueryResult(similarities=[], ids=[])

        # returned dimension is 1 x k
        node_idxs = indices[0]

        filtered_dists = []
        filtered_node_idxs = []
        for dist, idx in zip(dists, node_idxs):
            if idx < 0:
                continue
            # Ours: normalize the cosine sim score to [0,1]
            filtered_dists.append(normalize_cosine_similarity_score(dist))
            filtered_node_idxs.append(str(idx))

        return VectorStoreQueryResult(
            similarities=filtered_dists, ids=filtered_node_idxs
        )
