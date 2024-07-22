"""AnalyticDB vector store."""

import json
from typing import Any

from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
)
from llama_index.vector_stores.analyticdb import AnalyticDBVectorStore
from llama_index.vector_stores.analyticdb.base import _recursively_parse_adb_filter
from pai_rag.utils.score_utils import normalize_cosine_similarity_score


class MyAnalyticDBVectorStore(AnalyticDBVectorStore):
    """My AnalyticDB vector store.

    In this vector store, embeddings and docs are stored within a
    single table.

    During query time, the index uses AnalyticDB to query for the top
    k most similar nodes.

    Args:
        region_id: str
        instance_id: str
        account: str
        account_password: str
        namespace: str
        namespace_password: str
        embedding_dimension: int
        metrics: str
        collection: str
    """

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the vector store for top k most similar nodes.

        Args:
            query: VectorStoreQuery: the query to execute.

        Returns:
            VectorStoreQueryResult: the result of the query.
        """
        from alibabacloud_gpdb20160503 import models as gpdb_20160503_models

        self._initialize()
        vector = (
            query.query_embedding
            if query.mode in (VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID)
            else None
        )
        content = (
            query.query_str
            if query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID)
            else None
        )
        request = gpdb_20160503_models.QueryCollectionDataRequest(
            dbinstance_id=self.instance_id,
            region_id=self.region_id,
            namespace=self.namespace,
            namespace_password=self.namespace_password,
            collection=self.collection,
            include_values=kwargs.pop("include_values", True),
            metrics=self.metrics,
            vector=vector,
            content=content,
            top_k=query.similarity_top_k,
            filter=_recursively_parse_adb_filter(query.filters),
        )
        response = self._client.query_collection_data(request)
        nodes = []
        similarities = []
        ids = []
        for match in response.body.matches.match:
            node = metadata_dict_to_node(
                json.loads(match.metadata.get("metadata_")),
                match.metadata.get("content"),
            )
            nodes.append(node)
            similarities.append(normalize_cosine_similarity_score(match.score))
            ids.append(match.metadata.get("node_id"))
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
