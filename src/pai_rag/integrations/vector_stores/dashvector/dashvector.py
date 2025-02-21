"""DashVector vector store index.

An index that is built within DashVector.

"""

import math
import importlib
from typing import Any, List, Optional

import numpy as np
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from loguru import logger

from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BaseSparseEmbeddingFunction,
    get_default_sparse_embedding_function,
)

try:
    from dashtext import combine_dense_and_sparse
except Exception:
    combine_dense_and_sparse = None

from dashvector import Client, Collection, Doc

DEFAULT_BATCH_SIZE = 100
DEFAULT_NODE_ID_KEY = "id"


def _to_dashvector_filter(
    standard_filters: Optional[MetadataFilters] = None,
) -> Optional[str]:
    """Convert from standard filter to dashvector filter dict."""
    if standard_filters is None:
        return None

    filters = []
    for filter in standard_filters.legacy_filters():
        if isinstance(filter.value, str):
            value = f"'{filter.value}'"
        else:
            value = f"{filter.value}"
        filters.append(f"{filter.key} = {value}")
    return " and ".join(filters)


class DashVectorVectorStore(BasePydanticVectorStore):
    stores_text: bool = True  # if false, a local FAISS will be created

    endpoint: str
    api_key: str
    collection_name: str
    partition_name: Optional[str]
    dim: Optional[int] = None
    overwrite: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_sparse: bool = False
    sparse_embedding_function: Optional[BaseSparseEmbeddingFunction] = None

    _client: Client = PrivateAttr()
    _collection: Collection = PrivateAttr()

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        collection_name: str,
        partition_name: Optional[str] = None,
        dim: Optional[int] = None,
        overwrite: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_sparse: bool = False,
        sparse_embedding_function: Optional[BaseSparseEmbeddingFunction] = None,
        **client_kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            endpoint=endpoint,
            api_key=api_key,
            collection_name=collection_name,
            partition_name=partition_name,
            dim=dim,
            overwrite=overwrite,
            enable_sparse=enable_sparse,
            batch_size=batch_size,
            sparse_embedding_function=sparse_embedding_function,
        )

        if enable_sparse:
            if importlib.util.find_spec("dashtext") is None:
                raise ImportError(
                    "`dashtext` package not found, please run `pip install dashtext`"
                )

        self._client = Client(
            api_key=api_key,
            endpoint=endpoint,
            **client_kwargs,  # pass additional arguments such as timeout
        )

        list_collection_response = self._client.list()
        if not list_collection_response:
            raise Exception(f"Failed to list collections: {list_collection_response}")

        # Delete previous collection if overwriting
        if overwrite and collection_name in list_collection_response:
            resp = self._client.delete(collection_name)
            if not resp:
                raise Exception(f"Failed to drop collection: {resp}")
            logger.debug(f"Drop collection (overwrite=True): {self.collection_name}")

        # Create the collection if 1. it is overwritten 2.it does not exist
        if overwrite or collection_name not in list_collection_response:
            if dim is None:
                raise ValueError("Dim argument required for collection creation.")

            # by default, `cosine` metric is preferred, ref:
            # https://help.aliyun.com/document_detail/2584947.html?#af97dd4068v0t
            # however, hybrid search only support dotproduct metric
            resp = self._client.create(
                name=collection_name,
                dimension=dim,
                metric=self._metric,
                fields_schema={},
            )
            if not resp:
                raise Exception(f"Failed to create collection: {resp}")
            logger.debug(
                f"Successfully created a new collection: {self.collection_name}"
            )

        self._collection = self._client.get(name=collection_name)
        if self._collection is None:
            raise Exception(f"Failed to get collection: {self.collection_name}")

        if partition_name is not None:
            list_partition_response = self._collection.list_partitions()
            if not list_partition_response:
                raise Exception(f"Failed to list partitions: {list_partition_response}")
            if partition_name not in list_partition_response:
                resp = self._collection.create_partition(partition_name)
                if not resp:
                    raise Exception(f"Failed to create partition: {resp}")
                logger.debug(
                    f"Successfully created a new partition: {self.collection_name} partition: {self.partition_name}"
                )

        if self.enable_sparse:
            if sparse_embedding_function is None:
                logger.warning(
                    "Sparse embedding function is not provided, using default."
                )
                self.sparse_embedding_function = get_default_sparse_embedding_function()
            else:
                self.sparse_embedding_function = sparse_embedding_function

        logger.debug(
            f"Successfully init DashVector vector store for collection: {self.collection_name}"
        )

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add the embeddings and their nodes into DashVector.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings
                to insert.

        Raises:
            Exception: Failed to insert docs.

        Returns:
            List[str]: List of ids inserted.
        """
        logger.debug(f"Adding {len(nodes)} nodes to DashVector")

        # Although batch is already handled by _add_nodes_to_index() of llama_index.VectorStoreIndex,
        # limit batch_size here to avoid too large message inside DV
        for batch in iter_batch(nodes, self.batch_size):
            docs = [
                Doc(
                    id=node.node_id,
                    vector=node.embedding,
                    sparse_vector=self._gen_sparse_vector_and_convert(
                        node.text, is_document=True
                    ),
                    fields=node_to_metadata_dict(
                        node, remove_text=False, flat_metadata=True
                    ),
                )
                for node in batch
            ]

            resp = self._collection.upsert(docs, partition=self.partition_name)
            if not resp:
                raise Exception(f"Failed to insert docs, error: {resp}")

        logger.debug(
            f"Successfully {len(nodes)} nodes into: {self.collection_name} partition: {self.partition_name}"
        )
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        Raises:
            Exception: Failed to delete the doc.
        """
        # in the future, we can use delete_by_filter function here
        filter = f"{DEFAULT_DOC_ID_KEY}='{ref_doc_id}'"
        resp = self._collection.query(filter=filter)
        if not resp:
            raise Exception(f"Failed to delete doc by {filter}")

        if len(resp) > 0:
            self._collection.delete(ids=[doc.id for doc in resp])
        logger.debug(f"Successfully delete {len(resp)} doc by `{filter}`")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            embedding_field (Optional[str]): name of embedding field
        """

        logger.debug(f"Querying DashVector for query: {query}")

        # According to retrieval_type_to_search_mode(), there are only 3 search modes:
        # VectorStoreQueryMode.TEXT_SEARCH(for keyword), VectorStoreQueryMode.HYBRID, and VectorStoreQueryMode.DEFAULT
        # According to PaiMultiModalVectorIndexRetriever._text_retrieve(), `query` is constructed in _build_vector_store_query(),
        # and DEFAULT is used for image retrieval
        if query.mode == VectorStoreQueryMode.DEFAULT:
            pass
        elif (
            query.mode == VectorStoreQueryMode.HYBRID
            or query.mode == VectorStoreQueryMode.TEXT_SEARCH
        ):
            if self.enable_sparse is False:
                raise ValueError("QueryMode is HYBRID, but enable_sparse is False.")
        else:
            raise ValueError(f"DashVector does not support {query.mode} yet.")

        # Parse the filter
        filter_expr_list = []

        if query.filters is not None:
            filter_expr_list.append(_to_dashvector_filter(query.filters))

        # Parse any docs we are filtering on
        if query.doc_ids is not None and len(query.doc_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.doc_ids]
            filter_expr_list.append(f"{DEFAULT_DOC_ID_KEY} in [{','.join(expr_list)}]")

        # Parse any nodes we are filtering on
        if query.node_ids is not None and len(query.node_ids) != 0:
            expr_list = ['"' + entry + '"' for entry in query.node_ids]
            filter_expr_list.append(f"{DEFAULT_NODE_ID_KEY} in [{','.join(expr_list)}]")

        filter_expr = ""
        if len(filter_expr_list) != 0:
            filter_expr = " and ".join(filter_expr_list)

        # Perform the search
        query_embedding = [float(e) for e in query.query_embedding]
        sparse_vector = None

        if query.mode != VectorStoreQueryMode.DEFAULT:
            # already ensured: enable_support = True,  combine_dense_and_sparse != None

            alpha = 0.0  # query.mode == VectorStoreQueryMode.SPARSE:
            if query.mode == VectorStoreQueryMode.HYBRID:
                alpha = query.alpha if query.alpha is not None else 0.5

            sparse_vector = self._gen_sparse_vector_and_convert(
                query.query_str, is_document=False
            )

            query_embedding, sparse_vector = combine_dense_and_sparse(
                query_embedding, sparse_vector, alpha
            )

        rsp = self._collection.query(
            vector=query_embedding,
            sparse_vector=sparse_vector,
            topk=query.similarity_top_k,
            filter=filter_expr,
            include_vector=True,
            partition=self.partition_name,
        )
        if not rsp:
            raise Exception(f"Failed to query docs, error: {rsp}")

        top_k_ids = []
        top_k_nodes = []
        top_k_scores = []
        for doc in rsp:
            node = metadata_dict_to_node(doc.fields)

            top_k_ids.append(doc.id)
            top_k_nodes.append(node)
            top_k_scores.append(self._normalize_score(doc.score))

        logger.debug(f"Query done in collection: {self.collection_name}")

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    @property
    def _metric(self) -> str:
        # if enable_sparse is True, score is dotproduct
        # otherwise, it is cosine distance( 1 - cosine similarity)
        # ref: https://help.aliyun.com/document_detail/2584947.html
        return "dotproduct" if self.enable_sparse else "cosine"

    def _normalize_score(self, score):
        # Normalize score to [0, 1] by non-linear transformation, with score distribution changed
        if self._metric == "euclidean":
            return 1.0 - 2 * math.atan(score) / math.pi
        elif self._metric == "dotproduct":
            return 0.5 + math.atan(score) / math.pi
        elif self._metric == "cosine":
            return 1.0 - score / 2.0
        else:
            raise ValueError(f"metric not supported: {self._metric}")

    def _gen_sparse_vector_and_convert(
        self, text, is_document
    ) -> Optional[dict[int, float]]:
        if not self.enable_sparse:
            return None

        # support asymmetric sparse embedding like BM25
        _encode_func = (
            self.sparse_embedding_function.encode_documents
            if is_document
            else self.sparse_embedding_function.encode_queries
        )
        sparse_vector: dict[int, np.float32] = _encode_func([text])[0]
        return {key: float(value) for key, value in sparse_vector.items()}
