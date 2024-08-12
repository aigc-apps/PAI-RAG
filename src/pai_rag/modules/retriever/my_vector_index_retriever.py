"""Base vector store index query."""

from typing import List, Optional

from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.indices.utils import log_vector_store_query_result
from llama_index.core.schema import ObjectType
from llama_index.core.vector_stores.types import (
    VectorStoreQueryResult,
)
import llama_index.core.instrumentation as instrument

from llama_index.core.retrievers import (
    VectorIndexRetriever,
)
from pai_rag.integrations.retrievers.fusion_retriever import NodeWithScore

dispatcher = instrument.get_dispatcher(__name__)


class MyVectorIndexRetriever(VectorIndexRetriever):
    """PAI-RAG customized vector index retriever.

    Refactor the _build_node_list_from_query_result() function

    and return the results with the query_result.similarities sorted in descending order.

    Args:
        index (MyVectorIndexRetriever): vector store index.
        similarity_top_k (int): number of top k results to return.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        alpha (float): weight for sparse/dense retrieval, only used for
            hybrid query mode.
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

    """

    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> List[NodeWithScore]:
        if query_result.nodes is None:
            # NOTE: vector store does not keep text and returns node indices.
            # Need to recover all nodes from docstore
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index.index_struct, IndexDict)
            node_ids = [
                self._index.index_struct.nodes_dict[idx] for idx in query_result.ids
            ]
            nodes = self._docstore.get_nodes(node_ids)
            query_result.nodes = nodes
        else:
            # NOTE: vector store keeps text, returns nodes.
            # Only need to recover image or index nodes from docstore
            for i in range(len(query_result.nodes)):
                source_node = query_result.nodes[i].source_node
                if (not self._vector_store.stores_text) or (
                    source_node is not None and source_node.node_type != ObjectType.TEXT
                ):
                    node_id = query_result.nodes[i].node_id
                    if self._docstore.document_exists(node_id):
                        query_result.nodes[i] = self._docstore.get_node(
                            node_id
                        )  # type: ignore[index]

        log_vector_store_query_result(query_result)

        node_with_scores: List[NodeWithScore] = []
        query_result.similarities = sorted(query_result.similarities, reverse=True)
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
                node_with_scores.append(NodeWithScore(node=node, score=score))

        return node_with_scores
