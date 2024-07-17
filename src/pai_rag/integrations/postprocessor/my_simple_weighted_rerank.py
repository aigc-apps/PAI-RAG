"""LLM reranker."""
from typing import Callable, List, Optional, Dict

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.indices.utils import (
    default_format_node_batch_fn,
    default_parse_choice_select_answer_fn,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.service_context import ServiceContext
from pai_rag.integrations.retrievers.fusion_retriever import MyNodeWithScore


class MySimpleWeightedRerank(BaseNodePostprocessor):
    """LLM-based reranker."""

    vector_weight: float = Field(description="Weight of vector index retriever.")
    keyword_weight: float = Field(description="Weight of keyword index retriever.")
    top_n: int = Field(description="Top N nodes to return.")
    similarity_threshold: float = Field(
        default=None, description="Similarity threshold for the reranker scores."
    )

    _format_node_batch_fn: Callable = PrivateAttr()
    _parse_choice_select_answer_fn: Callable = PrivateAttr()

    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_n: int = 10,
        similarity_threshold: None = None,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        self._format_node_batch_fn = (
            format_node_batch_fn or default_format_node_batch_fn
        )
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn or default_parse_choice_select_answer_fn
        )

        super().__init__(
            service_context=service_context,
            top_n=top_n,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            similarity_threshold=similarity_threshold,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MySimpleWeightedRerank"

    def _postprocess_nodes(
        self,
        nodes: List[MyNodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        vector_nodes = [node for node in nodes if node.retriever_type == "vector"]
        bm25_nodes = [node for node in nodes if node.retriever_type == "bm25"]

        if len(vector_nodes) > 0 and len(bm25_nodes) == 0:
            if self.similarity_threshold:
                vector_nodes = [
                    node
                    for node in vector_nodes
                    if node.score > self.similarity_threshold
                ]
            return sorted(vector_nodes, key=lambda x: x.score or 0.0, reverse=True)[
                : self.top_n
            ]
        elif len(vector_nodes) == 0 and len(bm25_nodes) > 0:
            bm25_nodes = [node for node in bm25_nodes if node.score > 0]
            if len(bm25_nodes) > 0:
                bm25_scores = [node.score for node in bm25_nodes]
                max_score = max(bm25_scores)
                all_nodes: Dict[str, NodeWithScore] = {}
                for node_with_score in bm25_nodes:
                    node_with_score.score = node_with_score.score / max_score
                if self.similarity_threshold:
                    bm25_nodes = [
                        node
                        for node in bm25_nodes
                        if node.score > self.similarity_threshold
                    ]
                return sorted(bm25_nodes, key=lambda x: x.score or 0.0, reverse=True)[
                    : self.top_n
                ]
            else:
                return []
        else:
            all_nodes: Dict[str, NodeWithScore] = {}
            bm25_nodes = [node for node in bm25_nodes if node.score > 0]
            if len(bm25_nodes) > 0:
                for node_with_score in vector_nodes:
                    text = node_with_score.node.get_content()
                    all_nodes[text] = node_with_score
                    all_nodes[text].score = node_with_score.score * self.vector_weight

                bm25_scores = [node.score for node in bm25_nodes]
                max_score = max(bm25_scores)
                for node_with_score in bm25_nodes:
                    text = node_with_score.node.get_content()
                    node_with_score.score = node_with_score.score / max_score
                    if text in all_nodes:
                        all_nodes[text].score += (
                            node_with_score.score * self.keyword_weight
                        )
                    else:
                        all_nodes[text] = node_with_score
                        all_nodes[text].score = (
                            node_with_score.score * self.keyword_weight
                        )
            else:
                for node_with_score in vector_nodes:
                    text = node_with_score.node.get_content()
                    all_nodes[text] = node_with_score
                    all_nodes[text].score = node_with_score.score * 1.0

            if self.similarity_threshold:
                all_nodes = [
                    node
                    for node in all_nodes.values()
                    if node.score > self.similarity_threshold
                ]
                return sorted(all_nodes, key=lambda x: x.score or 0.0, reverse=True)[
                    : self.top_n
                ]
            else:
                return sorted(
                    all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True
                )[: self.top_n]
