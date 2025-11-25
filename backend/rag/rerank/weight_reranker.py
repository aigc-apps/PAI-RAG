"""Weight reranker for merging text and dense search results."""

from logging import getLogger
from typing import List

from llama_index.core.vector_stores.types import VectorStoreQueryResult

logger = getLogger(__name__)


def _to_llama_similarities(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range using min-max normalization."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(x - min_score) / (max_score - min_score) for x in scores]


class WeightReranker:
    """
    Weight reranker for merging text and dense search results.

    This class merges text search and dense vector search results using
    weighted sum of normalized scores.
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
    ):
        """
        Initialize the weight reranker.

        Args:
            vector_weight: Weight for vector search results (default 0.5)
            text_weight: Weight for text search results (default 0.5)
        """
        self.vector_weight = vector_weight
        self.text_weight = text_weight

    def rerank(
        self,
        text_result: VectorStoreQueryResult,
        dense_result: VectorStoreQueryResult,
        top_k: int = 10,
    ) -> VectorStoreQueryResult:
        """
        Merge text and dense search results using weighted sum.

        Args:
            text_result: Text search results
            dense_result: Dense vector search results
            top_k: Number of top results to return

        Returns:
            Merged VectorStoreQueryResult
        """
        logger.info(
            f"weight_reranker: Received {len(text_result.nodes)} text nodes "
            f"and {len(dense_result.nodes)} dense nodes"
        )

        # Normalize text_result scores using min-max normalization
        if text_result.similarities:
            normalized_text_scores = _to_llama_similarities(text_result.similarities)
            logger.info(
                f"weight_reranker: Normalized text scores - "
                f"min={min(text_result.similarities) if text_result.similarities else 'N/A'}, "
                f"max={max(text_result.similarities) if text_result.similarities else 'N/A'}"
            )
        else:
            normalized_text_scores = []

        logger.info(
            f"weight_reranker: Merging with weights - "
            f"vector_weight={self.vector_weight}, text_weight={self.text_weight}"
        )

        # Merge results: use dict to deduplicate, key is node_id,
        # value is (node, text_score, dense_score, node_id)
        merged_nodes = {}  # node_id -> (node, text_score, dense_score, node_id)

        # Process text search results (already normalized)
        for i, node in enumerate(text_result.nodes):
            node_id = text_result.ids[i] if i < len(text_result.ids) else node.node_id
            normalized_score = (
                normalized_text_scores[i] if i < len(normalized_text_scores) else 0.0
            )
            weighted_score = normalized_score * self.text_weight  # Apply weight

            if node_id not in merged_nodes:
                merged_nodes[node_id] = (node, weighted_score, None, node_id)
            else:
                # If already exists, update text score
                (
                    existing_node,
                    existing_text_score,
                    existing_dense_score,
                    existing_id,
                ) = merged_nodes[node_id]
                merged_nodes[node_id] = (
                    existing_node,
                    weighted_score,
                    existing_dense_score,
                    existing_id,
                )

        # Process dense search results
        for i, node in enumerate(dense_result.nodes):
            node_id = dense_result.ids[i] if i < len(dense_result.ids) else node.node_id
            raw_score = (
                dense_result.similarities[i] if i < len(dense_result.similarities) else 0.0
            )
            weighted_score = raw_score * self.vector_weight  # Apply weight

            if node_id not in merged_nodes:
                merged_nodes[node_id] = (node, None, weighted_score, node_id)
            else:
                # If already exists, update dense score
                (
                    existing_node,
                    existing_text_score,
                    existing_dense_score,
                    existing_id,
                ) = merged_nodes[node_id]
                merged_nodes[node_id] = (
                    existing_node,
                    existing_text_score,
                    weighted_score,
                    existing_id,
                )

        # Calculate final scores and build results
        merged_items = []
        for node_id, (node, text_score, dense_score, _) in merged_nodes.items():
            if text_score is not None and dense_score is not None:
                # Both searches found it, use weighted sum
                final_score = text_score + dense_score
            elif text_score is not None:
                # Only text search found it
                final_score = text_score
            elif dense_score is not None:
                # Only dense search found it
                final_score = dense_score
            else:
                # Should not happen in theory
                final_score = 0.0

            merged_items.append((node, final_score, node_id))

        # Sort by score and take top_k
        merged_items.sort(key=lambda x: x[1], reverse=True)
        merged_items = merged_items[:top_k]

        logger.info(f"weight_reranker: Merged to {len(merged_items)} unique nodes")

        # Build return result
        top_k_nodes = [item[0] for item in merged_items]
        top_k_scores = [item[1] for item in merged_items]
        top_k_ids = [item[2] for item in merged_items]

        return VectorStoreQueryResult(
            nodes=top_k_nodes,
            ids=top_k_ids,
            similarities=top_k_scores,
        )
