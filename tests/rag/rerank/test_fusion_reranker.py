import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.schema import TextNode

from rag.rerank.fusion_reranker import (
    min_max_normalize_scores,
    weight_rerank,
    merge_vector_store_results_by_text,
    filter_node_result,
)


def _make_node(node_id: str, text: str = "") -> TextNode:
    return TextNode(id_=node_id, text=text or f"text_{node_id}")


def _make_result(nodes, scores) -> VectorStoreQueryResult:
    return VectorStoreQueryResult(
        nodes=nodes,
        ids=[n.node_id for n in nodes],
        similarities=scores,
    )


class TestMinMaxNormalizeScores:
    def test_empty_list(self):
        assert min_max_normalize_scores([]) == []

    def test_all_same_positive(self):
        result = min_max_normalize_scores([5.0, 5.0, 5.0])
        assert result == [1.0, 1.0, 1.0]

    def test_all_same_zero(self):
        result = min_max_normalize_scores([0.0, 0.0])
        assert result == [0.0, 0.0]

    def test_normal_normalization(self):
        result = min_max_normalize_scores([1.0, 3.0, 5.0])
        assert result == [0.0, 0.5, 1.0]

    def test_with_negative_values(self):
        result = min_max_normalize_scores([-2.0, 0.0, 2.0])
        assert result == [0.0, 0.5, 1.0]


class TestWeightRerank:
    def test_no_overlap_weighted_merge(self):
        n1 = _make_node("a")
        n2 = _make_node("b")
        text_result = _make_result([n1], [0.8])
        dense_result = _make_result([n2], [0.6])
        result = weight_rerank(text_result, dense_result, vector_weight=0.5, top_k=10)
        assert len(result.nodes) == 2
        assert set(result.ids) == {"a", "b"}

    def test_duplicate_node_id_weighted(self):
        n1 = _make_node("a")
        n1_dup = _make_node("a")
        text_result = _make_result([n1], [0.8])
        dense_result = _make_result([n1_dup], [0.6])
        result = weight_rerank(text_result, dense_result, vector_weight=0.5, top_k=10)
        # Duplicate node_id should be merged
        assert len(result.nodes) == 1
        # Score = 0.8 * 0.5 + 0.6 * 0.5 = 0.7
        assert abs(result.similarities[0] - 0.7) < 1e-6

    def test_threshold_filtering(self):
        n1 = _make_node("a")
        n2 = _make_node("b")
        text_result = _make_result([n1], [0.1])
        dense_result = _make_result([n2], [0.1])
        result = weight_rerank(
            text_result, dense_result,
            vector_weight=0.5, top_k=10, similarity_threshold=0.5,
        )
        assert len(result.nodes) == 0

    def test_top_k_limit(self):
        nodes = [_make_node(f"n{i}") for i in range(5)]
        text_result = _make_result(nodes, [0.9, 0.8, 0.7, 0.6, 0.5])
        dense_result = _make_result([], [])
        result = weight_rerank(text_result, dense_result, vector_weight=0.5, top_k=3)
        assert len(result.nodes) == 3

    def test_sorted_descending(self):
        n1 = _make_node("a")
        n2 = _make_node("b")
        text_result = _make_result([n1, n2], [0.3, 0.9])
        dense_result = _make_result([], [])
        result = weight_rerank(text_result, dense_result, vector_weight=0.5, top_k=10)
        assert result.similarities[0] >= result.similarities[1]


class TestMergeVectorStoreResultsByText:
    def test_no_overlap_merge(self):
        n1 = _make_node("a", text="text_a")
        n2 = _make_node("b", text="text_b")
        r1 = _make_result([n1], [0.9])
        r2 = _make_result([n2], [0.8])
        result = merge_vector_store_results_by_text(r1, r2)
        assert len(result.nodes) == 2

    def test_same_text_dedup_text_result_priority(self):
        n1 = _make_node("a", text="same_text")
        n2 = _make_node("b", text="same_text")
        r1 = _make_result([n1], [0.9])
        r2 = _make_result([n2], [0.5])
        result = merge_vector_store_results_by_text(r1, r2)
        assert len(result.nodes) == 1
        # text_result (r1) should take priority
        assert result.similarities[0] == 0.9

    def test_empty_results(self):
        r1 = _make_result([], [])
        r2 = _make_result([], [])
        result = merge_vector_store_results_by_text(r1, r2)
        assert len(result.nodes) == 0

    def test_skips_empty_text_nodes(self):
        n_empty = TextNode(id_="a", text="")
        n_keep = TextNode(id_="b", text="real content")
        r1 = _make_result([n_empty, n_keep], [0.9, 0.8])
        r2 = _make_result([], [])
        result = merge_vector_store_results_by_text(r1, r2)
        assert result.ids == ["b"]

    def test_skips_whitespace_only_text_nodes(self):
        n_ws = TextNode(id_="a", text="   \n\t  ")
        n_keep = TextNode(id_="b", text="real content")
        r1 = _make_result([n_ws], [0.9])
        r2 = _make_result([n_keep], [0.8])
        result = merge_vector_store_results_by_text(r1, r2)
        assert result.ids == ["b"]

    def test_all_empty_text_returns_empty(self):
        r1 = _make_result([TextNode(id_="a", text="")], [0.9])
        r2 = _make_result([TextNode(id_="b", text="  ")], [0.8])
        result = merge_vector_store_results_by_text(r1, r2)
        assert len(result.nodes) == 0


class TestFilterNodeResult:
    def test_threshold_filter_and_sort(self):
        nodes = [_make_node("a"), _make_node("b"), _make_node("c")]
        result = _make_result(nodes, [0.3, 0.9, 0.6])
        filtered = filter_node_result(result, similarity_threshold=0.5)
        assert len(filtered.nodes) == 2
        assert filtered.similarities[0] >= filtered.similarities[1]

    def test_empty_result(self):
        filtered = filter_node_result(None, similarity_threshold=0.0)
        assert len(filtered.nodes) == 0

    def test_all_below_threshold(self):
        nodes = [_make_node("a"), _make_node("b")]
        result = _make_result(nodes, [0.1, 0.2])
        filtered = filter_node_result(result, similarity_threshold=0.5)
        assert len(filtered.nodes) == 0

    def test_zero_threshold_keeps_all(self):
        nodes = [_make_node("a"), _make_node("b")]
        result = _make_result(nodes, [0.1, 0.2])
        filtered = filter_node_result(result, similarity_threshold=0.0)
        assert len(filtered.nodes) == 2
