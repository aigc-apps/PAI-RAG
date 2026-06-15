"""Weight reranker for merging text and dense search results."""
from typing import Union
from rag.rerank.reranker import OpenAICompatibleReranker
from rag.rerank.dashscope_reranker import DashscopeReranker
from rag.rerank.multimodal_dashscope_reranker import MultimodalDashscopeReranker
from logging import getLogger
from typing import List
from common.knowledgebase.constants import DEFAULT_VECTOR_WEIGHT, DEFAULT_SIMILARITY_TOP_K, DEFAULT_RERANK_SIMILARITY_TOP_K
from llama_index.core.vector_stores.types import VectorStoreQueryResult

logger = getLogger(__name__)


def min_max_normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range using min-max normalization."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    logger.info(f"TEXT_SEARCH mode: Normalized scores - min={min(scores) if scores else 'N/A'}, max={max(scores) if scores else 'N/A'}")
    return [(x - min_score) / (max_score - min_score) for x in scores]


def weight_rerank(
    text_result: VectorStoreQueryResult,
    dense_result: VectorStoreQueryResult,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    top_k: int = DEFAULT_SIMILARITY_TOP_K,
    similarity_threshold: float = 0,
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
    text_weight = 1 - vector_weight
    logger.info(
        f"weight_reranker: Received {len(text_result.nodes)} text nodes "
        f"and {len(dense_result.nodes)} dense nodes"
         f"weight_reranker: Merging with weights vector_weight={vector_weight}"
    )
    ids=list(text_result.ids or [])
    nodes=list(text_result.nodes or [])
    scores = []
    if text_result.similarities:
        scores = [score * text_weight for score in text_result.similarities]
    id_index_map = {id_: i for i, id_ in enumerate(ids)}


    # 合并dense score
    for i, node in enumerate(dense_result.nodes):
        node_id = node.node_id
        if node_id in id_index_map:
            node_index = id_index_map[node_id]
            scores[node_index] += dense_result.similarities[i] * vector_weight
        else:
            ids.append(node.node_id)
            scores.append(dense_result.similarities[i] * vector_weight)
            nodes.append(node)

    if not ids:
        # 如果dense_result有结果，则返回dense_result
        return dense_result

    # Filter out scores below similarity_threshold before sorting
    filtered_data = [
        (id, node, score)
        for id, node, score in zip(ids, nodes, scores)
        if score >= similarity_threshold
    ]

    if not filtered_data:
        return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])

    ids, nodes, scores = zip(*sorted(filtered_data, key=lambda x: x[2], reverse=True))
    ids, nodes, scores = map(list, (ids, nodes, scores))

    top_k_nodes = nodes[:top_k]
    top_k_scores = scores[:top_k]
    top_k_ids = ids[:top_k]
    return VectorStoreQueryResult(
        nodes=top_k_nodes,
        ids=top_k_ids,
        similarities=top_k_scores,
    )

def merge_vector_store_results_by_text(text_result: VectorStoreQueryResult, dense_result: VectorStoreQueryResult) -> VectorStoreQueryResult:
    """
    合并两个 VectorStoreQueryResult，去重后返回合并结果。
    用于 reranker 场景，需要先合并再去重。
    空文本节点会被跳过（rerank API 不接受空文档）。

    Args:
        text_result: 文本搜索结果
        dense_result: 向量搜索结果

    Returns:
        合并并去重后的 VectorStoreQueryResult
    """
    merged_nodes = {}
    for node, similarity in zip(text_result.nodes, text_result.similarities):
        if not (node.text or "").strip():
            continue
        merged_nodes[node.text] = (node, similarity, node.node_id)

    for node, similarity in zip(dense_result.nodes, dense_result.similarities):
        if not (node.text or "").strip():
            continue
        merged_nodes.setdefault(node.text, (node, similarity, node.node_id))

    if merged_nodes:
        nodes, similarities, ids = map(
            list, zip(*merged_nodes.values())
        )
    else:
        nodes, similarities, ids = [], [], []

    return VectorStoreQueryResult(
        nodes=nodes,
        ids=ids,
        similarities=similarities,
    )

def filter_node_result(result: VectorStoreQueryResult, similarity_threshold: float = 0) -> VectorStoreQueryResult:
    """
    Filter nodes by similarity threshold.

    Args:
        result: VectorStoreQueryResult to filter
        similarity_threshold: Similarity threshold to filter nodes by

    Returns:
        VectorStoreQueryResult with nodes filtered by similarity threshold
    """

    if not result:
        return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])

    filtered_data = [
        (id, node, score)
        for id, node, score in zip(result.ids, result.nodes, result.similarities)
        if score >= similarity_threshold
    ]
    if not filtered_data:
        return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])

    ids, nodes, scores = zip(*sorted(filtered_data, key=lambda x: x[2], reverse=True))
    ids, nodes, scores = map(list, (ids, nodes, scores))
    return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)


async def arerank_fusion(
    query: str = None,
    text_result: VectorStoreQueryResult = None,
    dense_result: VectorStoreQueryResult = None,
    rerank_model: Union[DashscopeReranker, OpenAICompatibleReranker, MultimodalDashscopeReranker] = None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    top_k: int = DEFAULT_SIMILARITY_TOP_K,
    rerank_top_k: int = DEFAULT_RERANK_SIMILARITY_TOP_K,
    similarity_threshold: float = 0,
) -> VectorStoreQueryResult:
    if not text_result:
        if not rerank_model:
            return filter_node_result(dense_result, similarity_threshold)
        else:
            return await rerank_model.vector_store_rerank(
                query=query,
                vector_result=dense_result,
                top_n=rerank_top_k,
                similarity_threshold=similarity_threshold)
    elif not dense_result:
        if not rerank_model:
            return filter_node_result(text_result, similarity_threshold)
        else:
            return await rerank_model.vector_store_rerank(
                query=query,
                vector_result=text_result,
                top_n=rerank_top_k,
                similarity_threshold=similarity_threshold)
    else:
        if not rerank_model:
            return weight_rerank(text_result, dense_result, vector_weight, top_k, similarity_threshold)
        else:
            merged_result = merge_vector_store_results_by_text(text_result, dense_result)
            return await rerank_model.vector_store_rerank(
                query=query,
                vector_result=merged_result,
                top_n=rerank_top_k,
                similarity_threshold=similarity_threshold)
