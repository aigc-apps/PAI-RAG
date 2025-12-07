from typing import List, Optional
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStoreQueryResult
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from common.knowledgebase.types import VectorIndexRetrievalType
from llama_index.core.vector_stores.types import VectorStoreQuery, MetadataFilters, FilterCondition, FilterOperator, MetadataFilter
from loguru import logger
import asyncio
from rag.rerank.fusion_reranker import min_max_normalize_scores


def retrieval_type_to_search_mode(retrieval_type: VectorIndexRetrievalType):
    if retrieval_type == VectorIndexRetrievalType.fulltext:
        return VectorStoreQueryMode.TEXT_SEARCH
    elif retrieval_type == VectorIndexRetrievalType.hybrid:
        return VectorStoreQueryMode.HYBRID
    else:
        return VectorStoreQueryMode.DEFAULT


async def aquery_vector_store(
    vector_store: BasePydanticVectorStore,
    query: str,
    query_embedding: List[float],
    document_ids: List[str],
    query_mode: VectorIndexRetrievalType,
    top_k: int,
    use_docid_filter: bool = True,
) -> tuple[Optional[VectorStoreQueryResult], Optional[VectorStoreQueryResult]]:
    """
    执行向量存储查询，返回 text_result 和 dense_result。
    """
    query_mode = retrieval_type_to_search_mode(query_mode)

    text_result = None
    dense_result = None

    metadata_filters = None
    if not use_docid_filter and document_ids:
        metadata_filters = MetadataFilters(
            condition=FilterCondition.AND,
            filters=[
                MetadataFilter(
                    key="doc_id",
                    value=document_ids,
                    operator=FilterOperator.IN,
                )
            ],
        )
        logger.info(f"Using metadata filters {metadata_filters}.")
    else:
        logger.info("Using doc_id as filters.")

    def _build_query_kwargs(mode: VectorStoreQueryMode):
        kwargs = {
            "query_embedding": query_embedding,
            "similarity_top_k": top_k,
            "query_str": query,
            "mode": mode,
        }
        if use_docid_filter:
            kwargs["doc_ids"] = document_ids
        elif metadata_filters:
            kwargs["filters"] = metadata_filters
        return kwargs

    try:
        if query_mode == VectorStoreQueryMode.HYBRID:
            # 混合模式：并行执行文本搜索和向量搜索
            text_query = VectorStoreQuery(**_build_query_kwargs(VectorStoreQueryMode.TEXT_SEARCH))
            dense_query = VectorStoreQuery(**_build_query_kwargs(VectorStoreQueryMode.DEFAULT))

            text_result_task = vector_store.aquery(text_query)
            dense_result_task = vector_store.aquery(dense_query)
            text_result, dense_result = await asyncio.gather(text_result_task, dense_result_task)

            logger.info(f"HYBRID mode: Retrieved {len(text_result.nodes)} text nodes and {len(dense_result.nodes)} dense nodes.")
        else:
            vector_query = VectorStoreQuery(**_build_query_kwargs(query_mode))
            query_result = await vector_store.aquery(vector_query)

            if query_mode == VectorStoreQueryMode.TEXT_SEARCH:
                text_result = query_result
            else:
                dense_result = query_result

            logger.info(f"{query_mode} mode: Retrieved {len(query_result.nodes)} nodes.")

        # TEXT_SEARCH 模式的分数归一化
        if text_result and text_result.similarities:
            text_result.similarities = min_max_normalize_scores(text_result.similarities)

    except Exception as e:
        logger.error(f"Failed to query vector store: {e}")
        raise

    return text_result, dense_result
