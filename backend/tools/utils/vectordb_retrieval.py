from typing import List, Optional
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStoreQueryResult
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from common.knowledgebase.types import VectorIndexRetrievalType
from llama_index.core.vector_stores.types import VectorStoreQuery, MetadataFilters, FilterCondition, FilterOperator, MetadataFilter
from loguru import logger
import asyncio
from rag.rerank.fusion_reranker import min_max_normalize_scores
from extensions.trace.rag_wrapper import text_search_wrapper, vector_search_wrapper

def retrieval_type_to_search_mode(retrieval_type: VectorIndexRetrievalType):
    if retrieval_type == VectorIndexRetrievalType.fulltext:
        return VectorStoreQueryMode.TEXT_SEARCH
    elif retrieval_type == VectorIndexRetrievalType.hybrid:
        return VectorStoreQueryMode.HYBRID
    else:
        return VectorStoreQueryMode.DEFAULT


@text_search_wrapper
async def _aquery_text(
    kb_id: str,
    vector_store: BasePydanticVectorStore,
    query: str,
    document_ids: List[str],
    top_k: int,
    metadata_filters: Optional[MetadataFilters] = None,
) -> VectorStoreQueryResult:
    query_kwargs = {
        "query_str": query,
        "similarity_top_k": top_k,
        "mode": VectorStoreQueryMode.TEXT_SEARCH,
    }
    if metadata_filters:
        query_kwargs["filters"] = metadata_filters
    elif document_ids:
        query_kwargs["doc_ids"] = document_ids
    text_result = await vector_store.aquery(VectorStoreQuery(**query_kwargs))
    # TEXT_SEARCH 模式的分数归一化
    if text_result and text_result.similarities:
        text_result.similarities = min_max_normalize_scores(text_result.similarities)
    if text_result and text_result.nodes:
        for node in text_result.nodes:
            node.metadata.pop("page_bbox", None)

    if text_result and text_result.nodes:
        logger.info(f"Retrieved {len(text_result.nodes)} text nodes for query '{query}' against knowledgebase {kb_id}.")
    else:
        logger.info(f"No text nodes retrieved for query '{query}' against knowledgebase {kb_id}.")
    return text_result


@vector_search_wrapper
async def _aquery_vector(
    kb_id: str,
    vector_store: BasePydanticVectorStore,
    query: str,
    query_embedding: List[float],
    document_ids: List[str],
    top_k: int,
    metadata_filters: Optional[MetadataFilters] = None,
) -> VectorStoreQueryResult:
    query_kwargs = {
        "query_embedding": query_embedding,
        "similarity_top_k": top_k,
        "mode": VectorStoreQueryMode.DEFAULT,
        "query_str": query,
    }
    if metadata_filters:
        query_kwargs["filters"] = metadata_filters
    elif document_ids:
        query_kwargs["doc_ids"] = document_ids
    dense_result = await vector_store.aquery(VectorStoreQuery(**query_kwargs))
    if dense_result and dense_result.nodes:
        for node in dense_result.nodes:
            node.metadata.pop("page_bbox", None)

    if dense_result and dense_result.nodes:
        logger.info(f"Retrieved {len(dense_result.nodes)} dense nodes for query '{query}' against knowledgebase {kb_id}.")
    else:
        logger.info(f"No dense nodes retrieved for query '{query}' against knowledgebase {kb_id}.")
    return dense_result


async def aquery_vector_store(
    kb_id: str,
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


    try:
        if query_mode == VectorStoreQueryMode.HYBRID:
            # 混合模式：并行执行文本搜索和向量搜索
            text_result_task = _aquery_text(
                kb_id=kb_id,
                vector_store=vector_store,
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                metadata_filters=metadata_filters,
            )
            dense_result_task = _aquery_vector(
                kb_id=kb_id,
                vector_store=vector_store,
                query=query,
                query_embedding=query_embedding,
                document_ids=document_ids,
                top_k=top_k,
                metadata_filters=metadata_filters,
            )
            text_result, dense_result = await asyncio.gather(text_result_task, dense_result_task)

            logger.info(f"HYBRID mode: Retrieved {len(text_result.nodes)} text nodes and {len(dense_result.nodes)} dense nodes.")
        elif query_mode == VectorStoreQueryMode.TEXT_SEARCH:
            text_result = await _aquery_text(
                kb_id=kb_id,
                vector_store=vector_store,
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                metadata_filters=metadata_filters,
            )
            logger.info(f"{query_mode} mode: Retrieved {len(text_result.nodes)} nodes.")
        else:
            dense_result = await _aquery_vector(
                kb_id=kb_id,
                vector_store=vector_store,
                query=query,
                query_embedding=query_embedding,
                document_ids=document_ids,
                top_k=top_k,
                metadata_filters=metadata_filters,
            )
            logger.info(f"{query_mode} mode: Retrieved {len(dense_result.nodes)} nodes.")
    except Exception as e:
        logger.error(f"Failed to query vector store: {e}")
        raise

    return text_result, dense_result
