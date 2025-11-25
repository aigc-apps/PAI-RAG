from functools import partial
from typing import List, Optional
import asyncio
from config.providers.vectordb_provider import vectordb_provider
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQuery, MetadataFilters, MetadataFilter, FilterCondition, FilterOperator, VectorStoreQueryResult
from llama_index.core.tools import FunctionTool

from common.chat.models import RetrievalSetting
from db.models.knowledgebase.knowledgebase import KbEntity, RetrievalConfig
from common.knowledgebase.types import (
    VectorIndexRetrievalType,
    is_fulltext_supported,
)
from db.models.knowledgebase.metadata_filter import MetadataFilteringCondition, query_file_ids_with_metadata_filter
from rag.chunk_helper import (
    get_file_id_source_map,
)
from rag.vector_store.vector_connection import (
    create_vector_store,
    is_docid_filter_supported,
)
from rag.rerank.weight_reranker import WeightReranker, _to_llama_similarities
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from config.providers.knowledgebase_provider import knowledgebase_provider
from config.providers.embedding_provider import embedding_provider
from config.providers.reranker_provider import reranker_provider
from pairag.file.store.file_store_helper import file_store
from llama_index.core.schema import NodeWithScore
from loguru import logger
import re
import json
from typing import Annotated
from chat.tools.search_result import SearchResult
from utils.lru_cache import LruCache

MARKDOWN_IMAGE_PATTERN = r'!\[.*?\]\((.*?)\)\s*\n*\s*图片的描述:\s*(.*?)(?=\n\n|$)'
MAX_TRUNCATED_CHUNK_LEN = 8000

def retrieval_type_to_search_mode(retrieval_type: VectorIndexRetrievalType):
    if retrieval_type == VectorIndexRetrievalType.fulltext:
        return VectorStoreQueryMode.TEXT_SEARCH
    elif retrieval_type == VectorIndexRetrievalType.hybrid:
        return VectorStoreQueryMode.HYBRID
    else:
        return VectorStoreQueryMode.DEFAULT


kb_cache = LruCache(maxsize=100)

def get_kb_cache_key(knowledgebase: KbEntity):
    vector_connection = vectordb_provider.get_vector_db_connection()
    vector_str = vector_connection.model_dump_json()
    key_str = f"{knowledgebase.id}--{knowledgebase.embedding_model}--{vector_str}"
    return key_str


def merge_vector_store_results(
    text_result: VectorStoreQueryResult,
    dense_result: VectorStoreQueryResult,
) -> VectorStoreQueryResult:
    """
    合并两个 VectorStoreQueryResult，去重后返回合并结果。
    用于 reranker 场景，需要先合并再去重。

    Args:
        text_result: 文本搜索结果
        dense_result: 向量搜索结果

    Returns:
        合并并去重后的 VectorStoreQueryResult
    """
    # 使用字典去重，key 为 node_id
    merged_nodes = {}  # node_id -> (node, similarity, node_id)

    # 处理 text 搜索结果
    for i, node in enumerate(text_result.nodes):
        node_id = text_result.ids[i] if i < len(text_result.ids) else node.node_id
        similarity = text_result.similarities[i] if i < len(text_result.similarities) else 0.0

        if node_id not in merged_nodes:
            merged_nodes[node_id] = (node, similarity, node_id)
        else:
            # 如果已存在，保留相似度更高的
            existing_node, existing_similarity, existing_id = merged_nodes[node_id]
            if similarity > existing_similarity:
                merged_nodes[node_id] = (node, similarity, node_id)

    # 处理 dense 搜索结果
    for i, node in enumerate(dense_result.nodes):
        node_id = dense_result.ids[i] if i < len(dense_result.ids) else node.node_id
        similarity = dense_result.similarities[i] if i < len(dense_result.similarities) else 0.0

        if node_id not in merged_nodes:
            merged_nodes[node_id] = (node, similarity, node_id)
        else:
            # 如果已存在，保留相似度更高的
            existing_node, existing_similarity, existing_id = merged_nodes[node_id]
            if similarity > existing_similarity:
                merged_nodes[node_id] = (node, similarity, node_id)

    merged_items = list(merged_nodes.values())
    merged_items.sort(key=lambda x: x[1], reverse=True)

    top_k_nodes = [item[0] for item in merged_items]
    top_k_scores = [item[1] for item in merged_items]
    top_k_ids = [item[2] for item in merged_items]

    return VectorStoreQueryResult(
        nodes=top_k_nodes,
        ids=top_k_ids,
        similarities=top_k_scores,
    )


# 在线的知识库工具
class PaiKnowledgebaseTool:
    def create_vector_store_from_knowledgebase(
        self,
        knowledgebase: KbEntity,
        embed_model: BaseEmbedding = None,
    ) -> BasePydanticVectorStore:
        key = get_kb_cache_key(knowledgebase)
        vector_store = kb_cache.get(key)

        if vector_store is None:
            logger.info(f"Cache miss for knowledgebase {knowledgebase.id}, creating new vector store.")
            # TODO: 检查配置是否变化
            if not embed_model:
                embed_model = embedding_provider.get_embedding_model(knowledgebase.embedding_model)

            dimension = len(embed_model.get_text_embedding("0"))
            vector_connection = vectordb_provider.get_vector_db_connection()
            vector_store = create_vector_store(
                knowledgebase.id, dimension, vector_db_connection=vector_connection,
            )
            kb_cache.put(key, vector_store)
        return vector_store

    async def adelete_doc(
        self,
        kb_id: str,
        file_id: str,
    ):
        if not file_id or not kb_id:
            return

        knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
        vector_store = self.create_vector_store_from_knowledgebase(knowledgebase)
        try:
            await vector_store.adelete(ref_doc_id=file_id)
        except Exception as e:
            if "Doc is empty." in str(e):
                logger.warning("empty doc found for opensearch. skipping.")
            else:
                logger.error(f"Failed to delete doc from vector store: {e}")
                key = get_kb_cache_key(knowledgebase)
                kb_cache.delete(key) # 删除缓存，强制重新创建
                raise
        logger.info(
            f"Deleted file {file_id} from {kb_id} vector db successfully."
        )


    async def adelete_chunks_from_vectordb(
        self,
        kb_id: str,
        node_ids: List[str],
    ):
        if not node_ids:
            return

        knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
        vector_store = self.create_vector_store_from_knowledgebase(knowledgebase)
        try:
            await vector_store.adelete_nodes(node_ids=node_ids)
        except Exception as e:
            logger.error(f"Failed to delete nodes from vector store: {e}")
            key = get_kb_cache_key(knowledgebase)
            kb_cache.delete(key) # 删除缓存，强制重新创建
            raise
        logger.info(
            f"Deleted {len(node_ids)} chunks from {kb_id} vector db successfully."
        )


    def get_node_texts_for_embedding(self, nodes) -> list[str]:
        texts = []
        for node in nodes:
            base_text = f"filename: {node.metadata['file_name']}"
            chapter_name = node.metadata.get('chapter_name', '').strip()
            if chapter_name:
                base_text += f"\n\nchapter_name: {chapter_name}"

            base_text += f"\n\n{node.text}"

            texts.append(base_text[:3000])
        return texts


    async def ainsert_chunks_to_vectordb(
        self,
        kb_id: str,
        nodes,
    ):
        logger.info(f"Starting to insert {len(nodes)} into knowledgebase {kb_id}.")
        knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
        vector_store = self.create_vector_store_from_knowledgebase(knowledgebase)
        embed_model:BaseEmbedding = embedding_provider.get_embedding_model(
            knowledgebase.embedding_model
        )
        texts_to_embed = self.get_node_texts_for_embedding(nodes)
        embeddings = await embed_model.aget_text_embedding_batch(texts_to_embed, show_progress=True)
        for i in range(len(nodes)):
            nodes[i].embedding = embeddings[i]

        try:
            await vector_store.async_add(nodes)
        except Exception as e:
            logger.error(f"Failed to insert nodes into vector store: {e}")
            key = get_kb_cache_key(knowledgebase)
            kb_cache.delete(key) # 删除缓存，强制重新创建
            raise
        logger.info(f"Finished inserting {len(nodes)} into vector store.")

    async def aquery(
        self,
        query,
        knowledge_id: str,
        user_id: str = None,
        retrieval_setting: Optional[RetrievalSetting] = None,
        metadata_condition: Optional[MetadataFilteringCondition] = None,
    ) -> List[NodeWithScore]:
        logger.info(f"Starting to query knowledgebase {knowledge_id} with query: {query}.")
        knowledgebase = await knowledgebase_provider.aget_knowledgebase(knowledge_id)
        retrieval_config = RetrievalConfig.model_validate(
            knowledgebase.retrieval_config
        )
        logger.info(f"Get retrieval config: {retrieval_config}.")

        vector_store: BasePydanticVectorStore = self.create_vector_store_from_knowledgebase(knowledgebase)

        embed_model:BaseEmbedding = embedding_provider.get_embedding_model(
            knowledgebase.embedding_model
        )

        query_embedding = await embed_model.aget_query_embedding(query)
        document_ids = await query_file_ids_with_metadata_filter(kb_id=knowledge_id, user_id=user_id, metadata_filter=metadata_condition)
        logger.info(f"Successfully filtered {len(document_ids)} files with metadata filter: {document_ids}.")

        if not document_ids:
            # fail fast as no docs filtered.
            return []

        retrieval_mode = retrieval_config.retrieval_mode
        if retrieval_setting and retrieval_setting.retrieval_mode:
            retrieval_mode = retrieval_setting.retrieval_mode

        # 检查向量数据库是否支持全文检索和混合检索
        vector_db_type = vectordb_provider.get_vector_db_type()

        if not is_fulltext_supported(vector_db_type):
            # 如果不支持全文检索，强制使用向量检索
            if retrieval_mode in [VectorIndexRetrievalType.fulltext, VectorIndexRetrievalType.hybrid]:
                logger.warning(
                    f"Vector database type {vector_db_type} does not support {retrieval_mode} retrieval. "
                    f"Falling back to vector retrieval."
                )
                retrieval_mode = VectorIndexRetrievalType.vector

        query_mode = retrieval_type_to_search_mode(retrieval_mode)


        vector_weight = retrieval_config.vector_weight
        if retrieval_setting and retrieval_setting.vector_weight is not None:
            vector_weight = retrieval_setting.vector_weight

        top_k = retrieval_config.top_k
        if retrieval_setting and retrieval_setting.top_k is not None:
            top_k = retrieval_setting.top_k

        similarity_threshold = retrieval_config.similarity_threshold
        if retrieval_setting and retrieval_setting.similarity_threshold is not None:
            similarity_threshold = retrieval_setting.similarity_threshold

        enable_rerank = retrieval_config.enable_rerank
        if retrieval_setting and retrieval_setting.enable_rerank is not None:
            enable_rerank = retrieval_setting.enable_rerank

        reranker_top_k = None
        if enable_rerank:
            reranker_top_k = retrieval_config.rerank_top_k
            if retrieval_setting and retrieval_setting.rerank_top_k is not None:
                reranker_top_k = retrieval_setting.rerank_top_k

        if query_mode == VectorStoreQueryMode.HYBRID:
            if is_docid_filter_supported(vector_store=vector_store):
                text_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=top_k,
                    doc_ids=document_ids,
                    query_str=query,
                    mode=VectorStoreQueryMode.TEXT_SEARCH,
                )
                dense_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=top_k,
                    doc_ids=document_ids,
                    query_str=query,
                    mode=VectorStoreQueryMode.DEFAULT,
                )
            else:
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
                text_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=top_k,
                    query_str=query,
                    mode=VectorStoreQueryMode.TEXT_SEARCH,
                    filters=metadata_filters,
                )
                dense_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=top_k,
                    query_str=query,
                    mode=VectorStoreQueryMode.DEFAULT,
                    filters=metadata_filters,
                )

            try:
                text_result_task = vector_store.aquery(text_query)
                dense_result_task = vector_store.aquery(dense_query)
                text_result, dense_result = await asyncio.gather(text_result_task, dense_result_task)

                # 对 TEXT_SEARCH 模式的分数进行 min-max 归一化
                if text_result.similarities:
                    text_scores = text_result.similarities
                    normalized_text_scores = _to_llama_similarities(text_scores)
                    text_result.similarities = normalized_text_scores
                    logger.info(f"TEXT_SEARCH mode: Normalized scores - min={min(text_scores) if text_scores else 'N/A'}, max={max(text_scores) if text_scores else 'N/A'}")

                logger.info(f"HYBRID mode: Retrieved {len(text_result.nodes)} text nodes and {len(dense_result.nodes)} dense nodes.")
            except Exception as e:
                logger.error(f"Failed to query vector store: {e}")
                key = get_kb_cache_key(knowledgebase)
                kb_cache.delete(key) # 删除缓存，强制重新创建
                raise

            # 根据 enable_rerank 决定使用 reranker 还是 weight_reranker
            if enable_rerank and len(text_result.nodes) + len(dense_result.nodes) > 1 and query:
                merged_result = merge_vector_store_results(text_result, dense_result)
                logger.info(f"HYBRID mode: Merged to {len(merged_result.nodes)} unique nodes for reranker.")

                rerank_model = retrieval_config.rerank_model
                if retrieval_setting and retrieval_setting.rerank_model:
                    rerank_model = retrieval_setting.rerank_model
                raranker_model = reranker_provider.get_reranker_model(rerank_model)
                query_result = await raranker_model.vector_store_rerank(
                    query=query,
                    result=merged_result,
                    top_n=reranker_top_k)
                logger.info(f"Reranked {len(query_result.nodes)} nodes.")
            else:
                text_weight = 1 - vector_weight
                reranker = WeightReranker(
                    vector_weight=vector_weight,
                    text_weight=text_weight,
                )
                query_result = reranker.rerank(
                    text_result=text_result,
                    dense_result=dense_result,
                    top_k=top_k,
                )
                logger.info(f"HYBRID mode: Weight reranked to {len(query_result.nodes)} nodes.")
        else:
            # 直接按doc_id过滤
            if is_docid_filter_supported(vector_store=vector_store):
                logger.info("Using doc_id as filters.")
                vector_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=top_k,
                    doc_ids=document_ids,
                    query_str=query,
                    mode=query_mode,
                    alpha=vector_weight,
                )
            else:
                # 使用llama_index filters 过滤
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

                vector_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=top_k,
                    query_str=query,
                    mode=query_mode,
                    alpha=vector_weight,
                    filters=metadata_filters,
                )

            try:
                query_result = await vector_store.aquery(vector_query)

                # 对 TEXT_SEARCH 模式的分数进行 min-max 归一化
                if query_mode == VectorStoreQueryMode.TEXT_SEARCH and query_result.similarities:
                    text_scores = query_result.similarities
                    normalized_text_scores = _to_llama_similarities(text_scores)
                    query_result.similarities = normalized_text_scores
                    logger.info(f"TEXT_SEARCH mode: Normalized scores - min={min(text_scores) if text_scores else 'N/A'}, max={max(text_scores) if text_scores else 'N/A'}")

                logger.info(f"Retrieved {len(query_result.nodes)} nodes from vector index.")
            except Exception as e:
                logger.error(f"Failed to query vector store: {e}")
                key = get_kb_cache_key(knowledgebase)
                kb_cache.delete(key) # 删除缓存，强制重新创建
                raise

            if enable_rerank and len(query_result.nodes) > 1 and query:
                rerank_model = retrieval_config.rerank_model
                if retrieval_setting and retrieval_setting.rerank_model:
                    rerank_model = retrieval_setting.rerank_model
                raranker_model = reranker_provider.get_reranker_model(rerank_model)
                query_result = await raranker_model.vector_store_rerank(
                    query=query,
                    result=query_result,
                    top_n=reranker_top_k)
                logger.info(f"Reranked {len(query_result.nodes)} nodes.")

        result_nodes = []
        file_ids = []
        for i, node in enumerate(query_result.nodes):
            if query_result.similarities[i] >= similarity_threshold:
                images = []
                file_ids.append(node.metadata["doc_id"])
                origin_text = node.text
                pattern = MARKDOWN_IMAGE_PATTERN
                matches = re.findall(pattern, origin_text, re.DOTALL)
                for _, (src, desc)  in enumerate(matches):
                    image_url = file_store.get_url(src)
                    origin_text = origin_text.replace(src, image_url)
                    images.append({"url": image_url, "desc": desc})
                node.text = origin_text
                node.metadata["images_info"] = images
                result_nodes.append(NodeWithScore(node=node, score=query_result.similarities[i]))

        file_source_map = await get_file_id_source_map(kb_id=knowledge_id, file_ids=file_ids)
        for node in result_nodes:
            file_source = file_source_map.get(node.node.metadata["doc_id"])
            node.node.metadata["file_source"] = file_source

        logger.info(f"Get {len(result_nodes)} nodes above given threshold {similarity_threshold}.")
        return result_nodes

    async def aquery_for_attachments(
        self,
        query,
        knowledge_id: str,
        retrieval_setting: Optional[RetrievalSetting] = None,
        document_ids: List[str] = None,
    ) -> List[NodeWithScore]:
        if not document_ids:
            # fail fast as no attachment docs filtered.
            return []

        logger.info(f"Starting to query knowledgebase {knowledge_id} with query: {query}.")
        knowledgebase = await knowledgebase_provider.aget_knowledgebase(knowledge_id)
        retrieval_config = RetrievalConfig.model_validate(
            knowledgebase.retrieval_config
        )
        vector_store: BasePydanticVectorStore = self.create_vector_store_from_knowledgebase(knowledgebase)

        embed_model = embedding_provider.get_embedding_model(
            knowledgebase.embedding_model
        )

        query_embedding = await embed_model.aget_query_embedding(query)

        retrieval_mode = retrieval_config.retrieval_mode
        if retrieval_setting and retrieval_setting.retrieval_mode:
            retrieval_mode = retrieval_setting.retrieval_mode

        # 检查向量数据库是否支持全文检索和混合检索
        vector_db_type = vectordb_provider.get_vector_db_type()

        if not is_fulltext_supported(vector_db_type):
            # 如果不支持全文检索，强制使用向量检索
            if retrieval_mode in [VectorIndexRetrievalType.fulltext, VectorIndexRetrievalType.hybrid]:
                logger.warning(
                    f"Vector database type {vector_db_type} does not support {retrieval_mode} retrieval. "
                    f"Falling back to vector retrieval."
                )
                retrieval_mode = VectorIndexRetrievalType.vector

        query_mode = retrieval_type_to_search_mode(retrieval_mode)

        vector_weight = retrieval_config.vector_weight
        if retrieval_setting and retrieval_setting.vector_weight is not None:
            vector_weight = retrieval_setting.vector_weight

        top_k = retrieval_config.top_k
        if retrieval_setting and retrieval_setting.top_k is not None:
            top_k = retrieval_setting.top_k

        similarity_threshold = retrieval_config.similarity_threshold
        if retrieval_setting and retrieval_setting.similarity_threshold is not None:
            similarity_threshold = retrieval_setting.similarity_threshold

        enable_rerank = retrieval_config.enable_rerank
        if retrieval_setting and retrieval_setting.enable_rerank is not None:
            enable_rerank = retrieval_setting.enable_rerank

        reranker_top_k = None
        if enable_rerank:
            reranker_top_k = retrieval_config.rerank_top_k
            if retrieval_setting and retrieval_setting.rerank_top_k is not None:
                reranker_top_k = retrieval_setting.rerank_top_k

        if query_mode == VectorStoreQueryMode.HYBRID:
            # 创建 text 和 dense 查询
            text_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                doc_ids=document_ids,
                query_str=query,
                mode=VectorStoreQueryMode.TEXT_SEARCH,
            )
            dense_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                doc_ids=document_ids,
                query_str=query,
                mode=VectorStoreQueryMode.DEFAULT,
            )

            try:
                text_result_task = vector_store.aquery(text_query)
                dense_result_task = vector_store.aquery(dense_query)
                text_result, dense_result = await asyncio.gather(text_result_task, dense_result_task)

                # 对 TEXT_SEARCH 模式的分数进行 min-max 归一化
                if text_result.similarities:
                    text_scores = text_result.similarities
                    normalized_text_scores = _to_llama_similarities(text_scores)
                    text_result.similarities = normalized_text_scores
                    logger.info(f"TEXT_SEARCH mode: Normalized scores - min={min(text_scores) if text_scores else 'N/A'}, max={max(text_scores) if text_scores else 'N/A'}")

                logger.info(f"HYBRID mode: Retrieved {len(text_result.nodes)} text nodes and {len(dense_result.nodes)} dense nodes.")
            except Exception as e:
                logger.error(f"Failed to query vector store: {e}")
                key = get_kb_cache_key(knowledgebase)
                kb_cache.delete(key) # 删除缓存，强制重新创建
                raise

            # 根据 enable_rerank 决定使用 reranker 还是 weight_reranker
            if enable_rerank and len(text_result.nodes) + len(dense_result.nodes) > 1 and query:
                # 合并两个结果并去重
                merged_result = merge_vector_store_results(text_result, dense_result)
                logger.info(f"HYBRID mode: Merged to {len(merged_result.nodes)} unique nodes for reranker.")

                rerank_model = retrieval_config.rerank_model
                if retrieval_setting and retrieval_setting.rerank_model:
                    rerank_model = retrieval_setting.rerank_model
                raranker_model = reranker_provider.get_reranker_model(rerank_model)
                query_result = await raranker_model.vector_store_rerank(
                    query=query,
                    result=merged_result,
                    top_n=reranker_top_k)
                logger.info(f"Reranked {len(query_result.nodes)} nodes.")
            else:
                # 使用 weight_reranker 合并结果
                text_weight = 1 - vector_weight
                reranker = WeightReranker(
                    vector_weight=vector_weight,
                    text_weight=text_weight,
                )
                query_result = reranker.rerank(
                    text_result=text_result,
                    dense_result=dense_result,
                    top_k=top_k,
                )
                logger.info(f"HYBRID mode: Weight reranked to {len(query_result.nodes)} nodes.")
        else:
            vector_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                doc_ids=document_ids,
                query_str=query,
                mode=query_mode,
                alpha=vector_weight,
            )

            try:
                query_result = await vector_store.aquery(vector_query)

                # 对 TEXT_SEARCH 模式的分数进行 min-max 归一化
                if query_mode == VectorStoreQueryMode.TEXT_SEARCH and query_result.similarities:
                    text_scores = query_result.similarities
                    normalized_text_scores = _to_llama_similarities(text_scores)
                    query_result.similarities = normalized_text_scores
                    logger.info(f"TEXT_SEARCH mode: Normalized scores - min={min(text_scores) if text_scores else 'N/A'}, max={max(text_scores) if text_scores else 'N/A'}")
            except Exception as e:
                logger.error(f"Failed to query vector store: {e}")
                key = get_kb_cache_key(knowledgebase)
                kb_cache.delete(key) # 删除缓存，强制重新创建
                raise

            if enable_rerank and len(query_result.nodes) > 1 and query:
                rerank_model = retrieval_config.rerank_model
                if retrieval_setting and retrieval_setting.rerank_model:
                    rerank_model = retrieval_setting.rerank_model
                raranker_model = reranker_provider.get_reranker_model(rerank_model)
                query_result = await raranker_model.vector_store_rerank(
                    query=query,
                    result=query_result,
                    top_n=reranker_top_k)
                logger.info(f"Reranked {len(query_result.nodes)} nodes.")

        result_nodes = []
        for i, node in enumerate(query_result.nodes):
            if query_result.similarities[i] >= similarity_threshold:
                images = []
                origin_text = node.text
                pattern = MARKDOWN_IMAGE_PATTERN
                matches = re.findall(pattern, origin_text, re.DOTALL)
                for _, (src, desc)  in enumerate(matches):
                    image_url = file_store.get_url(src)
                    origin_text = origin_text.replace(src, image_url)
                    images.append({"url": image_url, "desc": desc})
                node.text = origin_text
                node.metadata["images_info"] = images
                result_nodes.append(NodeWithScore(node=node, score=query_result.similarities[i]))
        logger.info(f"Retrieved {len(result_nodes)} nodes from vector index.")
        return result_nodes


kb_tool = PaiKnowledgebaseTool()


async def aget_knowledgebase_result(query: str, kb_id: str, user_id: str="anonymous") -> str:
    """Get aliyun search tool"""
    logger.info(f"Searching knowledgebase with kb {kb_id} and user {user_id}.")
    result_nodes = await kb_tool.aquery(query=query, knowledge_id=kb_id, user_id=user_id)
    records = []
    for score_node in result_nodes:
        file_url = score_node.node.metadata.get("file_source")
        if not file_url:
            file_url = file_store.get_url(score_node.node.metadata.get("file_path", ""))
        records.append(
            SearchResult(
                score=score_node.score,
                content=score_node.node.get_content()[:MAX_TRUNCATED_CHUNK_LEN],
                images=score_node.node.metadata.get("images_info", []),
                url=file_url,
                title=score_node.node.metadata.get("file_name", ""),
            ).model_dump())
    return json.dumps({"result": records}, ensure_ascii=False)


async def aget_knowledgebase_tool(kb_id: str, user_id: Optional[str] = None):
    knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
    aquery_knowledgebase_func = partial(aget_knowledgebase_result, kb_id=kb_id, user_id=user_id)

    async def query_knowledgebase_handler(
        query: Annotated[
            str,
            "根据上下文添加必要的背景信息，改写一个新的独立问题，使问题更完整，注意指代消解、完善主语等",
        ] = "",
    ):
        return await aquery_knowledgebase_func(
            query=query,
        )

    search_knowledgebase_tool = FunctionTool.from_defaults(
        async_fn=query_knowledgebase_handler,
        name=f"search-knowledgebase-{kb_id}",
        description=f"根据上下文从知识库中搜索和用户查询相关的内容。\n知识库名称: {knowledgebase.name}\n知识库描述: {knowledgebase.description}\n",
    )
    return search_knowledgebase_tool
