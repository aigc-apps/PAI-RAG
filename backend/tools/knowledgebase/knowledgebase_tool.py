from functools import partial
from typing import List, Optional
from config.providers.vectordb_provider import vectordb_provider
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQuery, MetadataFilters, MetadataFilter, FilterCondition, FilterOperator
from llama_index.core.tools import FunctionTool

from common.chat.models import RetrievalSetting
from db.models.knowledgebase.knowledgebase import KbEntity, RetrievalConfig
from common.knowledgebase.types import (
    VectorIndexRetrievalType,
)
from db.models.knowledgebase.metadata_filter import MetadataFilteringCondition, query_file_ids_with_metadata_filter
from rag.chunk_helper import (
    get_file_id_source_map,
)
from rag.vector_store.vector_connection import (
    create_vector_store,
    is_docid_filter_supported,
)
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

MARKDOWN_IMAGE_PATTERN = r'!\[.*?\]\((.*?)\)\s*\n*\s*图片的描述:\s*(.*?)(?=\n\n|$)'
MAX_TRUNCATED_CHUNK_LEN = 8000

def retrieval_type_to_search_mode(retrieval_type: VectorIndexRetrievalType):
    if retrieval_type == VectorIndexRetrievalType.fulltext:
        return VectorStoreQueryMode.TEXT_SEARCH
    elif retrieval_type == VectorIndexRetrievalType.hybrid:
        return VectorStoreQueryMode.HYBRID
    else:
        return VectorStoreQueryMode.DEFAULT



class PaiKnowledgebaseTool:
    def create_vector_store_from_knowledgebase(
        self,
        knowledgebase: KbEntity,
        embed_model: BaseEmbedding = None,
    ) -> BasePydanticVectorStore:
        # TODO: 检查配置是否变化
        if not embed_model:
            embed_model = embedding_provider.get_embedding_model(knowledgebase.embedding_model)

        dimension = len(embed_model.get_text_embedding("0"))
        vector_connection = vectordb_provider.get_vector_db_connection()
        vector_store = create_vector_store(
            knowledgebase.id, dimension, vector_db_connection=vector_connection,
        )
        return vector_store
    async def adelete_chunks_from_vectordb(
        self,
        kb_id: str,
        node_ids: List[str],
    ):
        if not node_ids:
            return

        knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
        vector_store = self.create_vector_store_from_knowledgebase(knowledgebase)
        await vector_store.adelete_nodes(node_ids=node_ids)
        logger.info(
            f"Deleted {len(node_ids)} chunks from {kb_id} vector db successfully."
        )


    def get_node_texts_for_embedding(self, nodes) -> list[str]:
        texts = []
        for node in nodes:
            base_text = f"{node.text}\n\nfile_name: {node.metadata['file_name']}"
            chapter_name = node.metadata.get('chapter_name', '').strip()
            if chapter_name:
                base_text += f"\n\nchapter_name: {chapter_name}"

            texts.append(base_text)
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

        await vector_store.async_add(nodes)
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
        query_mode = retrieval_type_to_search_mode(retrieval_config.retrieval_mode)

        embed_model:BaseEmbedding = embedding_provider.get_embedding_model(
            knowledgebase.embedding_model
        )

        query_embedding = await embed_model.aget_query_embedding(query)
        document_ids = await query_file_ids_with_metadata_filter(kb_id=knowledge_id, user_id=user_id, metadata_filter=metadata_condition)
        logger.info(f"Successfully filtered {len(document_ids)} files with metadata filter: {document_ids}.")

        if not document_ids:
            # fail fast as no docs filtered.
            return []

        top_k = retrieval_config.top_k
        if retrieval_setting and retrieval_setting.top_k is not None:
            top_k = retrieval_setting.top_k
        # Optimization: we can double top_k when rerank model is given, otherwise reranking will be weak.
        if retrieval_config.enable_rerank:
            reranker_top_k = top_k
            top_k = 2 * top_k
        similarity_threshold = retrieval_config.similarity_threshold
        if retrieval_setting and retrieval_setting.score_threshold is not None:
            similarity_threshold = retrieval_setting.score_threshold

        # 直接按doc_id过滤
        if is_docid_filter_supported(vector_store=vector_store):
            logger.info("Using doc_id as filters.")
            vector_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                doc_ids=document_ids,
                query_str=query,
                mode=query_mode,
                alpha=retrieval_config.vector_weight,
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
                alpha=retrieval_config.vector_weight,
                filters=metadata_filters,
            )

        query_result = await vector_store.aquery(vector_query)
        logger.info(f"Retrieved {len(query_result.nodes)} nodes from vector index.")

        if retrieval_config.enable_rerank and len(query_result.nodes) > 0 and query:
            raranker_model = reranker_provider.get_reranker_model(
                retrieval_config.rerank_model
            )
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
        query_mode = retrieval_type_to_search_mode(retrieval_config.retrieval_mode)

        embed_model = embedding_provider.get_embedding_model(
            knowledgebase.embedding_model
        )

        query_embedding = await embed_model.aget_query_embedding(query)


        top_k = retrieval_config.top_k
        if retrieval_setting and retrieval_setting.top_k is not None:
            top_k = retrieval_setting.top_k
        # Optimization: we can double top_k when rerank model is given, otherwise reranking will be weak.
        if retrieval_config.enable_rerank:
            reranker_top_k = top_k
            top_k = 2 * top_k
        similarity_threshold = retrieval_config.similarity_threshold
        if retrieval_setting and retrieval_setting.score_threshold is not None:
            similarity_threshold = retrieval_setting.score_threshold

        vector_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            doc_ids=document_ids,
            query_str=query,
            mode=query_mode,
            alpha=retrieval_config.vector_weight,
        )

        query_result = await vector_store.aquery(vector_query)
        if retrieval_config.enable_rerank:
            raranker_model = reranker_provider.get_reranker_model(
                retrieval_config.rerank_model
            )
            query_result = await raranker_model.vector_store_rerank(
                query=query,
                result=query_result,
                top_n=reranker_top_k)

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
