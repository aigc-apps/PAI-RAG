from functools import partial
import traceback
from typing import Any, List, Optional
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQuery, MetadataFilters, MetadataFilter, FilterCondition, FilterOperator
from llama_index.core.tools import FunctionTool

from common.chat.models import RetrievalSetting
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.knowledgebase import KbEntity, RetrievalConfig
from common.knowledgebase.types import (
    ChunkStatus,
    FileStatus,
    VectorIndexRetrievalType,
)
from db.models.knowledgebase.metadata_filter import MetadataFilteringCondition, query_file_ids_with_metadata_filter
from rag.chunk_helper import (
    get_embedding_from_db,
    get_file_id_source_map,
    get_multimodal_llm_from_db,
    read_file_from_db,
    save_chunks_to_db_async,
    update_chunk_status_async,
    update_file_status_async,
)
from rag.file.models.file_item import FileItem
from rag.file.file_parser import FileParser
from rag.file.image_caption_tool import ImageCaptionTool
from rag.vector_store.vector_connection import (
    create_vector_db_connection_from_env,
    create_vector_store,
    is_docid_filter_supported,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from config.providers.knowledgebase_provider import fetch_knowledgebases_by_id, knowledgebase_provider
from config.providers.embedding_provider import embedding_provider
from config.providers.reranker_provider import reranker_provider
from rag.file.store.file_store_helper import file_store
from llama_index.core.schema import NodeWithScore
from loguru import logger
import re

def retrieval_type_to_search_mode(retrieval_type: VectorIndexRetrievalType):
    if retrieval_type == VectorIndexRetrievalType.fulltext:
        return VectorStoreQueryMode.TEXT_SEARCH
    elif retrieval_type == VectorIndexRetrievalType.hybrid:
        return VectorStoreQueryMode.HYBRID
    else:
        return VectorStoreQueryMode.DEFAULT



class PaiKnowledgebaseClient:
    def __init__(self):
        self.vector_connection = create_vector_db_connection_from_env()
        self.vector_store_cache = {}

    def create_vector_store_from_knowledgebase(
        self,
        knowledgebase: KbEntity,
        embed_model: BaseEmbedding = None,
    ) -> BasePydanticVectorStore:
        # TODO: 检查配置是否变化
        if not embed_model:
            embed_model = embedding_provider.get_embedding_model(knowledgebase.embedding_model)

        dimension = len(embed_model.get_text_embedding("0"))
        kb_key = f"{knowledgebase.id}_{dimension}"
        if kb_key not in self.vector_store_cache:
            vector_store = create_vector_store(
                knowledgebase.id, dimension, self.vector_connection
            )
            self.vector_store_cache[kb_key] = vector_store
            logger.info(f"Created vector index for knowledgebase {kb_key}.")

        return self.vector_store_cache[kb_key]

    def create_file_parser(self, knowledgebase: KbEntity, multimodal_llm: Any = None):
        image_caption_tool = None
        if multimodal_llm:
            image_caption_tool = ImageCaptionTool(multimodal_llm=multimodal_llm)
        file_parser = FileParser(
            file_store=file_store,
            image_caption_tool=image_caption_tool,
            knowledgebase=knowledgebase,
        )
        return file_parser


    # process file item, status -> processing
    # 这里是离线链路，所有的数据直接从db读取，不需要用到provider信息
    async def process_file_async(
        self,
        file_id: str,
        is_attachment: bool = False,
    ):
        logger.info(f"[WORKER] processing file {file_id} in background. Is attachment: {is_attachment}")
        file_entity: KbFileEntity = await read_file_from_db(file_id=file_id)

        logger.info(f"[WORKER] retrieved file {file_entity} for {file_id}.")

        file = file_store.load(file_entity.file_path)
        file_item = FileItem(
            id=file_entity.id,
            file_path=file_entity.file_path,
            file=file,
            kb_id=file_entity.kb_id,
            file_extension=file_entity.file_extension,
            file_name=file_entity.file_name,
            file_md5=file_entity.file_md5,
            file_size=file_entity.file_size,
        )

        kb_id = file_item.kb_id
        knowledgebase: KbEntity = await fetch_knowledgebases_by_id(
            kb_id=kb_id
        )
        logger.info(
            f"Start to add file {file_item.file_name} to knowledgebase {kb_id}."
        )
        await update_file_status_async(file_id=file_item.id, status=FileStatus.parsing, is_attachment=is_attachment)

        try:
            multimodal_llm = await get_multimodal_llm_from_db()
            file_parser = self.create_file_parser(knowledgebase, multimodal_llm=multimodal_llm)
            documents, nodes = file_parser.parse(file_item, is_attachment=is_attachment)

            old_chunk_ids, new_chunk_ids = await save_chunks_to_db_async(
                kb_id=kb_id, file_id=file_item.id, chunk_nodes=nodes
            )

            await update_file_status_async(
                file_id=file_item.id, status=FileStatus.persisting, is_attachment=is_attachment, documents=documents
            )

            logger.info(f"Starting to insert {len(nodes)} into knowledgebase {kb_id}.")

            embed_model:BaseEmbedding = await get_embedding_from_db(model_id=knowledgebase.embedding_model)

            vector_store = self.create_vector_store_from_knowledgebase(knowledgebase, embed_model=embed_model)
            if old_chunk_ids:
                vector_store.delete_nodes(node_ids=old_chunk_ids)
                logger.info(f"Removed {len(old_chunk_ids)} from vector store.")

            texts_to_embed = [f"{node.text}\n\nfile_name: {node.metadata['file_name']}" for node in nodes]
            embeddings = await embed_model.aget_text_embedding_batch(texts_to_embed, show_progress=True)
            for i in range(len(nodes)):
                nodes[i].embedding = embeddings[i]

            await vector_store.async_add(nodes)

            logger.info(f"Finished inserting {len(nodes)} into knowledgebase {kb_id}.")
            await update_chunk_status_async(
                chunk_ids=new_chunk_ids, status=ChunkStatus.succeeded
            )
            await update_file_status_async(
                file_id=file_item.id, status=FileStatus.succeeded, is_attachment=is_attachment
            )

            logger.info(
                f"Finished adding file {file_item.file_name} to knowledgebase {kb_id}."
            )
        except Exception:
            logger.exception(
                f"Error adding file {file_item.file_name} to knowledgebase {kb_id}. {traceback.format_exc()}"
            )
            await update_file_status_async(
                file_id=file_item.id, status=FileStatus.failed, is_attachment=is_attachment
            )

    async def adelete_kb(
        self,
        kb_id: str,
    ):
        # TODO: 是否delete表？
        if kb_id in self.vector_store_cache:
            del self.vector_store_cache[kb_id]


    async def adelete_chunks_from_vectordb(
        self,
        kb_id: str,
        node_ids: List[str],
    ):
        if not node_ids:
            return

        knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
        vector_store = self.create_vector_store_from_knowledgebase(knowledgebase)
        vector_store.delete_nodes(node_ids=node_ids)
        logger.info(
            f"Deleted {len(node_ids)} chunks from {kb_id} vector db successfully."
        )


    async def ainsert_chunks_to_vectordb(
        self,
        kb_id: str,
        nodes,
    ):
        logger.info(f"Starting to insert {len(nodes)} into knowledgebase {kb_id}.")
        knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
        vector_store = self.create_vector_store_from_knowledgebase(knowledgebase)
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
                file_ids.append(node.metadata["doc_id"])
                origin_text = node.text
                pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"'
                matches = re.findall(pattern, origin_text)
                for src, _ in matches:
                    image_url = file_store.get_url(src)
                    origin_text = origin_text.replace(src, image_url)
                node.text = origin_text
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
                origin_text = node.text
                pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"'
                matches = re.findall(pattern, origin_text)
                for src, _ in matches:
                    image_url = file_store.get_url(src)
                    origin_text = origin_text.replace(src, image_url)
                node.text = origin_text
                result_nodes.append(NodeWithScore(node=node, score=query_result.similarities[i]))
        logger.info(f"Retrieved {len(result_nodes)} nodes from vector index.")
        return result_nodes


kb_client = PaiKnowledgebaseClient()


def get_node_content(i: int, node: NodeWithScore):
    text = f"""
    ## chunk {i+1}
    file_name: {node.node.metadata.get("file_name", "")}
    chunk_content: {node.node.text}
    """

    return text


async def aget_knowledgebase_result(query: str, kb_id: str) -> str:
    """Get aliyun search tool"""
    result_nodes = await kb_client.aquery(query=query, knowledge_id=kb_id)

    retrieval_result = "\n---\n".join(
        [get_node_content(i, node) for i, node in enumerate(result_nodes)]
    )
    return retrieval_result


async def aget_knowledgebase_tool(kb_id: str):
    knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
    aquery_knowledgebase_func = partial(aget_knowledgebase_result, kb_id=kb_id)
    search_knowledgebase_tool = FunctionTool.from_defaults(
        async_fn=aquery_knowledgebase_func,
        name="search-knowledgebase",
        description=f"从知识库中搜索给定查询的最新内容。知识库描述: {knowledgebase.description}",
    )
    return search_knowledgebase_tool
