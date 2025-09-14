from functools import partial
import traceback
from typing import Any, List, Optional
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQuery, MetadataFilters, MetadataFilter, FilterCondition, FilterOperator
from llama_index.core.tools import FunctionTool

from common.chat.models import RetrievalSetting
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.knowledgebase import KbEntity, RetrievalConfig, ChunkConfig
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
from pairag.file.models.file_item import FileItem
from pairag.file.nodeparsers.file_parser import FileParser
from pairag.file.utils.image_caption_tool import ImageCaptionTool
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
from pairag.file.store.file_store_helper import file_store
from llama_index.core.schema import NodeWithScore
from loguru import logger
import re
import json
from rag.file_existence_guard import FileExistenceGuard, require_file_exists
from typing import Annotated

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
        chunk_config = ChunkConfig.model_validate(knowledgebase.chunk_config)
        file_parser = FileParser(
            file_store=file_store,
            image_caption_tool=image_caption_tool,
            chunk_config=chunk_config,
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
        # 创建守护器, 初始检查
        guard = FileExistenceGuard(file_id)
        if not await guard.check_exists():
            return

        file_entity: KbFileEntity = await read_file_from_db(file_id=file_id)

        logger.info(f"[WORKER] retrieved file {file_entity} for {file_id}.")
        try:
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
        except Exception as ex:
            logger.error(f"处理文件失败：{traceback.format_exc()}")
            await update_file_status_async(file_id=file_id, status=FileStatus.failed, failed_reason=str(ex), is_attachment=is_attachment)


        @require_file_exists()
        async def parse_file(guard_instance):
            multimodal_llm = await get_multimodal_llm_from_db()
            file_parser = self.create_file_parser(knowledgebase, multimodal_llm=multimodal_llm)
            documents, nodes = file_parser.parse(file_item, is_attachment=is_attachment)
            return documents, nodes

        @require_file_exists()
        async def save_chunks(guard_instance, nodes):
            old_chunk_ids, new_chunk_ids = await save_chunks_to_db_async(
                kb_id=kb_id, file_id=file_item.id, chunk_nodes=nodes
            )
            return old_chunk_ids, new_chunk_ids

        @require_file_exists()
        async def update_vector_store(guard_instance):
            logger.info(f"Starting to insert {len(nodes)} into knowledgebase {kb_id}.")
            embed_model:BaseEmbedding = await get_embedding_from_db(model_id=knowledgebase.embedding_model)

            vector_store = self.create_vector_store_from_knowledgebase(knowledgebase, embed_model=embed_model)
            if old_chunk_ids:
                await vector_store.adelete_nodes(node_ids=old_chunk_ids)
                logger.info(f"Removed {len(old_chunk_ids)} from vector store.")

            texts_to_embed = [f"{node.text}\n\nfile_name: {node.metadata['file_name']}\n\nchapter_name: {node.metadata['chapter_name']}" for node in nodes]
            embeddings = await embed_model.aget_text_embedding_batch(texts_to_embed, show_progress=True)
            for i in range(len(nodes)):
                nodes[i].embedding = embeddings[i]

            await vector_store.async_add(nodes)

            logger.info(f"Finished inserting {len(nodes)} into knowledgebase {kb_id}.")
            return True

        try:
            result = await parse_file(guard)
            if result:
                documents, nodes = result
                old_chunk_ids, new_chunk_ids = await save_chunks(guard, nodes)
                await update_file_status_async(
                    file_id=file_item.id, status=FileStatus.persisting,
                    is_attachment=is_attachment, documents=documents
                )

                await update_vector_store(guard)

                if await guard.check_exists():  # 最后检查
                    await update_chunk_status_async(chunk_ids=new_chunk_ids, status=ChunkStatus.succeeded)
                    await update_file_status_async(
                        file_id=file_item.id, status=FileStatus.succeeded, is_attachment=is_attachment
                    )
                    logger.info(
                        f"Finished adding file {file_item.file_name} to knowledgebase {kb_id}."
                    )
        except Exception as e:
            if await guard.check_exists():  # 只有文件还存在时才更新状态
                await update_file_status_async(
                    file_id=file_item.id, status=FileStatus.failed, is_attachment=is_attachment, failed_reason=str(e),
                )
            logger.error(f"Error processing file: {traceback.format_exc()}")

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
        await vector_store.adelete_nodes(node_ids=node_ids)
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
        embed_model:BaseEmbedding = embedding_provider.get_embedding_model(
            knowledgebase.embedding_model
        )
        texts_to_embed = [f"{node.text}\n\nfile_name: {node.metadata['file_name']}\n\nchapter_name: {node.metadata['chapter_name']}" for node in nodes]
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


async def aget_knowledgebase_result(query: str, kb_id: str, user_id: str="anonymous") -> str:
    """Get aliyun search tool"""
    logger.info(f"Searching knowledgebase with kb {kb_id} and user {user_id}.")
    result_nodes = await kb_client.aquery(query=query, knowledge_id=kb_id, user_id=user_id)
    records = []
    for score_node in result_nodes:
        images = []
        origin_text = score_node.node.get_content()
        pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"'
        matches = re.findall(pattern, origin_text)
        images = [{"url": src, "desc": alt} for src, alt in matches]
        records.append({
            "text": score_node.node.get_content(),
            "metadata": {
                "file_name": score_node.node.metadata.get("file_name", ""),
                "file_url": file_store.get_url(score_node.node.metadata.get("file_path", "")),
                "file_source": score_node.node.metadata.get("file_source", "")
            },
            "score": score_node.score,
            "images": images
        })
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
