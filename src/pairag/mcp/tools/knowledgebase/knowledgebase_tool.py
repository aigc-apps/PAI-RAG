from functools import partial
from typing import List
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.tools import FunctionTool

from pairag.db.models.knowledgebase.file import KbFileEntity
from pairag.db.models.knowledgebase.knowledgebase import KbEntity, RetrievalConfig
from pairag.common.knowledgebase.types import (
    ChunkStatus,
    FileStatus,
    VectorIndexRetrievalType,
)
from pairag.mcp.providers.chunk_helper import (
    read_file_from_db,
    save_chunks_to_db_async,
    update_chunk_status_async,
    update_file_status_async,
)
from pairag.mcp.rag.file.models.file_item import FileItem
from pairag.mcp.rag.file_parser import FileParser
from pairag.mcp.rag.image_caption_tool import ImageCaptionTool
from pairag.mcp.tools.knowledgebase.vector_connection import (
    create_vector_db_connection_from_env,
    create_vector_store,
)
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider
from pairag.mcp.providers.embedding_provider import embedding_provider
from pairag.mcp.providers.llm_provider import llm_provider
from pairag.mcp.rag.file.store.file_store_helper import file_store
from llama_index.core.schema import NodeWithScore
from loguru import logger


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
        self.vector_index_cache = {}

    def create_vector_index_from_knowledgebase(
        self,
        knowledgebase: KbEntity,
    ) -> VectorStoreIndex:
        # TODO: 检查配置是否变化
        if knowledgebase.id not in self.vector_index_cache:
            embed_model = embedding_provider.get_embedding_model(
                knowledgebase.embedding_model
            )
            vector_dimension = len(embed_model.get_text_embedding("0"))
            vector_store = create_vector_store(
                knowledgebase.id, vector_dimension, self.vector_connection
            )
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=embed_model
            )
            self.vector_index_cache[knowledgebase.id] = vector_index
            logger.info(f"Created vector index for knowledgebase {knowledgebase.id}.")

        return self.vector_index_cache[knowledgebase.id]

    def create_file_parser(self, knowledgebase: KbEntity):
        multimodal_llm = llm_provider.get_multimodal_llm()
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
    async def process_file_async(
        self,
        file_id: str,
    ):
        file_entity: KbFileEntity = await read_file_from_db(file_id)
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
        knowledgebase = await knowledgebase_provider.aget_knowledgebase(
            knowledgebase_id=kb_id
        )
        logger.info(
            f"Start to add file {file_item.file_name} to knowledgebase {kb_id}."
        )
        await update_file_status_async(file_id=file_item.id, status=FileStatus.parsing)

        try:
            file_parser = self.create_file_parser(knowledgebase)
            nodes = file_parser.parse(file_item)

            old_chunk_ids, new_chunk_ids = await save_chunks_to_db_async(
                kb_id=kb_id, file_id=file_item.id, chunk_nodes=nodes
            )
            await update_file_status_async(
                file_id=file_item.id, status=FileStatus.persisting
            )

            logger.info(f"Starting to insert {len(nodes)} into knowledgebase {kb_id}.")
            vector_index = self.create_vector_index_from_knowledgebase(knowledgebase)
            if old_chunk_ids:
                vector_index.delete_nodes(node_ids=old_chunk_ids)
                logger.info(f"Removed {len(old_chunk_ids)} from vector store.")
            await vector_index.ainsert_nodes(nodes)
            logger.info(f"Inserted {len(old_chunk_ids)} into vector store.")
            logger.info(f"Finished inserting {len(nodes)} into knowledgebase {kb_id}.")
            await update_chunk_status_async(
                chunk_ids=new_chunk_ids, status=ChunkStatus.succeeded
            )
            await update_file_status_async(
                file_id=file_item.id, status=FileStatus.succeeded
            )

            logger.info(
                f"Finished adding file {file_item.file_name} to knowledgebase {kb_id}."
            )
        except Exception as ex:
            logger.exception(
                f"Error adding file {file_item.file_name} to knowledgebase {kb_id}. {ex}"
            )
            await update_file_status_async(
                file_id=file_item.id, status=FileStatus.failed
            )

    async def adelete_kb(
        self,
        kb_id: str,
    ):
        if kb_id in self.vector_index_cache:
            del self.vector_index_cache[kb_id]

    async def adelete_chunks_from_vectordb(
        self,
        kb_id: str,
        node_ids: List[str],
    ):
        knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
        vector_index = self.create_vector_index_from_knowledgebase(knowledgebase)
        vector_index.delete_nodes(node_ids=node_ids)
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
        vector_index = self.create_vector_index_from_knowledgebase(knowledgebase)
        await vector_index.ainsert_nodes(nodes)
        logger.info(f"Finished inserting {len(nodes)} into vector store.")

    async def aquery(
        self,
        query_str: str,
        kb_id: str,
    ) -> List[NodeWithScore]:
        logger.info(f"Starting to query knowledgebase {kb_id} with query: {query_str}.")
        knowledgebase = await knowledgebase_provider.aget_knowledgebase(kb_id)
        retrieval_config = RetrievalConfig.model_validate(
            knowledgebase.retrieval_config
        )
        vector_index = self.create_vector_index_from_knowledgebase(knowledgebase)
        query_model = retrieval_type_to_search_mode(retrieval_config.retrieval_mode)
        retriever = vector_index.as_retriever(
            similarity_top_k=retrieval_config.top_k,
            vector_store_query_mode=query_model,
            alpha=retrieval_config.vector_weight,
        )
        scored_nodes = await retriever.aretrieve(str_or_query_bundle=query_str)
        for node in scored_nodes:
            images = node.metadata.get("images", [])
            if images:
                origin_text = node.node.text
                for image_file in images:
                    image_url = file_store.get_url(image_file)
                    origin_text = origin_text.replace(image_file, image_url)
                node.node.text = origin_text
        logger.info(f"Retrieved {len(scored_nodes)} nodes from vector index.")

        if retrieval_config.rerank_model:
            # TODO 1: Get reranker from reranker_provider and rerank results.
            # TODO 2: Maybe we can double top_k when rerank model is given, otherwise reranking will be weak.
            pass

        result_nodes = [
            node
            for node in scored_nodes
            if node.score >= retrieval_config.similarity_threshold
        ]
        return result_nodes


kb_client = PaiKnowledgebaseClient()


def get_node_content(i: int, score_node: NodeWithScore):
    text = f"""
    ## chunk {i+1}
    file_name: {score_node.node.metadata.get("file_name", "")}
    chunk_content: {score_node.node.text}
    """

    return text


async def aget_knowledgebase_result(query: str, kb_id: str) -> str:
    """Get aliyun search tool"""
    result_nodes = await kb_client.aquery(query_str=query, kb_id=kb_id)

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
