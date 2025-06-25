from functools import partial
import json
from typing import List
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.tools import FunctionTool

from pairag.db.models.knowledgebase.knowledgebase import KnowledgebaseEntity
from pairag.common.knowledgebase.types import VectorIndexRetrievalType
from pairag.mcp.tools.knowledgebase.vector_connection import (
    create_vector_db_connection_from_env,
    create_vector_store,
)
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider
from pairag.mcp.providers.embedding_provider import embedding_provider
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

    def create_vector_index_from_knowledgebase(
        self,
        knowledgebase: KnowledgebaseEntity,
    ) -> VectorStoreIndex:
        embedding_config = embedding_provider.get_embedding_config(
            knowledgebase.embedding_model
        )
        embed_model = embedding_provider.get_embedding_model(
            knowledgebase.embedding_model
        )
        vector_store = create_vector_store(
            knowledgebase.name, embedding_config.dimension, self.vector_connection
        )
        return VectorStoreIndex(vector_store=vector_store, embed_model=embed_model)

    async def ainsert_nodes(
        self,
        nodes: List[BaseNode],
        knowledgebase_name: str,
    ):
        logger.info(
            f"Starting to insert {len(nodes)} into knowledgebase {knowledgebase_name}."
        )
        knowledgebase = knowledgebase_provider.get_knowledgebase(knowledgebase_name)
        vector_index = self.create_vector_index_from_knowledgebase(knowledgebase)
        vector_index.ainsert_nodes(nodes)
        logger.info(
            f"Finished inserting {len(nodes)} into knowledgebase {knowledgebase_name}."
        )
        return

    async def aquery(
        self,
        query_str: str,
        knowledgebase_name: str,
    ) -> List[dict]:
        logger.info(
            f"Starting to query knowledgebase {knowledgebase_name} with query: {query_str}."
        )
        knowledgebase = knowledgebase_provider.get_knowledgebase(knowledgebase_name)
        vector_index = self.create_vector_index_from_knowledgebase(knowledgebase)
        query_model = retrieval_type_to_search_mode(
            knowledgebase.retrieval_config.retrieval_mode
        )
        retriever = vector_index.as_retriever(
            similarity_top_k=knowledgebase.retrieval_config.top_k,
            vector_store_query_mode=query_model,
            alpha=knowledgebase.retrieval_config.vector_weight,
        )
        scored_nodes = await retriever.aretrieve(str_or_query_bundle=query_str)
        logger.info(f"Retrieved {len(scored_nodes)} nodes from vector index.")

        if knowledgebase.retrieval_config.rerank_model:
            # TODO 1: Get reranker from reranker_provider and rerank results.
            # TODO 2: Maybe we can double top_k when rerank model is given, otherwise reranking will be weak.
            pass

        result_nodes = [
            node.to_dict()
            for node in scored_nodes
            if node.score >= knowledgebase.retrieval_config.similarity_threshold
        ]
        return result_nodes


knowledgebase_client = PaiKnowledgebaseClient()


async def aget_knowledgebase_result(query: str, knowledgebase_name):
    """Get aliyun search tool"""
    res = await knowledgebase_client.aquery(
        query_str=query, knowledgebase_name=knowledgebase_name
    )
    return json.dumps(res, ensure_ascii=False)


async def aget_knowledgebase_tool(knowledgebase_name: str):
    knowledgebase = knowledgebase_provider.get_knowledgebase(knowledgebase_name)
    aquery_knowledgebase_func = partial(
        aget_knowledgebase_result, knowledgebase_name=knowledgebase_name
    )
    search_knowledgebase_tool = FunctionTool.from_defaults(
        async_fn=aquery_knowledgebase_func,
        name="search-knowledgebase",
        description=f"从知识库中搜索给定查询的最新内容。知识库描述: {knowledgebase.description}",
    )
    return search_knowledgebase_tool
