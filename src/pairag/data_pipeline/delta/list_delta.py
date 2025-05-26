# 通过向量数据库中的record比对确定delta信息
from pairag.data_pipeline.delta.milvus import (
    list_docs_in_milvus_collection,
)
from pairag.data_pipeline.utils.vectordb_utils import get_vector_store
from llama_index.vector_stores.milvus import MilvusVectorStore

from loguru import logger


"""
获取PAI-RAG服务索引中的知识库文件列表
"""


def list_files_from_rag_service(
    rag_endpoint: str,
    rag_api_key: str,
    knowledgebase: str,
    embed_dims: int,
    oss_path_prefix: str,
):
    vector_store = get_vector_store(
        rag_endpoint=rag_endpoint,
        rag_api_key=rag_api_key,
        knowledgebase=knowledgebase,
        embed_dims=embed_dims,
    )
    if isinstance(vector_store, MilvusVectorStore):
        return list_docs_in_milvus_collection(
            collection=vector_store._collection,
            oss_path_prefix=oss_path_prefix,
        )
    else:
        logger.error("Only support Milvus vector store for now.")
        raise NotImplementedError
