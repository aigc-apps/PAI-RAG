# 通过向量数据库中的record比对确定delta信息
from typing import Dict
from langstudio.rag.consts import StoreType
from langstudio.rag.index_manifest import IndexManifest
import os
from pai_rag.data_ingestion.delta.milvus import (
    list_docs_in_milvus_collection,
    list_docs_in_milvus_from_langstudio_index_manifest,
)
from pai_rag.data_ingestion.delta.models import DocItem
from pai_rag.data_ingestion.utils.vectordb_utils import get_vector_store
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


"""
获取Langstudio索引的知识库文件列表
"""


def list_files_from_langstudio_index(
    target_index: str,
    target_index_version: str,
    oss_path_prefix: str,
) -> Dict[str, DocItem]:
    if os.path.exists(target_index):
        index_manifest = IndexManifest.from_path(target_index)
    else:
        index_manifest = IndexManifest.from_registered(
            id_=target_index, version=target_index_version
        )

    if index_manifest.store.type == StoreType.Faiss:
        raise NotImplementedError
    elif index_manifest.store.type == StoreType.Milvus:
        return list_docs_in_milvus_from_langstudio_index_manifest(
            index_manifest, oss_path_prefix
        )
    else:
        raise ValueError(
            f"Unsupported vector database type: {index_manifest.store.type}"
        )
