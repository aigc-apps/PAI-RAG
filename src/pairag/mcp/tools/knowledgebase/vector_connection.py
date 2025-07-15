from enum import Enum
import os
from typing import List
from pydantic import BaseModel, ConfigDict
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from pairag.common.knowledgebase.constants import DEFAULT_KNOWLEDGEBASE_PATH
from loguru import logger

from pairag.mcp.tools.knowledgebase.faiss_vector_store import FaissVectorStore


class VectorDbType(str, Enum):
    OPENSEARCH = "opensearch"
    ELASTICSEARCH = "elasticsearch"
    ANALYTICDB = "analyticdb"
    POSTGRESQL = "postgresql"
    HOLOGRES = "hologres"
    TABLESTORE = "tablestore"
    MILVUS = "milvus"
    FAISS = "faiss"
    DASHVECTOR = "dashvector"


class BaseVectorDbConnection(BaseModel):
    type: VectorDbType

    model_config = ConfigDict(coerce_numbers_to_str=True)


class ElasticSearchConnection(BaseVectorDbConnection):
    type: VectorDbType = VectorDbType.ELASTICSEARCH
    url: str
    user: str = "elastic"
    password: str


class MilvusConnection(BaseVectorDbConnection):
    type: VectorDbType = VectorDbType.MILVUS
    host: str
    port: int = 19530
    database: str = "default"
    user: str = "root"
    password: str = ""


class FaissConnection(BaseVectorDbConnection):
    type: VectorDbType = VectorDbType.FAISS


def get_value_from_multiple_envs(env_names: List[str], default=None):
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value
    return default


VECTORDB_TYPE_KEYS = ["VECTOR_DB_TYPE", "PAIRAG_RAG__INDEX__VECTOR_STORE__type"]
ELASTICSEARCH_URL_KEYS = [
    "ELASTICSEARCH_URL",
    "PAIRAG_RAG__INDEX__VECTOR_STORE__es_url",
]
ELASTICSEARCH_USER_KEYS = [
    "ELASTICSEARCH_USER",
    "PAIRAG_RAG__INDEX__VECTOR_STORE__es_user",
]
ELASTICSEARCH_PASSWORD_KEYS = [
    "ELASTICSEARCH_PASSWORD",
    "PAIRAG_RAG__INDEX__VECTOR_STORE__es_password",
]

MILVUS_HOST_KEYS = ["MILVUS_HOST", "PAIRAG_RAG__INDEX__VECTOR_STORE__host"]
MILVUS_PORT_KEYS = ["MILVUS_PORT", "PAIRAG_RAG__INDEX__VECTOR_STORE__port"]
MILVUS_USER_KEYS = ["MILVUS_USER", "PAIRAG_RAG__INDEX__VECTOR_STORE__user"]
MILVUS_PASSWORD_KEYS = ["MILVUS_PASSWORD", "PAIRAG_RAG__INDEX__VECTOR_STORE__password"]
MILVUS_DATABASE_KEYS = [
    "MILVUS_DATABASE",
    "PAIRAG_RAG__INDEX__VECTOR_STORE__database",
]


def create_vector_db_connection_from_env() -> BaseVectorDbConnection:
    vector_db_type = get_value_from_multiple_envs(
        VECTORDB_TYPE_KEYS, default="faiss"
    ).lower()

    if vector_db_type == VectorDbType.ELASTICSEARCH:
        es_url = get_value_from_multiple_envs(ELASTICSEARCH_URL_KEYS)
        es_user = get_value_from_multiple_envs(ELASTICSEARCH_USER_KEYS)
        es_password = get_value_from_multiple_envs(ELASTICSEARCH_PASSWORD_KEYS)
        return ElasticSearchConnection(url=es_url, user=es_user, password=es_password)

    elif vector_db_type == VectorDbType.MILVUS:
        host = get_value_from_multiple_envs(MILVUS_HOST_KEYS)
        port = get_value_from_multiple_envs(MILVUS_PORT_KEYS)
        user = get_value_from_multiple_envs(MILVUS_USER_KEYS)
        password = get_value_from_multiple_envs(MILVUS_PASSWORD_KEYS)
        database = get_value_from_multiple_envs(MILVUS_DATABASE_KEYS)
        return MilvusConnection(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )

    elif vector_db_type == VectorDbType.FAISS:
        return FaissConnection()


def create_vector_store(
    kb_id: str,
    dimension: int,
    vector_db_connection: BaseVectorDbConnection,
) -> BasePydanticVectorStore:
    if isinstance(vector_db_connection, MilvusConnection):
        milvus_url = (
            f"http://{vector_db_connection.host.strip('/')}:{vector_db_connection.port}"
        )
        token = f"{vector_db_connection.user}:{vector_db_connection.password}"

        logger.info(f"Creating Milvus vector store for {kb_id} with url: {milvus_url}.")
        return MilvusVectorStore(
            uri=milvus_url,
            token=token,
            collection_name=kb_id,
            dim=dimension,
            enable_sparse=True,
            similarity_metric="cosine",
            hybrid_ranker="WeightedRanker",
        )
    elif isinstance(vector_db_connection, ElasticSearchConnection):
        logger.info(
            f"Creating ElasticsearchStore for {kb_id} with url {vector_db_connection.url}."
        )
        return ElasticsearchStore(
            es_url=vector_db_connection.url,
            index_name=kb_id,
            es_user=vector_db_connection.user,
            es_password=vector_db_connection.password,
            dim=dimension,
        )
    elif isinstance(vector_db_connection, FaissConnection):
        persist_dir = os.path.join(DEFAULT_KNOWLEDGEBASE_PATH, kb_id, ".index")
        logger.info(f"Creating FaissVectorStore for {kb_id} with path {persist_dir}.")

        return FaissVectorStore.from_persist_dir(
            persist_dir=persist_dir,
            dimension=dimension,
        )
    else:
        raise ValueError(f"Unknown vector_db_connection: {vector_db_connection}.")
