from enum import Enum
import os
from typing import List
from urllib.parse import quote_plus
from pydantic import BaseModel, ConfigDict
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from loguru import logger

from rag.vector_store.local_chroma_service import DEFAULT_CHROMA_PORT
from rag.vector_store.local import LocalChromaVectorStore
from rag.vector_store.elasticsearch import ElasticsearchStore
from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction


class VectorDbType(str, Enum):
    OPENSEARCH = "opensearch"
    ELASTICSEARCH = "elasticsearch"
    ANALYTICDB = "analyticdb"
    POSTGRESQL = "postgresql"
    HOLOGRES = "hologres"
    TABLESTORE = "tablestore"
    MILVUS = "milvus"
    DASHVECTOR = "dashvector"
    LOCAL = "local"


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


class PostgresqlConnection(BaseVectorDbConnection):
    type: VectorDbType = VectorDbType.POSTGRESQL
    host: str
    port: int = 5432
    database: str
    user: str
    password: str


class LocalConnection(BaseVectorDbConnection):
    type: VectorDbType = VectorDbType.LOCAL


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
MILVUS_SPARSE_EMBEDDING_TYPE_KEYS = [
    "MILVUS_SAPARSE_TYPE",
]

POSTGRES_HOST_KEYS = ["POSTGRES_HOST", "PAIRAG_RAG__INDEX__VECTOR_STORE__host"]
POSTGRES_PORT_KEYS = ["POSTGRES_PORT", "PAIRAG_RAG__INDEX__VECTOR_STORE__port"]
POSTGRES_DATABASE_KEYS = ["POSTGRES_DATABASE", "PAIRAG_RAG__INDEX__VECTOR_STORE__database"]
POSTGRES_USER_KEYS = ["POSTGRES_USER", "PAIRAG_RAG__INDEX__VECTOR_STORE__username"]
POSTGRES_PASSWORD_KEYS = ["POSTGRES_PASSWORD", "PAIRAG_RAG__INDEX__VECTOR_STORE__password"]


def create_vector_db_connection_from_env() -> BaseVectorDbConnection:
    vector_db_type = get_value_from_multiple_envs(
        VECTORDB_TYPE_KEYS, default="local"
    ).lower()

    if vector_db_type == VectorDbType.ELASTICSEARCH:
        es_url = get_value_from_multiple_envs(ELASTICSEARCH_URL_KEYS)
        es_user = get_value_from_multiple_envs(ELASTICSEARCH_USER_KEYS)
        es_password = get_value_from_multiple_envs(ELASTICSEARCH_PASSWORD_KEYS)
        assert es_url, "elastic search url不能为空。"
        assert es_user, "elastic search user不能为空。"
        assert es_password, "elastic password不能为空。"

        logger.info(f"Created ElasticSearchConnection with url: {es_url}.")
        return ElasticSearchConnection(url=es_url, user=es_user, password=es_password)
    elif vector_db_type == VectorDbType.MILVUS:
        host = get_value_from_multiple_envs(MILVUS_HOST_KEYS)
        port = get_value_from_multiple_envs(MILVUS_PORT_KEYS)
        user = get_value_from_multiple_envs(MILVUS_USER_KEYS)
        password = get_value_from_multiple_envs(MILVUS_PASSWORD_KEYS)
        database = get_value_from_multiple_envs(MILVUS_DATABASE_KEYS)

        assert host, "Milvus host不能为空。"
        assert user, "Milvus user不能为空。"
        assert password, "Milvus password不能为空。"
        assert database, "Milvus database不能为空。"

        logger.info(f"Created MilvusConnection with host: {host} port: {port}, database: {database}.")
        return MilvusConnection(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
    elif vector_db_type == VectorDbType.POSTGRESQL:
        host = get_value_from_multiple_envs(POSTGRES_HOST_KEYS)
        port = get_value_from_multiple_envs(POSTGRES_PORT_KEYS, 5432)
        user = get_value_from_multiple_envs(POSTGRES_USER_KEYS)
        password = get_value_from_multiple_envs(POSTGRES_PASSWORD_KEYS)
        database = get_value_from_multiple_envs(POSTGRES_DATABASE_KEYS)

        assert host, "Postgres host不能为空。"
        assert user, "Postgres user不能为空。"
        assert password, "Postgres password不能为空。"
        assert database, "Postgres database不能为空。"

        logger.info(f"Created PostgresqlConnection with host:{host} port:{port} database:{database} user:{user}.")
        return PostgresqlConnection(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
    elif vector_db_type == VectorDbType.LOCAL:
        logger.info("Created local vector db connection.")
        return LocalConnection()
    else:
        raise ValueError(f"Unknown vector db type: {vector_db_type}")



def create_vector_store(
    kb_id: str,
    dimension: int,
    vector_db_connection: BaseVectorDbConnection,
) -> BasePydanticVectorStore:
    if isinstance(vector_db_connection, MilvusConnection):
        milvus_url = (
            f"http://{vector_db_connection.host.strip('/')}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        token = f"{vector_db_connection.user}:{vector_db_connection.password}"
        sparse_embedding_function = BM25BuiltInFunction()

        logger.info(f"Creating Milvus vector store for {kb_id} with url: {milvus_url}.")
        return MilvusVectorStore(
            uri=milvus_url,
            token=token,
            collection_name=kb_id,
            dim=dimension,
            enable_sparse=True,
            similarity_metric="cosine",
            hybrid_ranker="WeightedRanker",
            # TODO: add weighted reranker config
            hybrid_ranker_params={"weights": [0.5, 0.5]},
            sparse_embedding_function=sparse_embedding_function,
            db_name=vector_db_connection.database,
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
            retrieval_strategy=AsyncDenseVectorStrategy(
                hybrid=True, rrf={"window_size": 50}
            ),

        )
    elif isinstance(vector_db_connection, PostgresqlConnection):
        logger.info(
            f"Creating PostgresqlStore for {kb_id} with url {vector_db_connection.host} {vector_db_connection.database}."
        )
        conn_str = (
            f"postgresql+psycopg2://{vector_db_connection.user}:{quote_plus(vector_db_connection.password)}@{vector_db_connection.host}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        async_conn_str = (
            f"postgresql+asyncpg://{vector_db_connection.user}:{quote_plus(vector_db_connection.password)}@{vector_db_connection.host}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        return PGVectorStore(
            connection_string=conn_str,
            async_connection_string=async_conn_str,
            schema_name="public",
            table_name=kb_id,
            embed_dim=dimension,
            hybrid_search=True,
            text_search_config="jiebacfg",
        )
    elif isinstance(vector_db_connection, LocalConnection):
        logger.info(f"Creating LocalVectorStore for {kb_id} with port {DEFAULT_CHROMA_PORT}.")
        return LocalChromaVectorStore(
            collection_name=kb_id,
            host="localhost",
            port=DEFAULT_CHROMA_PORT,
        )
    else:
        raise ValueError(f"Unknown vector_db_connection: {vector_db_connection}.")


def is_docid_filter_supported(vector_store: BasePydanticVectorStore) -> bool:
    """
    Check if the vector store supports filtering by docid.

    Args:
        vector_store (BasePydanticVectorStore): The vector store to check.

    Returns:
        bool: True if the vector store supports filtering by docid, False otherwise.
    """
    if isinstance(vector_store, PGVectorStore):
        return False
    else:
        return True
