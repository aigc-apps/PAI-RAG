import os
from typing import List

from common.encrypt_utils import encrypt_key
from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection
from common.knowledgebase.vectordb.elastic import ElasticsearchConnection
from common.knowledgebase.vectordb.hologres import HologresConnection
from common.knowledgebase.vectordb.opensearch import OpensearchConnection
from common.knowledgebase.vectordb.tablestore import TablestoreConnection
from common.knowledgebase.vectordb.local import LocalConnection
from common.knowledgebase.vectordb.milvus import MilvusConnection
from common.knowledgebase.vectordb.postgres import PostgresqlConnection
from loguru import logger


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


HOLOGRES_HOST_KEYS = ["HOLOGRES_HOST", "PAIRAG_RAG__INDEX__VECTOR_STORE__host"]
HOLOGRES_PORT_KEYS = ["HOLOGRES_PORT", "PAIRAG_RAG__INDEX__VECTOR_STORE__port"]
HOLOGRES_DATABASE_KEYS = ["HOLOGRES_DATABASE", "PAIRAG_RAG__INDEX__VECTOR_STORE__database"]
HOLOGRES_USER_KEYS = ["HOLOGRES_USER", "PAIRAG_RAG__INDEX__VECTOR_STORE__username"]
HOLOGRES_PASSWORD_KEYS = ["HOLOGRES_PASSWORD", "PAIRAG_RAG__INDEX__VECTOR_STORE__password"]


OPENSEARCH_ENDPOINT_KEYS = ["OPENSEARCH_ENDPOINT", "PAIRAG_RAG__INDEX__VECTOR_STORE__endpoint"]
OPENSEARCH_INSTANCE_ID_KEYS = ["OPENSEARCH_INSTANCE_ID", "PAIRAG_RAG__INDEX__VECTOR_STORE__instance_id"]
OPENSEARCH_USERNAME_KEYS = ["OPENSEARCH_USERNAME", "PAIRAG_RAG__INDEX__VECTOR_STORE__username"]
OPENSEARCH_PASSWORD_KEYS = ["OPENSEARCH_PASSWORD", "PAIRAG_RAG__INDEX__VECTOR_STORE__password"]


TABLESTORE_ENDPOINT_KEYS = ["TABLESTORE_ENDPOINT", "PAIRAG_RAG__INDEX__VECTOR_STORE__endpoint"]
TABLESTORE_INSTANCE_NAME_KEYS = ["TABLESTORE_INSTANCE_NAME", "PAIRAG_RAG__INDEX__VECTOR_STORE__instance_name"]
TABLESTORE_ACCESS_KEY_ID_KEYS = ["TABLESTORE_ACCESS_KEY_ID", "PAIRAG_RAG__INDEX__VECTOR_STORE__access_key_id"]
TABLESTORE_ACCESS_KEY_SECRET_KEYS = ["TABLESTORE_ACCESS_KEY_SECRET", "PAIRAG_RAG__INDEX__VECTOR_STORE__access_key_secret"]



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
        return ElasticsearchConnection(endpoint=es_url, user=es_user, encrypted_password=encrypt_key(es_password))
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
            encrypted_password=encrypt_key(password),
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
            encrypted_password=encrypt_key(password),
            database=database,
        )
    elif vector_db_type == VectorDbType.HOLOGRES:
        host = get_value_from_multiple_envs(HOLOGRES_HOST_KEYS)
        port = get_value_from_multiple_envs(HOLOGRES_PORT_KEYS)
        user = get_value_from_multiple_envs(HOLOGRES_USER_KEYS)
        password = get_value_from_multiple_envs(HOLOGRES_PASSWORD_KEYS)
        database = get_value_from_multiple_envs(HOLOGRES_DATABASE_KEYS)

        assert host, "Hologres host不能为空。"
        assert user, "Hologres user不能为空。"
        assert password, "Hologres password不能为空。"
        assert database, "Hologres database不能为空。"

        logger.info(f"Created HologresConnection with host: {host} port: {port}, database: {database}.")
        return HologresConnection(
            host=host,
            port=port,
            user=user,
            encrypted_password=encrypt_key(password),
            database=database,
        )
    elif vector_db_type == VectorDbType.OPENSEARCH:
        endpoint = get_value_from_multiple_envs(OPENSEARCH_ENDPOINT_KEYS)
        instance_id = get_value_from_multiple_envs(OPENSEARCH_INSTANCE_ID_KEYS)
        username = get_value_from_multiple_envs(OPENSEARCH_USERNAME_KEYS)
        password = get_value_from_multiple_envs(OPENSEARCH_PASSWORD_KEYS)

        assert endpoint, "Opensearch endpoint 不能为空。"
        assert instance_id, "Opensearch instance_id 不能为空。"
        assert username, "Opensearch username 不能为空。"
        assert password, "Opensearch password 不能为空。"

        logger.info(f"Created OpensearchConnection with endpoint: {endpoint} instance_id: {instance_id}.")

        return OpensearchConnection(
            endpoint=endpoint,
            instance_id=instance_id,
            username=username,
            password=encrypt_key(password),
        )
    elif vector_db_type == VectorDbType.TABLESTORE:
        endpoint = get_value_from_multiple_envs(TABLESTORE_ENDPOINT_KEYS)
        instance_name = get_value_from_multiple_envs(TABLESTORE_INSTANCE_NAME_KEYS)
        ak = get_value_from_multiple_envs(TABLESTORE_ACCESS_KEY_ID_KEYS)
        sk = get_value_from_multiple_envs(TABLESTORE_ACCESS_KEY_SECRET_KEYS)

        assert endpoint, "Tablestore endpoint 不能为空。"
        assert instance_name, "Tablestore instance_name 不能为空"

        return TablestoreConnection(
            endpoint=endpoint,
            instance_name=instance_name,
            ak=ak or "",
            encrypted_sk=encrypt_key(sk) or "",
        )
    elif vector_db_type == VectorDbType.LOCAL:
        logger.info("Created local vector db connection.")
        return LocalConnection()
    else:
        logger.info(f"Unknown vector db type: {vector_db_type}. Using local vector db.")
        return LocalConnection()
