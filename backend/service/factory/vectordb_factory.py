
from common.knowledgebase.types import VectorDbType
from common.knowledgebase.vectordb.base import BaseVectorDbConnection
from common.knowledgebase.vectordb.elastic import ElasticsearchConnection
from common.knowledgebase.vectordb.local import LocalConnection
from common.knowledgebase.vectordb.milvus import MilvusConnection
from common.knowledgebase.vectordb.postgres import PostgresqlConnection
from common.knowledgebase.vectordb.hologres import HologresConnection
from common.knowledgebase.vectordb.tablestore import TablestoreConnection
from common.knowledgebase.vectordb.opensearch import OpensearchConnection
from db.models.vectordb import VectorDbConfig
from urllib.parse import quote_plus
from common.encrypt_utils import decrypt_key
import json
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from loguru import logger
import copy
from rag.vector_store.local_chroma_service import DEFAULT_CHROMA_PORT
from rag.vector_store.local import LocalChromaVectorStore
from rag.vector_store.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.vector_stores.hologres import HologresVectorStore
from llama_index.vector_stores.alibabacloud_opensearch import AlibabaCloudOpenSearchConfig, AlibabaCloudOpenSearchStore
import tablestore
from llama_index.vector_stores.tablestore import TablestoreVectorStore
from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy

from utils.lru_cache import LruCache
from rag.vector_store.vector_connection import cleanup_vector_store


vectordb_cache = LruCache(max_size=200, on_delete_func=cleanup_vector_store)


def get_vectordb_cache_key(kb_id: str, dimension: int, vector_config: VectorDbConfig, table_name: str):
    vec_config_dict = copy.copy(vector_config.config)
    if "encrypted_password" in vec_config_dict:
        vec_config_dict["password"] = decrypt_key(vec_config_dict["encrypted_password"])
        vec_config_dict.pop("encrypted_password")
    if "encrypted_sk" in vec_config_dict:
        vec_config_dict["sk"] = decrypt_key(vec_config_dict["encrypted_sk"])
        vec_config_dict.pop("encrypted_sk")

    vector_str = json.dumps(vec_config_dict, sort_keys=True, ensure_ascii=False)
    key_str = f"{table_name}--{kb_id}--{dimension}--{vector_str}"
    return key_str


def create_vector_store(
    kb_id: str,
    dimension: int,
    vector_config: VectorDbConfig,
    table_name: str,
) -> BasePydanticVectorStore:
    cache_key = get_vectordb_cache_key(kb_id, dimension, vector_config, table_name)
    vector_store = vectordb_cache.get(cache_key)
    if vector_store is not None:
        logger.info(f"Using cached vector store for {kb_id} with table name {table_name} with dimension {dimension}.")
        return vector_store

    logger.info(f"Creating new vector store for {kb_id} with table name {table_name} with dimension {dimension}.")

    vector_db_connection = create_vector_db_connection(vector_config.config)

    if isinstance(vector_db_connection, MilvusConnection):
        # milvus collection name should starts with non-numeric character
        if table_name[0].isdigit():
            table_name = "kb"+ table_name

        milvus_url = (
            f"http://{vector_db_connection.host.strip('/')}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        token = f"{vector_db_connection.user}:{decrypt_key(vector_db_connection.encrypted_password)}"
        sparse_embedding_function = BM25BuiltInFunction()

        logger.info(f"Creating Milvus vector store for {kb_id} with table name {table_name} with url: {milvus_url}.")
        vector_store = MilvusVectorStore(
            uri=milvus_url,
            token=token,
            collection_name=table_name,
            dim=dimension,
            enable_sparse=True,
            similarity_metric="cosine",
            hybrid_ranker="WeightedRanker",
            # TODO: add weighted reranker config
            hybrid_ranker_params={"weights": [0.5, 0.5]},
            sparse_embedding_function=sparse_embedding_function,
            db_name=vector_db_connection.database,
        )
    elif isinstance(vector_db_connection, ElasticsearchConnection):
        logger.info(
            f"Creating ElasticsearchStore for {kb_id} with table name {table_name} with url {vector_db_connection.endpoint}."
        )

        vector_store = ElasticsearchStore(
            es_url=vector_db_connection.endpoint,
            index_name=table_name,
            es_user=vector_db_connection.user,
            es_password=decrypt_key(vector_db_connection.encrypted_password),
            dimension=dimension,
            retrieval_strategy=AsyncDenseVectorStrategy(
                hybrid=True,
                rrf=False,
            ),
        )
    elif isinstance(vector_db_connection, PostgresqlConnection):
        logger.info(
            f"Creating PostgresqlStore for {kb_id} with table name {table_name} with url {vector_db_connection.host} {vector_db_connection.database}."
        )
        password = decrypt_key(vector_db_connection.encrypted_password)
        conn_str = (
            f"postgresql+psycopg2://{vector_db_connection.user}:{quote_plus(password)}@{vector_db_connection.host}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        async_conn_str = (
            f"postgresql+asyncpg://{vector_db_connection.user}:{quote_plus(password)}@{vector_db_connection.host}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        vector_store = PGVectorStore(
            connection_string=conn_str,
            async_connection_string=async_conn_str,
            schema_name="public",
            table_name=table_name,
            embed_dim=dimension,
            hybrid_search=True,
            text_search_config="jiebacfg",
        )
    elif isinstance(vector_db_connection, HologresConnection):
        logger.info(
            f"Creating HologresVectorStore for {kb_id} with table name {table_name} with url {vector_db_connection.host}:{vector_db_connection.port}/{vector_db_connection.database}."
        )
        password = quote_plus(decrypt_key(vector_db_connection.encrypted_password))
        vector_store = HologresVectorStore.from_param(
            host=vector_db_connection.host,
            port=vector_db_connection.port,
            database=vector_db_connection.database,
            user=vector_db_connection.user,
            password=password,
            embedding_dimension=dimension,
            table_name=table_name,
        )

    elif isinstance(vector_db_connection, OpensearchConnection):
        logger.info(
            f"Creating OpensearchVectorStore for {kb_id} with table name {table_name} with endpoint {vector_db_connection.endpoint}, instance_id {vector_db_connection.instance_id}, username: {vector_db_connection.username}"
        )

        password = quote_plus(decrypt_key(vector_db_connection.encrypted_password))
        output_fields = [
            "file_name",
            "file_path",
            "file_type",
            "image_url",
            "text",
            "doc_id",
        ]

        config = AlibabaCloudOpenSearchConfig(
            endpoint=vector_db_connection.endpoint,
            instance_id=vector_db_connection.instance_id,
            username=vector_db_connection.username,
            password=password,
            table_name=table_name[:20], # Opensearch 表名最长20
            field_mapping=dict(zip(output_fields, output_fields)),
        )

        vector_store = AlibabaCloudOpenSearchStore(config)
    elif isinstance(vector_db_connection, TablestoreConnection):
        vector_store = TablestoreVectorStore(
            endpoint=vector_db_connection.endpoint,
            instance_name=vector_db_connection.instance_name,
            access_key_id=vector_db_connection.ak,
            access_key_secret=decrypt_key(vector_db_connection.encrypted_sk),
            table_name=table_name,
            index_name="pairag_vector_store_ots_index_v1",
            vector_dimension=dimension,
            # metadata mapping is used to filter non-vector fields.
            metadata_mappings=[
                tablestore.FieldSchema(
                    "file_name",
                    tablestore.FieldType.KEYWORD,
                    index=True,
                    enable_sort_and_agg=True,
                ),
                tablestore.FieldSchema(
                    "file_type",
                    tablestore.FieldType.KEYWORD,
                    index=True,
                    enable_sort_and_agg=True,
                ),
                tablestore.FieldSchema(
                    "file_size",
                    tablestore.FieldType.LONG,
                    index=True,
                    enable_sort_and_agg=True,
                ),
                tablestore.FieldSchema(
                    "file_path",
                    tablestore.FieldType.TEXT,
                    index=True,
                    enable_sort_and_agg=False,
                ),
                tablestore.FieldSchema(
                    "doc_id",
                    tablestore.FieldType.TEXT,
                    index=True,
                    enable_sort_and_agg=False,
                ),
                tablestore.FieldSchema(
                    "creation_date",
                    tablestore.FieldType.DATE,
                    index=True,
                    enable_sort_and_agg=True,
                    date_formats=[
                        "yyyy-MM-dd",
                        "yyyy-MM-dd HH:mm",
                        "yyyy-MM-dd HH:mm:ss",
                        "yyyy-MM-dd HH:mm:ss.SSS",
                    ],
                ),
                tablestore.FieldSchema(
                    "last_modified_date",
                    tablestore.FieldType.DATE,
                    index=True,
                    enable_sort_and_agg=True,
                    date_formats=[
                        "yyyy-MM-dd",
                        "yyyy-MM-dd HH:mm",
                        "yyyy-MM-dd HH:mm:ss",
                        "yyyy-MM-dd HH:mm:ss.SSS",
                    ],
                ),
            ],
        )
        vector_store.create_table_if_not_exist()
        vector_store.create_search_index_if_not_exist()

    elif isinstance(vector_db_connection, LocalConnection):
        logger.info(f"Creating LocalVectorStore for {kb_id} with table name {table_name} with port {DEFAULT_CHROMA_PORT}.")
        vector_store = LocalChromaVectorStore(
            collection_name=table_name,
            host="localhost",
            port=DEFAULT_CHROMA_PORT,
        )
    else:
        raise ValueError(f"Unknown vector_db_connection: {vector_db_connection}.")

    vectordb_cache.put(cache_key, vector_store)
    return vector_store


def create_vector_db_connection(config: dict) -> BaseVectorDbConnection:
    vector_db_type = config.get("type", "local")
    if vector_db_type == VectorDbType.LOCAL:
        return LocalConnection()
    elif vector_db_type == VectorDbType.ELASTICSEARCH:
        return ElasticsearchConnection.from_dict(config)
    elif vector_db_type == VectorDbType.MILVUS:
        return MilvusConnection.from_dict(config)
    elif vector_db_type == VectorDbType.POSTGRESQL:
        return PostgresqlConnection.from_dict(config)
    elif vector_db_type == VectorDbType.HOLOGRES:
        return HologresConnection.from_dict(config)
    elif vector_db_type == VectorDbType.TABLESTORE:
        return TablestoreConnection.from_dict(config)
    elif vector_db_type == VectorDbType.OPENSEARCH:
        return OpensearchConnection.from_dict(config)
    else:
        raise ValueError(f"Invalid vector db type: {vector_db_type}")
