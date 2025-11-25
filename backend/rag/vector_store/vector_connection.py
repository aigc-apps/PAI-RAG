from urllib.parse import quote_plus
from common.encrypt_utils import decrypt_key
from common.knowledgebase.vectordb.base import BaseVectorDbConnection
from common.knowledgebase.vectordb.elastic import ElasticsearchConnection
from common.knowledgebase.vectordb.hologres import HologresConnection
from common.knowledgebase.vectordb.opensearch import OpensearchConnection
from common.knowledgebase.vectordb.tablestore import TablestoreConnection
from common.knowledgebase.vectordb.local import LocalConnection
from common.knowledgebase.vectordb.milvus import MilvusConnection
from common.knowledgebase.vectordb.postgres import PostgresqlConnection
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from loguru import logger

from rag.vector_store.local_chroma_service import DEFAULT_CHROMA_PORT
from rag.vector_store.local import LocalChromaVectorStore
from rag.vector_store.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.vector_stores.hologres import HologresVectorStore
from llama_index.vector_stores.alibabacloud_opensearch import AlibabaCloudOpenSearchConfig, AlibabaCloudOpenSearchStore
import tablestore
from llama_index.vector_stores.tablestore import TablestoreVectorStore
from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy

def create_vector_store(
    kb_id: str,
    dimension: int,
    vector_db_connection: BaseVectorDbConnection,
) -> BasePydanticVectorStore:
    if isinstance(vector_db_connection, MilvusConnection):
        milvus_url = (
            f"http://{vector_db_connection.host.strip('/')}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        token = f"{vector_db_connection.user}:{decrypt_key(vector_db_connection.encrypted_password)}"
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
    elif isinstance(vector_db_connection, ElasticsearchConnection):
        logger.info(
            f"Creating ElasticsearchStore for {kb_id} with url {vector_db_connection.endpoint}."
        )
        return ElasticsearchStore(
            es_url=vector_db_connection.endpoint,
            index_name=kb_id,
            es_user=vector_db_connection.user,
            es_password=decrypt_key(vector_db_connection.encrypted_password),
            dim=dimension,
            retrieval_strategy=AsyncDenseVectorStrategy(
                hybrid=True, rrf=False
            ),
        )
    elif isinstance(vector_db_connection, PostgresqlConnection):
        logger.info(
            f"Creating PostgresqlStore for {kb_id} with url {vector_db_connection.host} {vector_db_connection.database}."
        )
        password = decrypt_key(vector_db_connection.encrypted_password)
        conn_str = (
            f"postgresql+psycopg2://{vector_db_connection.user}:{quote_plus(password)}@{vector_db_connection.host}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        async_conn_str = (
            f"postgresql+asyncpg://{vector_db_connection.user}:{quote_plus(password)}@{vector_db_connection.host}:{vector_db_connection.port}/{vector_db_connection.database}"
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
    elif isinstance(vector_db_connection, HologresConnection):
        logger.info(
            f"Creating HologresVectorStore for {kb_id} with url {vector_db_connection.host}:{vector_db_connection.port}/{vector_db_connection.database}."
        )
        password = quote_plus(decrypt_key(vector_db_connection.encrypted_password))
        vector_store = HologresVectorStore.from_param(
            host=vector_db_connection.host,
            port=vector_db_connection.port,
            database=vector_db_connection.database,
            user=vector_db_connection.user,
            password=password,
            embedding_dimension=dimension,
            table_name=kb_id,
        )
        return vector_store
    elif isinstance(vector_db_connection, OpensearchConnection):
        logger.info(
            f"Creating OpensearchVectorStore for {kb_id} with endpoint {vector_db_connection.endpoint}, instance_id {vector_db_connection.instance_id}, username: {vector_db_connection.username}"
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
            table_name=kb_id[:20], # Opensearch 表名最长20
            field_mapping=dict(zip(output_fields, output_fields)),
        )

        vector_store = AlibabaCloudOpenSearchStore(config)
        return vector_store
    elif isinstance(vector_db_connection, TablestoreConnection):
        tablestore_store = TablestoreVectorStore(
            endpoint=vector_db_connection.endpoint,
            instance_name=vector_db_connection.instance_name,
            access_key_id=vector_db_connection.ak,
            access_key_secret=decrypt_key(vector_db_connection.encrypted_sk),
            table_name=kb_id,
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
        tablestore_store.create_table_if_not_exist()
        tablestore_store.create_search_index_if_not_exist()
        return tablestore_store

    elif isinstance(vector_db_connection, LocalConnection):
        logger.info(f"Creating LocalVectorStore for {kb_id} with port {DEFAULT_CHROMA_PORT}.")
        return LocalChromaVectorStore(
            collection_name=kb_id,
            host="localhost",
            port=DEFAULT_CHROMA_PORT,
        )
    else:
        raise ValueError(f"Unknown vector_db_connection: {vector_db_connection}.")


async def cleanup_vector_store(vector_store: BasePydanticVectorStore):
    """
    Clean up vector store connections.
    For PGVectorStore, safely close the connection to avoid greenlet errors and connection leaks.
    For AlibabaCloudOpenSearchStore, close aiohttp connections to avoid connection leaks.
    """
    if isinstance(vector_store, PGVectorStore):
        try:
            # 优先关闭引擎连接池，这会关闭所有连接池中的连接
            # 这样可以避免连接泄漏，即使 close() 方法失败
            if hasattr(vector_store, '_engine') and vector_store._engine:
                try:
                    # dispose(close=True) 会关闭所有连接并清理连接池
                    await vector_store._engine.dispose(close=True)
                except Exception as e:
                    logger.debug(f"Error disposing engine: {e}")

            # 然后尝试关闭所有会话（如果 close_all 可用）
            # 注意：这可能会因为 greenlet 错误而失败，但我们已经关闭了引擎
            try:
                await vector_store.close()
            except Exception as e:
                # 捕获关闭时的错误，避免 greenlet 相关错误影响主流程
                # 由于我们已经关闭了引擎，这个错误通常可以安全忽略
                logger.debug(f"Error calling close() on PGVectorStore (usually safe to ignore): {e}")

        except Exception as e:
            # 捕获所有其他错误，避免影响主流程
            logger.warning(f"Error cleaning up PGVectorStore connections: {e}")
    elif isinstance(vector_store, AlibabaCloudOpenSearchStore):
        try:
            # 关闭 AlibabaCloudOpenSearchStore 的 aiohttp 连接
            if hasattr(vector_store, '_client') and vector_store._client:
                try:
                    if hasattr(vector_store._client, 'close'):
                        await vector_store._client.close()
                    if hasattr(vector_store._client, '_connector') and vector_store._client._connector:
                        await vector_store._client._connector.close()
                except Exception as e:
                    logger.debug(f"Error closing AlibabaCloudOpenSearchStore client: {e}")
            # 尝试调用 close 方法（如果存在）
            if hasattr(vector_store, 'close'):
                try:
                    await vector_store.close()
                except Exception as e:
                    logger.debug(f"Error calling close() on AlibabaCloudOpenSearchStore: {e}")
        except Exception as e:
            logger.warning(f"Error cleaning up AlibabaCloudOpenSearchStore connections: {e}")


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
