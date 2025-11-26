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
    table_name = kb_id

    if isinstance(vector_db_connection, MilvusConnection):
        # milvus collection name should starts with non-numeric character
        if len(kb_id) <= 32:
            table_name = "kb"+ kb_id

        milvus_url = (
            f"http://{vector_db_connection.host.strip('/')}:{vector_db_connection.port}/{vector_db_connection.database}"
        )
        token = f"{vector_db_connection.user}:{decrypt_key(vector_db_connection.encrypted_password)}"
        sparse_embedding_function = BM25BuiltInFunction()

        logger.info(f"Creating Milvus vector store for {kb_id} with url: {milvus_url}.")
        return MilvusVectorStore(
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
            f"Creating ElasticsearchStore for {kb_id} with url {vector_db_connection.endpoint}."
        )
        return ElasticsearchStore(
            es_url=vector_db_connection.endpoint,
            index_name=table_name,
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
            table_name=table_name,
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
            table_name=table_name,
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
            table_name=table_name[:20], # Opensearch 表名最长20
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
    if isinstance(vector_store, PGVectorStore):
        await vector_store.close()


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
