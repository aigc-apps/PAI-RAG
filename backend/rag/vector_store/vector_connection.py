from urllib.parse import quote_plus
from common.encrypt_utils import decrypt_key
from common.knowledgebase.vectordb.base import BaseVectorDbConnection
from common.knowledgebase.vectordb.elastic import ElasticsearchConnection
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
from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

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
        print(f"KEY: {token}")
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
            f"Creating ElasticsearchStore for {kb_id} with url {vector_db_connection.url}."
        )
        return ElasticsearchStore(
            es_url=vector_db_connection.endpoint,
            index_name=kb_id,
            es_user=vector_db_connection.user,
            es_password=decrypt_key(vector_db_connection.encrypted_password),
            dim=dimension,
            retrieval_strategy=AsyncDenseVectorStrategy(
                hybrid=True, rrf={"window_size": 50}
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
