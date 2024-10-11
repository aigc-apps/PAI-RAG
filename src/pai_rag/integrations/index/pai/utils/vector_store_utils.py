import hashlib
import faiss
import logging
import os
import json
from llama_index.core.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.core.vector_stores.types import DEFAULT_PERSIST_FNAME
from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
from pai_rag.integrations.vector_stores.hologres.hologres import HologresVectorStore
from pai_rag.integrations.vector_stores.elasticsearch.my_elasticsearch import (
    MyElasticsearchStore,
)
from pai_rag.integrations.vector_stores.faiss.my_faiss import MyFaissVectorStore
from pai_rag.integrations.vector_stores.milvus.my_milvus import MyMilvusVectorStore
from pai_rag.integrations.vector_stores.analyticdb.my_analyticdb import (
    MyAnalyticDBVectorStore,
)
from pai_rag.integrations.vector_stores.postgresql.postgresql import PGVectorStore
from pai_rag.integrations.index.pai.vector_store_config import (
    BaseVectorStoreConfig,
    AnalyticDBVectorStoreConfig,
    PostgreSQLVectorStoreConfig,
    FaissVectorStoreConfig,
    MilvusVectorStoreConfig,
    ElasticSearchVectorStoreConfig,
    OpenSearchVectorStoreConfig,
    HologresVectorStoreConfig,
)


logger = logging.getLogger(__name__)

DEFAULT_PERSIST_IMAGE_NAMESPACE = "image"


def create_vector_store(
    vectordb_config: BaseVectorStoreConfig,
    embed_dims: int,
    is_image_store: bool = False,
):
    if isinstance(vectordb_config, FaissVectorStoreConfig):
        create_vector_store_func = create_faiss
    elif isinstance(vectordb_config, MilvusVectorStoreConfig):
        create_vector_store_func = create_milvus
    elif isinstance(vectordb_config, HologresVectorStoreConfig):
        create_vector_store_func = create_hologres
    elif isinstance(vectordb_config, AnalyticDBVectorStoreConfig):
        create_vector_store_func = create_adb
    elif isinstance(vectordb_config, ElasticSearchVectorStoreConfig):
        create_vector_store_func = create_elasticsearch
    elif isinstance(vectordb_config, PostgreSQLVectorStoreConfig):
        create_vector_store_func = create_postgresql
    elif isinstance(vectordb_config, OpenSearchVectorStoreConfig):
        create_vector_store_func = create_opensearch
    else:
        raise ValueError(f"Unknown vector store config {vectordb_config}.")

    return create_vector_store_func(
        vectordb_config,
        embed_dims=embed_dims,
        is_image_store=is_image_store,
    )


def create_faiss(
    faiss_config: FaissVectorStoreConfig,
    embed_dims: int,
    is_image_store: bool = False,
):
    if is_image_store:
        faiss_vector_index_path = os.path.join(
            faiss_config.persist_path,
            f"{DEFAULT_PERSIST_IMAGE_NAMESPACE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}",
        )
    else:
        faiss_vector_index_path = os.path.join(
            faiss_config.persist_path,
            f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}",
        )

    if os.path.exists(faiss_vector_index_path):
        faiss_store = MyFaissVectorStore.from_persist_path(faiss_vector_index_path)
    else:
        faiss_index = faiss.IndexFlatIP(embed_dims)
        faiss_store = MyFaissVectorStore(faiss_index=faiss_index)
    return faiss_store


def create_hologres(
    hologres_config: HologresVectorStoreConfig,
    embed_dims: int,
    is_image_store: bool = False,
):
    table_name = hologres_config.table_name
    if is_image_store:
        table_name = f"{table_name}__image"
    hologres = HologresVectorStore.from_param(
        host=hologres_config.host,
        port=hologres_config.port,
        user=hologres_config.user,
        password=hologres_config.password,
        database=hologres_config.database,
        table_name=table_name,
        embedding_dimension=embed_dims,
        pre_delete_table=hologres_config.pre_delete_table,
    )

    return hologres


def create_adb(
    adb_config: AnalyticDBVectorStoreConfig,
    embed_dims: int,
    is_image_store: bool = False,
):
    collection = adb_config.collection
    if is_image_store:
        collection = f"{collection}__image"

    adb = MyAnalyticDBVectorStore.from_params(
        access_key_id=adb_config.ak,
        access_key_secret=adb_config.sk,
        region_id=adb_config.region_id,
        instance_id=adb_config.instance_id,
        account=adb_config.account,
        account_password=adb_config.account_password,
        namespace=adb_config.namespace,
        collection=collection,
        metrics="cosine",
        embedding_dimension=embed_dims,
    )
    return adb


def create_elasticsearch(
    es_config: ElasticSearchVectorStoreConfig,
    embed_dims: int,
    is_image_store: bool = False,
):
    index_name = es_config.es_index
    if is_image_store:
        index_name = f"{index_name}__image"

    es_store = MyElasticsearchStore(
        index_name=index_name,
        es_url=es_config.es_url,
        es_user=es_config.es_user,
        es_password=es_config.es_password,
        embedding_dimension=embed_dims,
        retrieval_strategy=AsyncDenseVectorStrategy(
            hybrid=True, rrf={"window_size": 50}
        ),
    )

    return es_store


def create_milvus(
    milvus_config: MilvusVectorStoreConfig,
    embed_dims: int,
    is_image_store: bool = False,
):
    collection_name = milvus_config.collection_name
    if is_image_store:
        collection_name = f"{collection_name}__image"

    milvus_url = f"http://{milvus_config.host.strip('/')}:{milvus_config.port}/{milvus_config.database}"
    token = f"{milvus_config.user}:{milvus_config.password}"
    milvus_store = MyMilvusVectorStore(
        uri=milvus_url,
        token=token,
        collection_name=collection_name,
        dim=embed_dims,
        enable_sparse=True if not is_image_store else False,
        sparse_embedding_function=BGEM3SparseEmbeddingFunction()
        if not is_image_store
        else None,
        similarity_metric="cosine",
        hybrid_ranker="WeightedRanker",
        # TODO: add weighted reranker config
        hybrid_ranker_params={"weights": milvus_config.reranker_weights},
    )

    return milvus_store


def create_opensearch(
    opensearch_config: OpenSearchVectorStoreConfig,
    embed_dims: int,
    is_image_store: bool = False,
):
    from llama_index.vector_stores.alibabacloud_opensearch import (
        AlibabaCloudOpenSearchStore,
        AlibabaCloudOpenSearchConfig,
    )

    table_name = opensearch_config.table_name
    if is_image_store:
        table_name = f"{table_name}_image"  # opensearch does not support __ in naming

    if is_image_store:
        output_fields = [
            "file_name",
            "file_path",
            "file_type",
            "image_url",
            "text",
            "doc_id",
        ]
    else:
        output_fields = [
            "file_name",
            "file_path",
            "file_type",
            "image_url",
            "text",
            "doc_id",
        ]

    db_config = AlibabaCloudOpenSearchConfig(
        endpoint=opensearch_config.endpoint,
        instance_id=opensearch_config.instance_id,
        username=opensearch_config.username,
        password=opensearch_config.password,
        table_name=table_name,
        # OpenSearch constructor has bug in dealing with output fields
        field_mapping=dict(zip(output_fields, output_fields)),
    )

    opensearch_store = AlibabaCloudOpenSearchStore(config=db_config)
    return opensearch_store


def create_postgresql(
    pg_config: PostgreSQLVectorStoreConfig,
    embed_dims: int,
    is_image_store: bool = False,
):
    table_name = pg_config.table_name
    if is_image_store:
        table_name = f"{table_name}__image"

    pg = PGVectorStore.from_params(
        host=pg_config.host,
        port=pg_config.port,
        database=pg_config.database,
        table_name=table_name,
        user=pg_config.username,
        password=pg_config.password,
        embed_dim=embed_dims,
        hybrid_search=True,
        text_search_config="jiebacfg",
    )
    return pg


# change persist path to sub folders in the persist_path to separate different vector index
def resolve_store_path(store_config: BaseVectorStoreConfig, ndims: int = 1536):
    if isinstance(store_config, FaissVectorStoreConfig):
        raw_text = {"type": "faiss"}
    elif isinstance(store_config, HologresVectorStoreConfig):
        json_data = {
            "host": store_config.host,
            "port": store_config.port,
            "database": store_config.database,
            "table_name": store_config.table_name,
        }
        raw_text = json.dumps(json_data)
    elif isinstance(store_config, OpenSearchVectorStoreConfig):
        json_data = {
            "endpoint": store_config.endpoint,
            "instance_id": store_config.instance_id,
            "table_name": store_config.table_name,
        }
        raw_text = json.dumps(json_data)
    else:
        raw_text = repr(store_config)

    encoded_raw_text = f"{raw_text}_{ndims}".encode()
    hash = hashlib.sha256(encoded_raw_text).hexdigest()
    return os.path.join(store_config.persist_path, hash)
