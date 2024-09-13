import os
import chromadb
import faiss
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.core.vector_stores.types import DEFAULT_PERSIST_FNAME

# from llama_index.vector_stores.analyticdb import AnalyticDBVectorStore
# from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy

from pai_rag.modules.index.sparse_embedding import BGEM3SparseEmbeddingFunction
from llama_index.core import StorageContext
import logging

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
from pai_rag.integrations.postprocessor.my_simple_weighted_rerank import (
    MySimpleWeightedRerank,
)

DEFAULT_CHROMA_COLLECTION_NAME = "pairag"
DEFAULT_PERSIST_IMAGE_NAMESPACE = "image"
logger = logging.getLogger(__name__)


class RagStore:
    def __init__(
        self,
        config,
        postprocessor,
        persist_dir,
        is_empty,
        embed_dims,
        multi_modal_embed_dims,
    ):
        self.store_config = config
        self.postprocessor = postprocessor
        self.embed_dims = embed_dims
        self.persist_dir = persist_dir
        self.is_empty = is_empty
        self.multi_modal_embed_dims = multi_modal_embed_dims

    def get_storage_context(self):
        storage_context = self._get_or_create_storage_context()
        return storage_context

    def _get_or_create_storage_context(self):
        self.vector_store = None
        self.doc_store = None
        self.index_store = None
        self.image_store = None
        persist_dir = None

        vector_store_type = (
            self.store_config["vector_store"].get("type", "faiss").lower()
        )

        if vector_store_type == "faiss":
            self.doc_store = self._get_or_create_simple_doc_store()
            self.index_store = self._get_or_create_simple_index_store()
            self.vector_store, self.image_store = self._get_or_create_faiss()
            logger.info("initialized FAISS vector & image store.")
        elif vector_store_type == "hologres":
            self.vector_store, self.image_store = self._get_or_create_hologres()
            logger.info("initialized Hologres vector & image store.")
        # TODO: not supported yet, need more tests
        elif vector_store_type == "elasticsearch":
            self.vector_store, self.image_store = self._get_or_create_es()
            logger.info("initialized ElasticSearch vector & image store.")
        elif vector_store_type == "milvus":
            self.vector_store, self.image_store = self._get_or_create_milvus()
            logger.info("initialized Milvus vector & image store.")
        elif vector_store_type == "opensearch":
            (
                self.vector_store,
                self.image_store,
            ) = self._get_or_create_open_search_store()
            logger.info("initialized OpenSearch vector & image store.")
        elif vector_store_type == "postgresql":
            self.vector_store, self.image_store = self._get_or_create_postgresql_store()
            logger.info("initialized Postgresql vector & image store.")
        # Not used yet
        elif vector_store_type == "chroma":
            self.vector_store = self._get_or_create_chroma()
            logger.info("initialized Chroma vector store.")
        elif vector_store_type == "analyticdb":
            self.vector_store = self._get_or_create_adb()
            logger.info("initialized AnalyticDB vector store.")
        else:
            raise ValueError(f"Unknown vector_store type '{vector_store_type}'.")

        storage_context = StorageContext.from_defaults(
            docstore=self.doc_store,
            index_store=self.index_store,
            vector_store=self.vector_store,
            image_store=self.image_store,
            persist_dir=persist_dir,
        )
        return storage_context

    def _get_or_create_chroma(self):
        chroma_path = os.path.join(self.persist_dir, "chroma")
        chroma_db = chromadb.PersistentClient(path=chroma_path)

        collection_name = self.store_config["vector_store"].get(
            "collection_name", DEFAULT_CHROMA_COLLECTION_NAME
        )
        chroma_collection = chroma_db.get_or_create_collection(collection_name)

        return ChromaVectorStore(chroma_collection=chroma_collection)

    def _get_or_create_faiss(self):
        faiss_vector_index_path = os.path.join(
            self.persist_dir,
            f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}",
        )
        faiss_image_vector_index_path = os.path.join(
            self.persist_dir,
            f"{DEFAULT_PERSIST_IMAGE_NAMESPACE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}",
        )

        if os.path.exists(faiss_vector_index_path):
            faiss_store = MyFaissVectorStore.from_persist_path(faiss_vector_index_path)
        else:
            faiss_index = faiss.IndexFlatIP(self.embed_dims)
            faiss_store = MyFaissVectorStore(faiss_index=faiss_index)

        if os.path.exists(faiss_image_vector_index_path):
            faiss_image_store = MyFaissVectorStore.from_persist_path(
                faiss_image_vector_index_path
            )
        else:
            image_index = faiss.IndexFlatIP(self.multi_modal_embed_dims)
            faiss_image_store = MyFaissVectorStore(faiss_index=image_index)

        return faiss_store, faiss_image_store

    def _get_or_create_hologres(self):
        hologres_config = self.store_config["vector_store"]
        hologres_text_store = HologresVectorStore.from_param(
            host=hologres_config["host"],
            port=hologres_config["port"],
            user=hologres_config["user"],
            password=hologres_config["password"],
            database=hologres_config["database"],
            table_name=hologres_config["table_name"],
            embedding_dimension=self.embed_dims,
            pre_delete_table=hologres_config["pre_delete_table"],
        )
        hologres_image_store = HologresVectorStore.from_param(
            host=hologres_config["host"],
            port=hologres_config["port"],
            user=hologres_config["user"],
            password=hologres_config["password"],
            database=hologres_config["database"],
            table_name=hologres_config["table_name"] + "__image",
            embedding_dimension=self.multi_modal_embed_dims,
            pre_delete_table=hologres_config["pre_delete_table"],
        )
        return hologres_text_store, hologres_image_store

    def _get_or_create_adb(self):
        adb_config = self.store_config["vector_store"]
        adb = MyAnalyticDBVectorStore.from_params(
            access_key_id=adb_config["ak"],
            access_key_secret=adb_config["sk"],
            region_id=adb_config["region_id"],
            instance_id=adb_config["instance_id"],
            account=adb_config["account"],
            account_password=adb_config["account_password"],
            namespace=adb_config["namespace"],
            collection=adb_config["collection"],
            metrics=adb_config.get("metrics", "cosine"),
            embedding_dimension=self.embed_dims,
        )
        return adb

    def _get_or_create_es(self):
        es_config = self.store_config["vector_store"]

        es_text_store = MyElasticsearchStore(
            index_name=es_config["es_index"],
            es_url=es_config["es_url"],
            es_user=es_config["es_user"],
            es_password=es_config["es_password"],
            embedding_dimension=self.embed_dims,
            retrieval_strategy=AsyncDenseVectorStrategy(
                hybrid=True, rrf={"window_size": 50}
            ),
        )
        es_image_store = MyElasticsearchStore(
            index_name=es_config["es_index"] + "__image",
            es_url=es_config["es_url"],
            es_user=es_config["es_user"],
            es_password=es_config["es_password"],
            embedding_dimension=self.multi_modal_embed_dims,
            retrieval_strategy=AsyncDenseVectorStrategy(hybrid=False),
        )
        return es_text_store, es_image_store

    def _get_or_create_milvus(self):
        milvus_config = self.store_config["vector_store"]
        milvus_host = milvus_config["host"]
        milvus_port = milvus_config["port"]
        milvus_user = milvus_config["user"]
        milvus_password = milvus_config["password"]
        milvus_database = milvus_config["database"]

        milvus_url = f"http://{milvus_host.strip('/')}:{milvus_port}/{milvus_database}"
        token = f"{milvus_user}:{milvus_password}"
        weighted_reranker = False
        weights = []
        for item in self.postprocessor:
            if isinstance(item, MySimpleWeightedRerank):
                weighted_reranker = True
                weights.append(item.vector_weight)
                weights.append(item.keyword_weight)
                print("weighted_reranker", weighted_reranker, weights)
                break
        milvus_text_store = MyMilvusVectorStore(
            uri=milvus_url,
            token=token,
            collection_name=milvus_config["collection_name"],
            dim=self.embed_dims,
            enable_sparse=True,
            sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
            similarity_metric="cosine",
            hybrid_ranker="WeightedRanker",
            hybrid_ranker_params={"weights": weights} if weighted_reranker else {},
        )
        milvus_image_store = MyMilvusVectorStore(
            uri=milvus_url,
            token=token,
            collection_name=milvus_config["collection_name"] + "__image",
            dim=self.multi_modal_embed_dims,
            enable_sparse=False,
            similarity_metric="cosine",
        )
        return milvus_text_store, milvus_image_store

    def _get_or_create_open_search_store(self):
        from llama_index.vector_stores.alibabacloud_opensearch import (
            AlibabaCloudOpenSearchStore,
            AlibabaCloudOpenSearchConfig,
        )

        open_search_config = self.store_config["vector_store"]
        text_output_fields = [
            "file_name",
            "file_path",
            "file_type",
            "image_url_list_str",
            "text",
            "doc_id",
        ]
        text_db_config = AlibabaCloudOpenSearchConfig(
            endpoint=open_search_config["endpoint"],
            instance_id=open_search_config["instance_id"],
            username=open_search_config["username"],
            password=open_search_config["password"],
            table_name=open_search_config["table_name"],
            # OpenSearch constructor has bug in dealing with output fields
            field_mapping=dict(zip(text_output_fields, text_output_fields)),
        )
        image_output_fields = [
            "file_name",
            "file_path",
            "file_type",
            "image_url",
            "text",
            "doc_id",
        ]
        image_db_config = AlibabaCloudOpenSearchConfig(
            endpoint=open_search_config["endpoint"],
            instance_id=open_search_config["instance_id"],
            username=open_search_config["username"],
            password=open_search_config["password"],
            table_name=open_search_config["table_name"] + "__image",
            # OpenSearch constructor has bug in dealing with output fields
            field_mapping=dict(zip(image_output_fields, image_output_fields)),
        )

        return AlibabaCloudOpenSearchStore(
            config=text_db_config
        ), AlibabaCloudOpenSearchStore(config=image_db_config)

    def _get_or_create_postgresql_store(self):
        pg_config = self.store_config["vector_store"]
        text_pg = PGVectorStore.from_params(
            host=pg_config["host"],
            port=pg_config["port"],
            database=pg_config["database"],
            table_name=pg_config["table_name"]
            if pg_config["table_name"].strip()
            else "default",
            user=pg_config["username"],
            password=pg_config["password"],
            embed_dim=self.embed_dims,
            hybrid_search=True,
            text_search_config="jiebacfg",
        )
        image_pg = PGVectorStore.from_params(
            host=pg_config["host"],
            port=pg_config["port"],
            database=pg_config["database"],
            table_name=pg_config["table_name"] + "__image"
            if pg_config["table_name"].strip()
            else "default__image",
            user=pg_config["username"],
            password=pg_config["password"],
            embed_dim=self.multi_modal_embed_dims,
            hybrid_search=False,
            text_search_config="jiebacfg",
        )
        return text_pg, image_pg

    def _get_or_create_simple_doc_store(self):
        if self.is_empty:
            doc_store = SimpleDocumentStore()
        else:
            doc_store = SimpleDocumentStore().from_persist_dir(self.persist_dir)
        return doc_store

    def _get_or_create_simple_index_store(self):
        if self.is_empty:
            index_store = SimpleIndexStore()
        else:
            index_store = SimpleIndexStore().from_persist_dir(self.persist_dir)
        return index_store
