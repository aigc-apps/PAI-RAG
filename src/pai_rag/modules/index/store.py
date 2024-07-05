import faiss
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.vector_stores.analyticdb import AnalyticDBVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy

from pai_rag.integrations.vector_stores.vector_stores_hologres.hologres import (
    HologresVectorStore,
)
from pai_rag.modules.index.my_milvus_vector_store import MyMilvusVectorStore
from pai_rag.modules.index.sparse_embedding import BGEM3SparseEmbeddingFunction
from llama_index.core import StorageContext
import logging

from pai_rag.modules.retriever.my_elasticsearch_store import MyElasticsearchStore

logger = logging.getLogger(__name__)


class RagStore:
    def __init__(self, config, persist_dir, is_empty, embed_dims):
        self.store_config = config
        self.embed_dims = embed_dims
        self.persist_dir = persist_dir
        self.is_empty = is_empty

    def get_storage_context(self):
        storage_context = self._get_or_create_storage_context()
        return storage_context

    def _get_or_create_storage_context(self):
        self.vector_store = None
        self.doc_store = None
        self.index_store = None
        persist_dir = None

        vector_store_type = (
            self.store_config["vector_store"].get("type", "faiss").lower()
        )

        if vector_store_type == "faiss":
            self.doc_store = self._get_or_create_simple_doc_store()
            self.index_store = self._get_or_create_simple_index_store()
            persist_dir = self.persist_dir
            self.vector_store = self._get_or_create_faiss()
            logger.info("initialized FAISS vector store.")
        elif vector_store_type == "hologres":
            self.vector_store = self._get_or_create_hologres()
            logger.info("initialized Hologres vector store.")
        elif vector_store_type == "analyticdb":
            self.vector_store = self._get_or_create_adb()
            logger.info("initialized AnalyticDB vector store.")
        elif vector_store_type == "elasticsearch":
            self.vector_store = self._get_or_create_es()
            logger.info("initialized ElasticSearch vector store.")
        elif vector_store_type == "milvus":
            self.vector_store = self._get_or_create_milvus()
        else:
            raise ValueError(f"Unknown vector_store type '{vector_store_type}'.")

        storage_context = StorageContext.from_defaults(
            docstore=self.doc_store,
            index_store=self.index_store,
            vector_store=self.vector_store,
            persist_dir=persist_dir,
        )
        return storage_context

    def _get_or_create_faiss(self):
        if self.is_empty:
            faiss_index = faiss.IndexFlatL2(self.embed_dims)
            faiss_store = FaissVectorStore(faiss_index=faiss_index)
        else:
            faiss_store = FaissVectorStore.from_persist_dir(self.persist_dir)

        return faiss_store

    def _get_or_create_hologres(self):
        hologres_config = self.store_config["vector_store"]
        hologres = HologresVectorStore.from_param(
            host=hologres_config["host"],
            port=hologres_config["port"],
            user=hologres_config["user"],
            password=hologres_config["password"],
            database=hologres_config["database"],
            table_name=hologres_config["table_name"],
            embedding_dimension=self.embed_dims,
            pre_delete_table=hologres_config["pre_delete_table"],
        )
        return hologres

    def _get_or_create_adb(self):
        adb_config = self.store_config["vector_store"]
        adb = AnalyticDBVectorStore.from_params(
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

        return MyElasticsearchStore(
            index_name=es_config["es_index"],
            es_url=es_config["es_url"],
            es_user=es_config["es_user"],
            es_password=es_config["es_password"],
            embedding_dimension=self.embed_dims,
            retrieval_strategy=AsyncDenseVectorStrategy(
                hybrid=True, rrf={"window_size": 50}
            ),
        )

    def _get_or_create_milvus(self):
        milvus_config = self.store_config["vector_store"]
        milvus_host = milvus_config["host"]
        milvus_port = milvus_config["port"]
        milvus_user = milvus_config["user"]
        milvus_password = milvus_config["password"]
        milvus_database = milvus_config["database"]

        milvus_url = f"http://{milvus_host.strip('/')}:{milvus_port}/{milvus_database}"
        token = f"{milvus_user}:{milvus_password}"
        return MyMilvusVectorStore(
            uri=milvus_url,
            token=token,
            collection_name=milvus_config["collection_name"],
            dim=self.embed_dims,
            enable_sparse=True,
            sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
        )

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
