import os
import chromadb
import faiss
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore

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

logger = logging.getLogger(__name__)


class RagStore:
    def __init__(self, config, postprocessor, persist_dir, is_empty, embed_dims):
        self.store_config = config
        print("self.store_config", self.store_config)
        self.postprocessor = postprocessor
        print("self.postprocessor", self.postprocessor)
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

        if vector_store_type == "chroma":
            self.vector_store = self._get_or_create_chroma()
            logger.info("initialized Chroma vector store.")
        elif vector_store_type == "faiss":
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
        elif vector_store_type == "opensearch":
            self.vector_store = self._get_or_create_open_search_store()
        elif vector_store_type == "postgresql":
            self.vector_store = self._get_or_create_postgresql_store()
        else:
            raise ValueError(f"Unknown vector_store type '{vector_store_type}'.")

        storage_context = StorageContext.from_defaults(
            docstore=self.doc_store,
            index_store=self.index_store,
            vector_store=self.vector_store,
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
        if self.is_empty:
            # faiss_index = faiss.IndexFlatL2(self.embed_dims)
            faiss_index = faiss.IndexFlatIP(self.embed_dims)
            faiss_store = MyFaissVectorStore(faiss_index=faiss_index)
        else:
            faiss_store = MyFaissVectorStore.from_persist_dir(self.persist_dir)

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
        weighted_reranker = False
        weights = []
        for item in self.postprocessor:
            if isinstance(item, MySimpleWeightedRerank):
                weighted_reranker = True
                weights.append(item.vector_weight)
                weights.append(item.keyword_weight)
                print("weighted_reranker", weighted_reranker, weights)
                break
        return MyMilvusVectorStore(
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

    def _get_or_create_open_search_store(self):
        from llama_index.vector_stores.alibabacloud_opensearch import (
            AlibabaCloudOpenSearchStore,
            AlibabaCloudOpenSearchConfig,
        )

        open_search_config = self.store_config["vector_store"]
        output_fields = ["file_name", "file_path", "file_type", "text", "doc_id"]
        db_config = AlibabaCloudOpenSearchConfig(
            endpoint=open_search_config["endpoint"],
            instance_id=open_search_config["instance_id"],
            username=open_search_config["username"],
            password=open_search_config["password"],
            table_name=open_search_config["table_name"],
            # OpenSearch constructor has bug in dealing with output fields
            field_mapping=dict(zip(output_fields, output_fields)),
        )

        return AlibabaCloudOpenSearchStore(config=db_config)

    def _get_or_create_postgresql_store(self):
        pg_config = self.store_config["vector_store"]
        pg = PGVectorStore.from_params(
            host=pg_config["host"],
            port=pg_config["port"],
            database=pg_config["database"],
            table_name=pg_config["table_name"],
            user=pg_config["username"],
            password=pg_config["password"],
            embed_dim=self.embed_dims,
            hybrid_search=True,
            text_search_config="jiebacfg",
        )
        return pg

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
