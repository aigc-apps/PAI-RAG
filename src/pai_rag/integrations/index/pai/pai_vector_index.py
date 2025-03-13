import os
import asyncio
from typing import Coroutine, List, Any, Sequence
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from pai_rag.integrations.index.pai.vector_store_config import BaseVectorStoreConfig
from pai_rag.integrations.index.pai.multimodal.multimodal_index import (
    PaiMultiModalVectorStoreIndex,
)
from pai_rag.integrations.index.pai.utils.index_utils import load_index_from_storage
import llama_index.core.storage.docstore.types as DocStoreTypes
import llama_index.core.storage.index_store.types as IndexStoreTypes
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core import StorageContext
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from pai_rag.integrations.index.pai.utils.vector_store_utils import (
    create_vector_store,
    resolve_store_path,
)
from pai_rag.integrations.index.pai.vector_store_config import (
    VECTOR_STORE_TYPES_WITH_HYBRID_SEARCH,
    VectorIndexRetrievalType,
)
from pai_rag.integrations.vector_stores.milvus.my_milvus import MyMilvusVectorStore
from pai_rag.integrations.vector_stores.elasticsearch.my_elasticsearch import (
    MyElasticsearchStore,
)
from pai_rag.integrations.vector_stores.faiss.my_faiss import MyFaissVectorStore
from pai_rag.integrations.index.pai.local.local_bm25_index import LocalBm25IndexStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from llama_index.core.constants import (
    DEFAULT_SIMILARITY_TOP_K,
    DEFAULT_IMAGE_SIMILARITY_TOP_K,
)

from loguru import logger


def retrieval_type_to_search_mode(retrieval_type: VectorIndexRetrievalType):
    if retrieval_type == VectorIndexRetrievalType.keyword:
        return VectorStoreQueryMode.TEXT_SEARCH
    elif retrieval_type == VectorIndexRetrievalType.hybrid:
        return VectorStoreQueryMode.HYBRID
    else:
        return VectorStoreQueryMode.DEFAULT


class PaiVectorStoreIndex(VectorStoreIndex):
    _vector_store: BasePydanticVectorStore = PrivateAttr()
    _embed_model: BaseEmbedding = PrivateAttr()
    _storage_context: StorageContext = PrivateAttr()
    _vector_index: VectorStoreIndex = PrivateAttr()

    _persist_path: str
    # Enable local keyword index for

    def __init__(
        self,
        vector_store_config: BaseVectorStoreConfig,
        embed_model: BaseEmbedding,
        enable_local_keyword_index: bool = False,
        vector_index_retrieval_type: VectorIndexRetrievalType = VectorIndexRetrievalType.embedding,
        similarity_top_k=DEFAULT_SIMILARITY_TOP_K,
        image_similarity_top_k=DEFAULT_IMAGE_SIMILARITY_TOP_K,
        retriever_weights: List[float] = None,
    ):
        self.vector_store_config = vector_store_config

        embed_dims = len(embed_model.get_text_embedding("0"))
        self._embed_model = embed_model

        # change persist path to subfolder
        self._persist_path = resolve_store_path(vector_store_config, ndims=embed_dims)

        self._vector_store = create_vector_store(
            vector_store_config,
            embed_dims=embed_dims,
            persist_path=self._persist_path,
        )

        self._storage_context = self._create_storage_context()

        self._vector_index_retrieval_type = vector_index_retrieval_type
        self._vector_store_query_mode = retrieval_type_to_search_mode(
            vector_index_retrieval_type
        )

        self._similarity_top_k = similarity_top_k
        self._image_similarity_top_k = image_similarity_top_k

        self._enable_local_keyword_index = (
            enable_local_keyword_index
            and self.vector_store_config.type
            not in VECTOR_STORE_TYPES_WITH_HYBRID_SEARCH
        )

        self._local_bm25_index = None
        if self._enable_local_keyword_index:
            self._local_bm25_index = LocalBm25IndexStore(self._persist_path)
            self._retriever_weights = retriever_weights
            logger.info("Using local bm25 index.")
        else:
            self._retriever_weights = None

        logger.info(
            f"""
            Create PAI vector store index:
                Vector store type: {self.vector_store_config.type}
                Vector store path: {self._persist_path}
                Embedding model: {self._embed_model.model_name}
                Text embedding dims: {embed_dims}
                Enable local keyword index: {self._enable_local_keyword_index}
            """
        )
        self._vector_index = self._create_index()

    def _create_index(self):
        if os.path.exists(self._persist_path) and not self._vector_store.stores_text:
            ## Load from local FAISS store
            vector_index = load_index_from_storage(
                storage_context=self.storage_context,
                embed_model=self._embed_model,
                enable_local_keyword_index=self._enable_local_keyword_index,
                vector_index_retrieval_type=self._vector_index_retrieval_type,
                similarity_top_k=self._similarity_top_k,
                retriever_weights=self._retriever_weights,
            )
            logger.info(
                f"Loaded {len(vector_index.docstore.docs)} documents from local FAISS store."
            )
            return vector_index

        return PaiMultiModalVectorStoreIndex(
            nodes=[],
            storage_context=self.storage_context,
            embed_model=self._embed_model,
        )

    def _create_storage_context(self):
        doc_store = None
        index_store = None
        persist_dir = None

        # For faiss that don't stores text
        if not self._vector_store.stores_text:
            persist_dir = self._persist_path

            doc_store_path = os.path.join(
                self._persist_path, DocStoreTypes.DEFAULT_PERSIST_FNAME
            )
            if os.path.exists(doc_store_path):
                doc_store = SimpleDocumentStore.from_persist_path(doc_store_path)
            else:
                doc_store = SimpleDocumentStore()

            index_store_path = os.path.join(
                self._persist_path, IndexStoreTypes.DEFAULT_PERSIST_FNAME
            )
            if os.path.exists(index_store_path):
                index_store = SimpleIndexStore.from_persist_path(index_store_path)
            else:
                index_store = SimpleIndexStore()

        return StorageContext.from_defaults(
            docstore=doc_store,
            index_store=index_store,
            vector_store=self._vector_store,
            persist_dir=persist_dir,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        if (
            "vector_store_query_mode" not in kwargs
            or kwargs["vector_store_query_mode"] is None
        ):
            kwargs["vector_store_query_mode"] = self._vector_store_query_mode
        if "similarity_top_k" not in kwargs or kwargs["similarity_top_k"] is None:
            kwargs["similarity_top_k"] = self._similarity_top_k
        if (
            "image_similarity_top_k" not in kwargs
            or kwargs["image_similarity_top_k"] is None
        ):
            kwargs["image_similarity_top_k"] = self._image_similarity_top_k

        return self._vector_index.as_retriever(
            supports_hybrid_search=not self._enable_local_keyword_index,
            local_bm25_index=self._local_bm25_index,
            **kwargs,
        )

    def as_query_engine(self, llm: Any, **kwargs: Any) -> BaseQueryEngine:
        raise NotImplementedError

    def as_chat_engine(
        self, chat_mode: ChatMode = ChatMode.BEST, llm: Any = None, **kwargs: Any
    ) -> BaseChatEngine:
        raise NotImplementedError

    def insert_nodes(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        self._vector_index.insert_nodes(nodes, **insert_kwargs)
        if not self._vector_store.stores_text:
            self._storage_context.persist(self._persist_path)

        if self._enable_local_keyword_index:
            text_nodes = [node for node in nodes if isinstance(node, TextNode)]
            self._local_bm25_index.add_docs(text_nodes)

    def build_index_from_nodes(
        self, nodes: Sequence[BaseNode], **insert_kwargs: Any
    ) -> IndexDict:
        return self._vector_index.build_index_from_nodes(nodes, **insert_kwargs)

    def delete_nodes(
        self,
        node_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        if isinstance(self._vector_store, MyMilvusVectorStore) or isinstance(
            self._vector_store, MyElasticsearchStore
        ):
            return self._vector_index.delete_nodes(
                node_ids, delete_from_docstore, **delete_kwargs
            )
        else:
            logger.warning(
                "Currently delete_nodes supports for Milvus & ElasticSearch vector stores"
            )
            raise Exception(
                "Deleting nodes only supported for Milvus & Elasticsearch vectorstore."
            )

    async def adelete_nodes(
        self,
        node_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> Coroutine[Any, Any, None]:
        if isinstance(self._vector_store, MyMilvusVectorStore):
            await self._vector_store.adelete_nodes(
                node_ids, delete_from_docstore, **delete_kwargs
            )
            # delete from docstore only if needed
            if (
                not self._vector_store.stores_text or self._store_nodes_override
            ) and delete_from_docstore:
                for node_id in node_ids:
                    self._docstore.delete_document(node_id, raise_error=False)
                return
        else:
            logger.warning("Currently delete_nodes supports for Milvus vector store")
            raise NotImplementedError(
                "Deleting nodes only supported for Milvus & Elasticsearch vectorstore."
            )

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        if isinstance(self._vector_store, MyMilvusVectorStore) or isinstance(
            self._vector_store, MyElasticsearchStore
        ):
            return self._vector_index.delete_ref_doc(
                ref_doc_id, delete_from_docstore, **delete_kwargs
            )
        elif isinstance(self._vector_store, MyFaissVectorStore):
            ref_doc_info = self._vector_index.storage_context.docstore.get_ref_doc_info(
                ref_doc_id
            )
            if ref_doc_id is not None and ref_doc_info is not None:
                to_del_node_ids = ref_doc_info.node_ids.copy()
                if to_del_node_ids:
                    # 删除docstore中的doc并更新引用
                    for node_id in to_del_node_ids:
                        self._vector_index.storage_context.docstore.delete_document(
                            node_id
                        )
                        # 按键的数值顺序排序索引结构
                        sorted_items = sorted(
                            self._vector_index.index_struct.nodes_dict.items(),
                            key=lambda item: int(item[0]),
                        )
                        # 更新索引结构的节点引用
                        remaining_nodes_dict = {}
                        remaining_key = 0
                        to_del_keys = []
                        for k, v in sorted_items:
                            if v not in to_del_node_ids:
                                remaining_nodes_dict[str(remaining_key)] = v
                                remaining_key += 1
                            else:
                                to_del_keys.append(k)

                        logger.debug(
                            f"remaining_nodes_dict: {remaining_nodes_dict} , to_del_keys:{to_del_keys}"
                        )
                        self._vector_index.index_struct.nodes_dict = (
                            remaining_nodes_dict
                        )
                        self.storage_context.index_store.add_index_struct(
                            self._vector_index.index_struct
                        )
                        # 保存更改
                        self._vector_index.storage_context.persist(self._persist_path)
                        # 删除faiss中的id
                        self._vector_index._vector_store.remove_ids(to_del_keys)
            else:
                raise ValueError(
                    f"ref_doc_id {ref_doc_id} not found in faiss vector store."
                )
        else:
            logger.warning(
                "Currently delete_ref_doc supports for Milvus & ElasticSearch vector stores"
            )
            raise NotImplementedError(
                "Deleting docs only supported for Milvus & Elasticsearch vectorstore."
            )

    async def adelete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> Coroutine[Any, Any, None]:
        if isinstance(self._vector_store, MyMilvusVectorStore) or isinstance(
            self._vector_store, MyElasticsearchStore
        ):
            return await asyncio.to_thread(
                self._vector_index.delete_ref_doc(
                    ref_doc_id, delete_from_docstore, **delete_kwargs
                )
            )
        else:
            logger.warning("Currently delete_ref_doc supports for Milvus vector store")
            raise NotImplementedError(
                "Deleting docs only supported for Milvus & Elasticsearch vectorstore."
            )

    def clear(
        self,
    ):
        """清空索引中的内容并删除索引"""
        if isinstance(self._vector_store, MyMilvusVectorStore) or isinstance(
            self._vector_store, MyElasticsearchStore
        ):
            self._vector_store.clear()
        else:
            logger.warning(
                "Currently clear supports for Milvus & ElasticSearch vector stores"
            )
            raise NotImplementedError

    async def aclear(
        self,
    ):
        """清空索引中的内容并删除索引"""
        if isinstance(self._vector_store, MyMilvusVectorStore) or isinstance(
            self._vector_store, MyElasticsearchStore
        ):
            await self._vector_store.aclear()
        else:
            logger.warning(
                "Currently clear supports for Milvus & ElasticSearch vector stores"
            )
            raise NotImplementedError
