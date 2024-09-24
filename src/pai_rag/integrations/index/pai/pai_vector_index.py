import logging
import os
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
from pai_rag.integrations.index.pai.local.local_bm25_index import LocalBm25IndexStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K

logger = logging.getLogger(__name__)


def retrieval_type_to_search_mode(retrieval_type: VectorIndexRetrievalType):
    if retrieval_type == VectorIndexRetrievalType.keyword:
        return VectorStoreQueryMode.TEXT_SEARCH
    elif retrieval_type == VectorIndexRetrievalType.hybrid:
        return VectorStoreQueryMode.HYBRID
    else:
        return VectorStoreQueryMode.DEFAULT


class PaiVectorStoreIndex(VectorStoreIndex):
    _vector_store: BasePydanticVectorStore = PrivateAttr()
    _image_store: BasePydanticVectorStore = PrivateAttr()
    _embed_model: BaseEmbedding = PrivateAttr()
    _storage_context: StorageContext = PrivateAttr()
    _vector_index: VectorStoreIndex = PrivateAttr()
    _multi_modal_embed_model: BaseEmbedding = PrivateAttr()

    _persist_path: str
    # Enable local keyword index for

    def __init__(
        self,
        vector_store_config: BaseVectorStoreConfig,
        embed_model: BaseEmbedding,
        enable_multimodal: bool = False,
        multi_modal_embed_model: BaseEmbedding = None,
        enable_local_keyword_index: bool = False,
        vector_index_retrieval_type: VectorIndexRetrievalType = VectorIndexRetrievalType.embedding,
        similarity_top_k=DEFAULT_SIMILARITY_TOP_K,
        image_similarity_top_k=DEFAULT_SIMILARITY_TOP_K,
        retriever_weights: List[float] = None,
    ):
        self.vector_store_config = vector_store_config
        self._enable_multimodal = enable_multimodal
        self._image_store = None

        embed_dims = len(embed_model.get_text_embedding("0"))
        self._embed_model = embed_model

        # change persist path to subfolder
        self.vector_store_config.persist_path = resolve_store_path(
            vector_store_config, ndims=embed_dims
        )
        self._persist_path = self.vector_store_config.persist_path

        self._vector_store = create_vector_store(
            vector_store_config, embed_dims=embed_dims
        )
        multi_modal_embed_dims = -1  # multimodal not enabled
        # assert multi_modal_embed_model is not None, "Multi-modal embedding model must be provided."
        if self._enable_multimodal:
            multi_modal_embed_dims = len(
                multi_modal_embed_model.get_text_embedding("0")
            )
            self._multi_modal_embed_model = multi_modal_embed_model

            self._image_store = create_vector_store(
                vectordb_config=vector_store_config,
                embed_dims=multi_modal_embed_dims,
                is_image_store=True,
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

        logger.info(
            f"""
            Create PAI vector store index:
                Vector store type: {self.vector_store_config.type}
                Vector store path: {self._persist_path}
                Embedding model: {self._embed_model.model_name}
                Enable multimodal: {self._enable_multimodal}
                Text embedding dims: {embed_dims}
                Image embedding dims: {multi_modal_embed_dims}
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
                image_embed_model=self._multi_modal_embed_model,
                enable_multimodal=self._enable_multimodal,
                enable_local_keyword_index=self._enable_local_keyword_index,
                vector_index_retrieval_type=self._vector_index_retrieval_type,
                similarity_top_k=self._similarity_top_k,
                image_similarity_top_k=self._image_similarity_top_k,
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
            enable_multimodal=self._enable_multimodal,
            image_embed_model=self._multi_modal_embed_model,
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
            image_store=self._image_store,
            persist_dir=persist_dir,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        if (
            "vector_store_query_mode" not in kwargs
            or kwargs["vector_store_query_mode"] is None
        ):
            kwargs["similarity_top_k"] = self._vector_store_query_mode
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

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        return self._vector_index.delete_ref_doc(
            ref_doc_id, delete_from_docstore, **delete_kwargs
        )

    def delete_nodes(
        self,
        node_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        return self._vector_index.delete_nodes(
            node_ids, delete_from_docstore, **delete_kwargs
        )

    def adelete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> Coroutine[Any, Any, None]:
        return self._vector_index.adelete_ref_doc(
            ref_doc_id, delete_from_docstore, **delete_kwargs
        )
