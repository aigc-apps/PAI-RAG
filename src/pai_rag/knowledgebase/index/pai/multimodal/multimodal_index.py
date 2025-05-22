"""Multi Modal Vector Store Index.

An index that is built on top of multiple vector stores for different modalities.

"""

from typing import Any, List, Optional, Sequence, cast

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.data_structs.data_structs import (
    IndexDict,
    MultiModelIndexDict,
)
from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.core.indices.utils import (
    async_embed_image_nodes,
    async_embed_nodes,
    embed_image_nodes,
    embed_nodes,
)

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.llms.utils import LLMType
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.query_engine.multi_modal import SimpleMultiModalQueryEngine
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.settings import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import (
    DEFAULT_VECTOR_STORE,
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from pai_rag.knowledgebase.index.pai.multimodal.multimodal_retriever import (
    PaiMultiModalVectorIndexRetriever,
)
from loguru import logger


class PaiMultiModalVectorStoreIndex(VectorStoreIndex):
    """Multi-Modal Vector Store Index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    image_namespace = "image"
    index_struct_cls = MultiModelIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[MultiModelIndexDict] = None,
        embed_model: Optional[BaseEmbedding] = None,
        storage_context: Optional[StorageContext] = None,
        use_async: bool = False,
        store_nodes_override: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        storage_context = storage_context or StorageContext.from_defaults()

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=show_progress,
            use_async=use_async,
            store_nodes_override=store_nodes_override,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> PaiMultiModalVectorIndexRetriever:
        return PaiMultiModalVectorIndexRetriever(
            self,
            node_ids=list(self.index_struct.nodes_dict.values()),
            **kwargs,
        )

    def as_query_engine(
        self,
        llm: Optional[LLMType] = None,
        **kwargs: Any,
    ) -> SimpleMultiModalQueryEngine:
        retriever = cast(PaiMultiModalVectorIndexRetriever, self.as_retriever(**kwargs))

        llm = llm or Settings.llm
        assert isinstance(llm, MultiModalLLM)

        return SimpleMultiModalQueryEngine(
            retriever,
            multi_modal_llm=llm,
            **kwargs,
        )

    @classmethod
    def from_vector_store(
        cls,
        vector_store: BasePydanticVectorStore,
        embed_model: Optional[EmbedType] = None,
        # deprecated
        # Image-related kwargs
        image_vector_store: Optional[BasePydanticVectorStore] = None,
        image_embed_model: EmbedType = "clip",
        **kwargs: Any,
    ) -> "PaiMultiModalVectorStoreIndex":
        if not vector_store.stores_text:
            raise ValueError(
                "Cannot initialize from a vector store that does not store text."
            )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, image_store=image_vector_store
        )

        return cls(
            nodes=[],
            storage_context=storage_context,
            image_embed_model=image_embed_model,
            embed_model=(
                resolve_embed_model(
                    embed_model, callback_manager=kwargs.get("callback_manager", None)
                )
                if embed_model
                else Settings.embed_model
            ),
            **kwargs,
        )

    def _get_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        is_image: bool = False,
    ) -> List[BaseNode]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        if is_image:
            id_to_embed_map = embed_image_nodes(
                nodes,
                embed_model=None,
                show_progress=show_progress,
            )
        else:
            id_to_embed_map = embed_nodes(
                nodes,
                embed_model=self._embed_model,
                show_progress=show_progress,
            )

        results = []
        for node in nodes:
            if node.embedding is None:
                embedding = id_to_embed_map[node.node_id]
                result = node.copy()
                result.embedding = embedding
            else:
                result = node.copy()

            results.append(result)
        return results

    async def _aget_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        is_image: bool = False,
    ) -> List[BaseNode]:
        """Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        if is_image:
            id_to_embed_map = await async_embed_image_nodes(
                nodes,
                embed_model=None,
                show_progress=show_progress,
            )
        else:
            id_to_embed_map = await async_embed_nodes(
                nodes,
                embed_model=self._embed_model,
                show_progress=show_progress,
            )

        results = []
        for node in nodes:
            if node.embedding is None:
                embedding = id_to_embed_map[node.node_id]
                result = node.copy()
                result.embedding = embedding
            else:
                result = node.copy()

            results.append(result)
        return results

    async def _async_add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Asynchronously add nodes to index."""
        if not nodes:
            return

        text_nodes: List[BaseNode] = []
        new_text_ids: List[str] = []

        for node in nodes:
            if isinstance(node, TextNode) and node.text:
                text_nodes.append(node)
        if len(text_nodes) > 0:
            # embed all nodes as text - include image nodes that have text attached
            text_nodes = await self._aget_node_with_embedding(
                text_nodes, show_progress, is_image=False
            )
            new_text_ids = await self.storage_context.vector_stores[
                DEFAULT_VECTOR_STORE
            ].async_add(text_nodes, **insert_kwargs)

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        if not self._vector_store.stores_text or self._store_nodes_override:
            for node, new_id in zip(text_nodes, new_text_ids):
                # NOTE: remove embedding from node to avoid duplication
                node_without_embedding = node.copy()
                node_without_embedding.embedding = None

                index_struct.add_node(node_without_embedding, text_id=new_id)
                self._docstore.add_documents(
                    [node_without_embedding], allow_update=True
                )

    def _add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Add document to index."""
        if not nodes:
            return

        text_nodes: List[BaseNode] = []
        new_text_ids: List[str] = []

        for node in nodes:
            if isinstance(node, TextNode) and node.text:
                text_nodes.append(node)

        if len(text_nodes) > 0:
            # embed all nodes as text - include image nodes that have text attached
            text_nodes = self._get_node_with_embedding(
                text_nodes, show_progress, is_image=False
            )
            logger.info(f"Added {len(text_nodes)} text nodes to index.")
            new_text_ids = self.storage_context.vector_stores[DEFAULT_VECTOR_STORE].add(
                text_nodes, **insert_kwargs
            )
            origin_text_ids = [node.node_id for node in text_nodes]
            logger.info(f"Added {len(text_nodes)} TextNodes to index.")
            logger.debug(f"NodeIds: {origin_text_ids}, new NodeIds: {new_text_ids}.")

        # if the vector store doesn't store text, we need to add the nodes to the
        # index struct and document store
        if not self._vector_store.stores_text or self._store_nodes_override:
            for node, new_id in zip(text_nodes, new_text_ids):
                # NOTE: remove embedding from node to avoid duplication
                node_without_embedding = node.copy()
                node_without_embedding.embedding = None

                index_struct.add_node(node_without_embedding, text_id=new_id)
                self._docstore.add_documents(
                    [node_without_embedding], allow_update=True
                )

    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        # delete from all vector stores

        for vector_store in self._storage_context.vector_stores.values():
            vector_store.delete(ref_doc_id)

            if self._store_nodes_override or self._vector_store.stores_text:
                ref_doc_info = self._docstore.get_ref_doc_info(ref_doc_id)
                if ref_doc_info is not None:
                    for node_id in ref_doc_info.node_ids:
                        self._index_struct.delete(node_id)
                        self._vector_store.delete(node_id)

        if delete_from_docstore:
            self._docstore.delete_ref_doc(ref_doc_id, raise_error=False)

        self._storage_context.index_store.add_index_struct(self._index_struct)
