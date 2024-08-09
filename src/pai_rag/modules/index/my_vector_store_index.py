"""Base vector store index.

An index that is built on top of an existing vector store.

"""

import asyncio
import logging
from typing import Any, List, Sequence
from llama_index.core import VectorStoreIndex
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
)
from llama_index.core.utils import iter_batch

from pai_rag.utils.embed_utils import async_embed_nodes

logger = logging.getLogger(__name__)


class MyVectorStoreIndex(VectorStoreIndex):
    async def _process_one_batch(
        self,
        nodes_batch: Sequence[Sequence[BaseNode]],
        index_struct: IndexDict,
        semaphore: asyncio.Semaphore,
        **insert_kwargs: Any,
    ):
        async with semaphore:
            new_ids = await self._vector_store.async_add(nodes_batch, **insert_kwargs)

            # if the vector store doesn't store text, we need to add the nodes to the
            # index struct and document store
            if not self._vector_store.stores_text or self._store_nodes_override:
                for node, new_id in zip(nodes_batch, new_ids):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )

    async def _postprocess_all_batch(
        self,
        nodes_batch_list: Sequence[Sequence[BaseNode]],
        index_struct: IndexDict,
        **insert_kwargs: Any,
    ):
        asyncio_semaphore = asyncio.Semaphore(10)
        batch_process_coroutines = []
        for nodes_batch in nodes_batch_list:
            batch_process_coroutines.append(
                self._process_one_batch(
                    nodes_batch, index_struct, asyncio_semaphore, **insert_kwargs
                )
            )
        await asyncio.gather(*batch_process_coroutines)

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

        node_batch_list = []
        for nodes_batch in iter_batch(nodes, 100):
            nodes_batch = await self._aget_node_with_embedding(
                nodes, show_progress=show_progress
            )
            node_batch_list.append(nodes_batch)
        await self._postprocess_all_batch(
            node_batch_list, index_struct, **insert_kwargs
        )

    async def _insert_async(
        self, nodes: Sequence[BaseNode], **insert_kwargs: Any
    ) -> None:
        """Insert a document."""
        await self._async_add_nodes_to_index(
            self._index_struct, nodes, show_progress=True, **insert_kwargs
        )

    async def insert_nodes_async(
        self, nodes: Sequence[BaseNode], **insert_kwargs: Any
    ) -> None:
        """Insert nodes.

        NOTE: overrides BaseIndex.insert_nodes.
            VectorStoreIndex only stores nodes in document store
            if vector store does not store text
        """
        for node in nodes:
            if isinstance(node, IndexNode):
                try:
                    node.dict()
                except ValueError:
                    self._object_map[node.index_id] = node.obj
                    node.obj = None

        with self._callback_manager.as_trace("insert_nodes"):
            await self._insert_async(nodes, **insert_kwargs)
            self._storage_context.index_store.add_index_struct(self._index_struct)

    async def _aget_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = await async_embed_nodes(
            nodes=nodes,
            embed_model=self._embed_model,
            show_progress=show_progress,
        )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.copy()
            result.embedding = embedding
            results.append(result)
        return results
