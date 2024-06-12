"""Base vector store index.

An index that is built on top of an existing vector store.

"""

import asyncio
import logging
from typing import Any, Sequence
from llama_index.core import VectorStoreIndex
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
)
from llama_index.core.utils import iter_batch

logger = logging.getLogger(__name__)


def call_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)


class MyVectorStoreIndex(VectorStoreIndex):
    async def _postprocess_batch(
        self,
        index_struct: IndexDict,
        nodes_batch: Sequence[BaseNode],
        **insert_kwargs: Any,
    ):
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

        batch_process_coroutines = []

        for nodes_batch in iter_batch(nodes, self._insert_batch_size):
            nodes_batch = await self._aget_node_with_embedding(
                nodes_batch, show_progress
            )

            batch_process_coroutines.append(
                self._postprocess_batch(index_struct, nodes_batch, **insert_kwargs)
            )
        await asyncio.gather(*batch_process_coroutines)

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
