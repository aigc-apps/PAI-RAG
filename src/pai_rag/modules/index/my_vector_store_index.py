"""Base vector store index.

An index that is built on top of an existing vector store.

"""

import logging
from typing import Any, Sequence
from queue import Queue, Empty
import threading
from llama_index.core import VectorStoreIndex
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    IndexNode,
)
from llama_index.core.utils import iter_batch

logger = logging.getLogger(__name__)


class MyVectorStoreIndex(VectorStoreIndex):
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

        for nodes_batch in iter_batch(nodes, self._insert_batch_size):
            nodes_batch = await self._aget_node_with_embedding(
                nodes_batch, show_progress
            )
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
            else:
                # NOTE: if the vector store keeps text,
                # we only need to add image and index nodes
                for node, new_id in zip(nodes_batch, new_ids):
                    if isinstance(node, (ImageNode, IndexNode)):
                        # NOTE: remove embedding from node to avoid duplication
                        node_without_embedding = node.copy()
                        node_without_embedding.embedding = None

                        index_struct.add_node(node_without_embedding, text_id=new_id)
                        self._docstore.add_documents(
                            [node_without_embedding], allow_update=True
                        )

    def _add_nodes_batch_to_index(
        self,
        q: Queue,
        index_struct,
        insert_kwargs,
    ):
        i = 0
        while True:
            try:
                nodes_batch = q.get(timeout=3)
            except Empty:
                continue

            if nodes_batch is None:
                q.task_done()
                return

            i += 1
            print(f"Consuming batch {i}, batch size {len(nodes_batch)}")
            new_ids = self._vector_store.add(nodes_batch, **insert_kwargs)

            if not self._vector_store.stores_text or self._store_nodes_override:
                print("saving to docstore!!!")
                # NOTE: if the vector store doesn't store text,
                # we need to add the nodes to the index struct and document store
                for node, new_id in zip(nodes_batch, new_ids):
                    # NOTE: remove embedding from node to avoid duplication
                    node_without_embedding = node.copy()
                    node_without_embedding.embedding = None

                    index_struct.add_node(node_without_embedding, text_id=new_id)
                    self._docstore.add_documents(
                        [node_without_embedding], allow_update=True
                    )
            else:
                print("Skipping saving to docstore!!!")
                # NOTE: if the vector store keeps text,
                # we only need to add image and index nodes
                for node, new_id in zip(nodes_batch, new_ids):
                    if isinstance(node, (ImageNode, IndexNode)):
                        # NOTE: remove embedding from node to avoid duplication
                        node_without_embedding = node.copy()
                        node_without_embedding.embedding = None

                        index_struct.add_node(node_without_embedding, text_id=new_id)
                        self._docstore.add_documents(
                            [node_without_embedding], allow_update=True
                        )
            q.task_done()

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

        q = Queue(maxsize=100)

        work_thread = threading.Thread(
            target=self._add_nodes_batch_to_index, args=(q, index_struct, insert_kwargs)
        )
        work_thread.start()

        i = 0
        for nodes_batch in iter_batch(nodes, 500):
            nodes_batch = self._get_node_with_embedding(nodes_batch, show_progress)
            q.put(nodes_batch)
            i += 1
            print(f"produced batch {i}, batch size {len(nodes_batch)}")

        q.put(None)
        q.join()
