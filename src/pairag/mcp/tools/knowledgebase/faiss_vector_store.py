"""Faiss Vector store index. To be deleted in the future

An index that is built on top of an existing vector store.

"""

import time
import json
import logging
import os
from typing import Any, Dict, List, Optional, cast
import threading

import fsspec
import numpy as np
from fsspec.implementations.local import LocalFileSystem
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.core.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic import Field

logger = logging.getLogger()

DEFAULT_PERSIST_PATH = os.path.join(
    DEFAULT_PERSIST_DIR, f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
)


class FaissVectorStore(BasePydanticVectorStore):
    """Faiss Vector Store.

    Embeddings are stored within a Faiss index.

    During query time, the index uses Faiss to query for the top
    k embeddings, and returns the corresponding indices.

    Args:
        faiss_index (faiss.Index): Faiss index instance

    Examples:
        `pip install llama-index-vector-stores-faiss faiss-cpu`

        ```python
        from llama_index.vector_stores.faiss import FaissVectorStore
        import faiss

        # create a faiss index
        d = 1536  # dimension
        faiss_index = faiss.IndexFlatL2(d)

        vector_store = FaissVectorStore(faiss_index=faiss_index)
        ```
    """

    stores_text: bool = True

    _faiss_index = PrivateAttr()
    faiss_id_map: Dict[str, dict] = Field(default={})
    faiss_index_path: str = Field(default=None)
    faiss_id_path: str = Field(default=None)
    inverted_id_map: Dict[str, str] = Field(default={})
    last_mtime: float = Field(default=0)
    _monitor_thread = PrivateAttr()

    def __init__(
        self,
        faiss_index: Any,
        faiss_id_map: Dict[str, dict] = {},
        faiss_index_path: str = DEFAULT_PERSIST_PATH,
        faiss_id_path: str = None,
    ) -> None:
        """Initialize params."""
        import_err_msg = """
            `faiss` package not found. For instructions on
            how to install `faiss` please visit
            https://github.com/facebookresearch/faiss/wiki/Installing-Faiss
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(import_err_msg)

        super().__init__()

        self._faiss_index = cast(faiss.Index, faiss_index)
        self.faiss_id_map = faiss_id_map
        self.faiss_id_path = faiss_id_path
        self.faiss_index_path = faiss_index_path
        self.inverted_id_map = {v["faiss_id"]: k for k, v in faiss_id_map.items()}

        self.last_mtime = -1
        if os.path.exists(self.faiss_id_path):
            self.last_mtime = os.path.getmtime(self.faiss_id_path)
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def _monitor_loop(self):
        while True:
            try:
                current_mtime = -1
                if os.path.exists(self.faiss_id_path):
                    current_mtime = os.path.getmtime(self.faiss_id_path)
                if current_mtime > self.last_mtime:
                    import faiss

                    logger.info("Detected index file change, reloading index...")
                    self.last_mtime = current_mtime
                    self._faiss_index = faiss.read_index(self.faiss_index_path)
                    self.faiss_id_map = json.load(open(self.faiss_id_path, "r"))
                    self.inverted_id_map = {
                        v["faiss_id"]: k for k, v in self.faiss_id_map.items()
                    }
                    logger.info(
                        f"Update faiss from background: Loaded {self._faiss_index.ntotal} docs from {self.faiss_index_path}."
                    )
            except Exception as e:
                logger.error(
                    f"Update faiss from background: Failed to load faiss index from {self.faiss_index_path}: {e}."
                )

            time.sleep(5)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        dimension: int,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "FaissVectorStore":
        import faiss

        # I don't think FAISS supports fsspec, it requires a path in the SWIG interface
        # TODO: copy to a temp file and load into memory from there
        if fs and not isinstance(fs, LocalFileSystem):
            raise NotImplementedError("FAISS only supports local storage for now.")

        os.makedirs(persist_dir, exist_ok=True)
        faiss_index_path = os.path.join(persist_dir, "faiss_index.bin")
        faiss_id_path = os.path.join(persist_dir, "faiss_ids.json")

        if not os.path.exists(faiss_index_path):
            logger.info(f"Creating new {__name__} from {faiss_index_path}.")
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss.write_index(faiss_index, faiss_index_path)
        else:
            logger.info(f"Loading {__name__} from {faiss_index_path}.")
            faiss_index = faiss.read_index(faiss_index_path)
            logger.info(f"Loaded {faiss_index.ntotal} docs from {faiss_index_path}.")

        if not os.path.exists(faiss_id_path):
            faiss_id_map = {}
        else:
            faiss_id_map = json.load(open(faiss_id_path, "r"))

        return cls(
            faiss_index=faiss_index,
            faiss_id_map=faiss_id_map,
            faiss_index_path=faiss_index_path,
            faiss_id_path=faiss_id_path,
        )

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        NOTE: in the Faiss vector store, we do not store text in Faiss.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        logger.warning(f"### add {len(nodes)} nodes to index.")
        import faiss

        new_ids = []
        for node in nodes:
            text_embedding = node.get_embedding()
            text_embedding_np = np.array(text_embedding, dtype="float32")[np.newaxis, :]
            new_id = str(self._faiss_index.ntotal)
            self._faiss_index.add(text_embedding_np)
            new_ids.append(new_id)
            self.faiss_id_map[node.id_] = {
                "id": node.id_,
                "metadata": node.metadata,
                "text": node.text,
                "faiss_id": new_id,
            }
            self.inverted_id_map[new_id] = node.id_

        faiss.write_index(self._faiss_index, self.faiss_index_path)

        with open(self.faiss_id_path, "w") as id_file:
            id_file.write(json.dumps(self.faiss_id_map))

        self.last_mtime = os.path.getmtime(self.faiss_id_path)
        logger.info(f"Saved {len(nodes)} chunks to FAISS successfully.")
        return new_ids

    @property
    def client(self) -> Any:
        """Return the faiss index."""
        return self._faiss_index

    def delete_nodes(self, node_ids: List[str]):
        import faiss

        logger.info(f"Deleting {len(node_ids)} nodes from FAISS vector store")
        ids_to_delete = []
        for node_id in node_ids:
            if node_id in self.faiss_id_map:
                ids_to_delete.append(self.faiss_id_map[node_id]["faiss_id"])

        num_deleted = self._faiss_index.remove_ids(np.array(ids_to_delete))
        faiss.write_index(self._faiss_index, self.faiss_index_path)

        for node_id in node_ids:
            if node_id in self.faiss_id_map:
                del self.faiss_id_map[node_id]

        with open(self.faiss_id_path, "w") as id_file:
            id_file.write(json.dumps(self.faiss_id_map))
        self.last_mtime = os.path.getmtime(self.faiss_id_path)

        self.inverted_id_map = {v: k for k, v in self.faiss_id_map.items()}
        logger.info(f"Deleted {num_deleted} nodes from FAISS vector store.")

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("Delete not yet implemented for Faiss index.")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for Faiss yet.")

        query_embedding = cast(List[float], query.query_embedding)
        query_embedding_np = np.array(query_embedding, dtype="float32")[np.newaxis, :]
        dists, indices = self._faiss_index.search(
            query_embedding_np, query.similarity_top_k
        )
        dists = list(dists[0])
        # if empty, then return an empty response
        if len(indices) == 0:
            return VectorStoreQueryResult(similarities=[], ids=[], nodes=[])

        # returned dimension is 1 x k
        node_idxs = indices[0]

        filtered_dists = []
        filtered_node_idxs = []
        nodes = []
        for dist, idx in zip(dists, node_idxs):
            if idx < 0:
                continue
            filtered_dists.append(dist)
            node_id = self.inverted_id_map[str(idx)]
            filtered_node_idxs.append(node_id)

            node = TextNode(
                id=node_id,
                text=self.faiss_id_map[node_id]["text"],
                metadata=self.faiss_id_map[node_id]["metadata"],
            )
            nodes.append(node)

        return VectorStoreQueryResult(
            similarities=filtered_dists,
            ids=filtered_node_idxs,
            nodes=nodes,
        )
