import logging
import os
import sys
from typing import Dict, List, Any

from llama_index.core import VectorStoreIndex

from llama_index.core import load_index_from_storage
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.index.store import RagStore
from llama_index.vector_stores.faiss import FaissVectorStore
from pai_rag.utils.store_utils import get_store_persist_directory_name, store_path

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_PERSIST_DIR = "./storage"


class IndexModule(ConfigurableModule):
    """Class for managing indices.

    RagIndex to manage vector indices for RagApplication.
    When initializing, the index is empty or load from existing index.
    User can add nodes to index when needed.
    """

    @staticmethod
    def get_dependencies() -> List[str]:
        return ["EmbeddingModule"]

    def _get_embed_vec_dim(self):
        # Get dimension size of embedding vector
        return len(self.embed_model._get_text_embedding("test"))

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.config = new_params[MODULE_PARAM_CONFIG]
        self.embed_model = new_params["EmbeddingModule"]
        self.embed_dims = self._get_embed_vec_dim()
        persist_path = self.config.get("persist_path", DEFAULT_PERSIST_DIR)
        folder_name = get_store_persist_directory_name(self.config, self.embed_dims)
        store_path.persist_path = os.path.join(persist_path, folder_name)

        self.is_empty = not os.path.exists(store_path.persist_path)
        rag_store = RagStore(
            self.config, store_path.persist_path, self.is_empty, self.embed_dims
        )
        self.storage_context = rag_store.get_storage_context()

        if self.is_empty:
            return self.create_indices()
        else:
            return self.load_indices()

    def create_indices(self):
        logging.info("Empty index, need to create indices.")

        vector_index = VectorStoreIndex(
            nodes=[], storage_context=self.storage_context, embed_model=self.embed_model
        )
        logging.info("Created vector_index.")

        return vector_index

    def load_indices(self):
        if isinstance(self.storage_context.vector_store, FaissVectorStore):
            vector_index = load_index_from_storage(storage_context=self.storage_context)
        else:
            vector_index = VectorStoreIndex(
                nodes=[],
                storage_context=self.storage_context,
                embed_model=self.embed_model,
            )
        return vector_index
