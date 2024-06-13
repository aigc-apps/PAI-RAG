import logging
import os
import sys
from typing import Dict, List, Any

from pai_rag.modules.index.my_vector_store_index import MyVectorStoreIndex
from pai_rag.modules.index.index_utils import load_index_from_storage
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

    def _get_embed_vec_dim(self, embed_model):
        # Get dimension size of embedding vector
        return len(embed_model._get_text_embedding("test"))

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        embed_model = new_params["EmbeddingModule"]
        embed_dims = self._get_embed_vec_dim(embed_model)
        persist_path = config.get("persist_path", DEFAULT_PERSIST_DIR)
        folder_name = get_store_persist_directory_name(config, embed_dims)
        store_path.persist_path = os.path.join(persist_path, folder_name)
        is_empty = not os.path.exists(store_path.persist_path)
        rag_store = RagStore(config, store_path.persist_path, is_empty, embed_dims)
        storage_context = rag_store.get_storage_context()

        if is_empty:
            return self.create_indices(storage_context, embed_model)
        else:
            return self.load_indices(storage_context, embed_model)

    def create_indices(self, storage_context, embed_model):
        logging.info("Empty index, need to create indices.")

        vector_index = MyVectorStoreIndex(
            nodes=[], storage_context=storage_context, embed_model=embed_model
        )
        logging.info("Created vector_index.")

        return vector_index

    def load_indices(self, storage_context, embed_model):
        if isinstance(storage_context.vector_store, FaissVectorStore):
            vector_index = load_index_from_storage(storage_context=storage_context)
            return vector_index
        else:
            vector_index = MyVectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=embed_model,
            )
        return vector_index
