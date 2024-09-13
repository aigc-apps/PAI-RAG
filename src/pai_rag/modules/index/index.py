import logging
import os
from typing import Dict, List, Any
from llama_index.vector_stores.faiss import FaissVectorStore
from pai_rag.integrations.index.multi_modal_index import MyMultiModalVectorStoreIndex
from pai_rag.modules.index.index_utils import load_index_from_storage
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.index.my_vector_store_index import MyVectorStoreIndex
from pai_rag.modules.index.store import RagStore
from pai_rag.utils.store_utils import get_store_persist_directory_name
from pai_rag.modules.index.index_entry import index_entry

DEFAULT_PERSIST_DIR = "./storage"

logger = logging.getLogger(__name__)


class RagIndex:
    def __init__(self, config, embed_model, multi_modal_embed_model, postprocessor):
        self.config = config
        self.embed_model = embed_model
        self.multi_modal_embed_model = multi_modal_embed_model
        self.embed_dims = self._get_embed_vec_dim(embed_model)
        if self.multi_modal_embed_model:
            self.multi_modal_embed_dims = self._get_embed_vec_dim(
                self.multi_modal_embed_model
            )
        else:
            self.multi_modal_embed_dims = 0
        persist_path = config.get("persist_path", DEFAULT_PERSIST_DIR)
        folder_name = get_store_persist_directory_name(config, self.embed_dims)
        self.persist_path = os.path.join(persist_path, folder_name)
        self.postprocessor = postprocessor
        index_entry.register(self.persist_path)

        is_empty = not os.path.exists(self.persist_path)
        rag_store = RagStore(
            config,
            self.postprocessor,
            self.persist_path,
            is_empty,
            self.embed_dims,
            self.multi_modal_embed_dims,
        )
        self.storage_context = rag_store.get_storage_context()

        self.vectordb_type = config["vector_store"].get("type", "faiss").lower()
        if is_empty:
            self.create_indices(self.storage_context, embed_model)
        else:
            self.load_indices(self.storage_context, embed_model)

    def _get_embed_vec_dim(self, embed_model):
        # Get dimension size of embedding vector
        return len(embed_model._get_text_embedding("test"))

    def create_indices(self, storage_context, embed_model):
        logging.info("Empty index, need to create indices.")

        if self.multi_modal_embed_model:
            self.vector_index = MyMultiModalVectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=embed_model,
                image_embed_model=self.multi_modal_embed_model,
            )
        else:
            self.vector_index = MyVectorStoreIndex(
                nodes=[], storage_context=storage_context, embed_model=embed_model
            )

        logging.info("Created vector_index.")

    def load_indices(self, storage_context, embed_model):
        if isinstance(storage_context.vector_store, FaissVectorStore):
            self.vector_index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=embed_model,
                image_embed_model=self.multi_modal_embed_model,
            )
        else:
            if self.multi_modal_embed_model:
                self.vector_index = MyMultiModalVectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context,
                    embed_model=embed_model,
                    image_embed_model=self.multi_modal_embed_model,
                )
            else:
                self.vector_index = MyVectorStoreIndex(
                    nodes=[], storage_context=storage_context, embed_model=embed_model
                )

    def reload(self):
        if isinstance(self.storage_context.vector_store, FaissVectorStore):
            rag_store = RagStore(
                self.config,
                self.postprocessor,
                self.persist_path,
                False,
                self.embed_dims,
                self.multi_modal_embed_dims,
            )
            self.storage_context = rag_store.get_storage_context()

            self.vector_index = load_index_from_storage(
                storage_context=self.storage_context,
                image_embed_model=self.multi_modal_embed_model,
            )
            logger.info(
                f"FaissIndex {self.persist_path} reloaded with {len(self.vector_index.docstore.docs)} nodes."
            )
        return


class IndexModule(ConfigurableModule):
    """Class for managing indices.

    RagIndex to manage vector indices for RagApplication.
    When initializing, the index is empty or load from existing index.
    User can add nodes to index when needed.
    """

    @staticmethod
    def get_dependencies() -> List[str]:
        return ["EmbeddingModule", "PostprocessorModule", "MultiModalEmbeddingModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        embed_model = new_params["EmbeddingModule"]
        postprocessor = new_params["PostprocessorModule"]
        multi_modal_embed_model = new_params["MultiModalEmbeddingModule"]
        print("multi_modal_embed_model", multi_modal_embed_model)
        return RagIndex(config, embed_model, multi_modal_embed_model, postprocessor)
