import logging
from typing import Dict, List, Any
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.index.pai.vector_store_config import PaiVectorIndexConfig
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class IndexModule(ConfigurableModule):
    """Class for managing indices.

    RagIndex to manage vector indices for RagApplication.
    When initializing, the index is empty or load from existing index.
    User can add nodes to index when needed.
    """

    @staticmethod
    def get_dependencies() -> List[str]:
        return ["EmbeddingModule", "MultiModalEmbeddingModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        index_config = PaiVectorIndexConfig.model_validate(config)
        embed_model = new_params["EmbeddingModule"]
        multi_modal_embed_model = new_params["MultiModalEmbeddingModule"]
        return PaiVectorStoreIndex(
            vector_store_config=index_config.vector_store,
            embed_model=embed_model,
            enable_multimodal=index_config.enable_multimodal,
            multi_modal_embed_model=multi_modal_embed_model,
            enable_local_keyword_index=True,
        )
