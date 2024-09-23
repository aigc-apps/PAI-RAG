from typing import Dict, List, Any
from llama_index.core import Settings
from pai_rag.integrations.embeddings.pai.pai_embedding_config import parse_embed_config
from pai_rag.integrations.embeddings.pai.pai_multimodal_embedding import (
    PaiMultiModalEmbedding,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
import logging

logger = logging.getLogger(__name__)

DEFAULT_EMBED_BATCH_SIZE = 10


class MultiModalEmbeddingModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        if config is not None and config.get("source"):
            embed_config = parse_embed_config(config)
            multi_modal_embed_model = PaiMultiModalEmbedding(
                multi_modal_embed_config=embed_config
            )
            logger.info(
                f"""
                MultiModal Embedding Module created with config:
                {config}
            """
            )
        else:
            multi_modal_embed_model = None
            logger.info(
                f"""
                No MultiModal Embedding Module created with config:
                {config}
            """
            )

        if multi_modal_embed_model:
            multi_modal_embed_model.callback_manager = Settings.callback_manager

        return multi_modal_embed_model
