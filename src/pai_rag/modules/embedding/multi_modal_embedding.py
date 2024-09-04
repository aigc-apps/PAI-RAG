from typing import Dict, List, Any
from pai_rag.integrations.embeddings.clip.cnclip_embedding import CnClipEmbedding
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
        if config is None:
            logger.info("Don't use Multi-Modal.")
            return None
        source = config["source"].lower()
        embed_batch_size = config.get("embed_batch_size", DEFAULT_EMBED_BATCH_SIZE)

        if not isinstance(embed_batch_size, int):
            raise TypeError("embed_batch_size must be of type int")

        if source == "cnclip":
            multi_modal_embed_model = CnClipEmbedding(embed_batch_size=embed_batch_size)
            logger.info(
                f"Initialized CnClip MultiModal embedding model with {embed_batch_size} batch size."
            )

        else:
            return None

        return multi_modal_embed_model
