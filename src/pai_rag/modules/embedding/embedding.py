from typing import Dict, List, Any
from llama_index.core import Settings
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding
from pai_rag.integrations.embeddings.pai.pai_embedding_config import parse_embed_config
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
import logging

logger = logging.getLogger(__name__)

DEFAULT_EMBED_BATCH_SIZE = 10
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "bge-large-zh-v1.5"


class EmbeddingModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]

        embed_config = parse_embed_config(config)
        embed_model = PaiEmbedding(embed_config=embed_config)
        logger.info(
            f"""
            Embedding Module created with config:
            {config}
        """
        )

        Settings.embed_model = embed_model
        return embed_model
