from typing import Dict, List, Any
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.embedding.my_huggingface_embedding import MyHuggingFaceEmbedding
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_EMBED_BATCH_SIZE = 10
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "bge-small-zh-v1.5"


class EmbeddingModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        source = config["source"].lower()
        embed_batch_size = config.get("embed_batch_size", DEFAULT_EMBED_BATCH_SIZE)

        if not isinstance(embed_batch_size, int):
            raise TypeError("embed_batch_size must be of type int")

        if source == "openai":
            embed_model = OpenAIEmbedding(
                api_key=config.get("api_key", None),
                embed_batch_size=embed_batch_size,
            )
            logger.info(
                f"Initialized Open AI embedding model with {embed_batch_size} batch size."
            )

        elif source == "azureopenai":
            embed_model = AzureOpenAIEmbedding(
                api_key=config.get("api_key", None),
                embed_batch_size=embed_batch_size,
            )
            logger.info(
                f"Initialized Azure Open AI embedding model with {embed_batch_size} batch size."
            )

        elif source == "huggingface":
            model_dir = config.get("model_dir", DEFAULT_MODEL_DIR)
            model_name = config.get("model_name", DEFAULT_HUGGINGFACE_EMBEDDING_MODEL)

            model_path = os.path.join(model_dir, model_name)
            embed_model = MyHuggingFaceEmbedding(
                model_name=model_path,
                embed_batch_size=embed_batch_size,
            )

            logger.info(
                f"Initialized HuggingFace embedding model {model_name} with {embed_batch_size} batch size."
            )

        elif source == "dashscope":
            embed_model = DashScopeEmbedding(
                api_key=config.get("api_key", None),
                embed_batch_size=embed_batch_size,
            )
            logger.info(
                f"Initialized DashScope embedding model with {embed_batch_size} batch size."
            )
        else:
            raise ValueError(f"Unknown Embedding source: {config['source']}")

        Settings.embed_model = embed_model
        return embed_model
