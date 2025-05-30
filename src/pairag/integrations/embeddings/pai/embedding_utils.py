import os
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pairag.utils.cuda_utils import infer_cuda_device
from pairag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
    DashScopeEmbeddingConfig,
    OpenAIEmbeddingConfig,
    HuggingFaceEmbeddingConfig,
)
from pairag.utils.mdoelscope_utils import download_model_to_directory

from loguru import logger


def create_embedding(embed_config: PaiBaseEmbeddingConfig, model_dir: str = None):
    if isinstance(embed_config, OpenAIEmbeddingConfig):
        if embed_config.model is not None:
            embed_model = OpenAIEmbedding(
                api_key=embed_config.api_key,
                api_base=embed_config.api_base,
                embed_batch_size=embed_config.embed_batch_size,
                callback_manager=Settings.callback_manager,
                model=embed_config.model,
            )
        else:
            embed_model = OpenAIEmbedding(
                api_key=embed_config.api_key,
                api_base=embed_config.api_base,
                embed_batch_size=embed_config.embed_batch_size,
                callback_manager=Settings.callback_manager,
            )
        logger.info(
            f"Initialized Open AI embedding model with {embed_config.embed_batch_size} batch size."
        )
    elif isinstance(embed_config, DashScopeEmbeddingConfig):
        if embed_config.model is not None:
            embed_model = DashScopeEmbedding(
                api_key=embed_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
                embed_batch_size=embed_config.embed_batch_size,
                callback_manager=Settings.callback_manager,
                model_name=embed_config.model,
            )
        else:
            embed_model = DashScopeEmbedding(
                api_key=embed_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
                embed_batch_size=embed_config.embed_batch_size,
                callback_manager=Settings.callback_manager,
            )
        logger.info(
            f"Initialized DashScope embedding model with {embed_config.embed_batch_size} batch size."
        )
    elif isinstance(embed_config, HuggingFaceEmbeddingConfig):
        model_dir = model_dir or os.getenv("PAIRAG_MODEL_DIR", "./model_repository")
        pai_model_path = os.path.join(model_dir, embed_config.model)
        download_model_to_directory(embed_config.model, model_dir)

        embed_model = HuggingFaceEmbedding(
            model_name=pai_model_path,
            embed_batch_size=embed_config.embed_batch_size,
            trust_remote_code=True,
            callback_manager=Settings.callback_manager,
            device=infer_cuda_device(),
        )

        logger.info(
            f"Initialized HuggingFace embedding model {embed_config.model} from model_dir_path {model_dir} with {embed_config.embed_batch_size} batch size."
        )
    else:
        raise ValueError(f"Unknown Embedding source: {embed_config}")

    return embed_model
