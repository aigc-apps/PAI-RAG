from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
    DashScopeEmbeddingConfig,
    OpenAIEmbeddingConfig,
    HuggingFaceEmbeddingConfig,
    CnClipEmbeddingConfig,
)

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pai_rag.integrations.embeddings.clip.cnclip_embedding import CnClipEmbedding
import os
from loguru import logger
from pai_rag.utils.download_models import ModelScopeDownloader


def create_embedding(
    embed_config: PaiBaseEmbeddingConfig, pai_rag_model_dir: str = None
):
    if isinstance(embed_config, OpenAIEmbeddingConfig):
        embed_model = OpenAIEmbedding(
            api_key=embed_config.api_key,
            embed_batch_size=embed_config.embed_batch_size,
            callback_manager=Settings.callback_manager,
        )
        logger.info(
            f"Initialized Open AI embedding model with {embed_config.embed_batch_size} batch size."
        )
    elif isinstance(embed_config, DashScopeEmbeddingConfig):
        embed_model = DashScopeEmbedding(
            api_key=embed_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
            embed_batch_size=embed_config.embed_batch_size,
            callback_manager=Settings.callback_manager,
        )
        logger.info(
            f"Initialized DashScope embedding model with {embed_config.embed_batch_size} batch size."
        )
    elif isinstance(embed_config, HuggingFaceEmbeddingConfig):
        pai_rag_model_dir = pai_rag_model_dir or os.getenv(
            "PAI_RAG_MODEL_DIR", "./model_repository"
        )
        pai_model_path = os.path.join(pai_rag_model_dir, embed_config.model)
        if not os.path.exists(pai_model_path):
            logger.info(
                f"Embedding model {embed_config.model} not found in {pai_rag_model_dir}, try download it."
            )
            download_models = ModelScopeDownloader(
                fetch_config=True, download_directory_path=pai_rag_model_dir
            )
            download_models.load_model(model=embed_config.model)
            logger.info(
                f"Embedding model {embed_config.model} downloaded to {pai_model_path}."
            )
        embed_model = HuggingFaceEmbedding(
            model_name=pai_model_path,
            embed_batch_size=embed_config.embed_batch_size,
            trust_remote_code=True,
            callback_manager=Settings.callback_manager,
        )

        logger.info(
            f"Initialized HuggingFace embedding model {embed_config.model} from model_dir_path {pai_rag_model_dir} with {embed_config.embed_batch_size} batch size."
        )

    elif isinstance(embed_config, CnClipEmbeddingConfig):
        pai_rag_model_dir = pai_rag_model_dir or os.getenv(
            "PAI_RAG_MODEL_DIR", "./model_repository"
        )
        pai_model_path = os.path.join(
            pai_rag_model_dir, "chinese-clip-vit-large-patch14"
        )
        print("create_embedding pai_model_path:", pai_model_path)
        if not os.path.exists(pai_model_path):
            logger.info(
                f"Embedding model {embed_config.model} not found in {pai_rag_model_dir}, try download it."
            )
            download_models = ModelScopeDownloader(
                fetch_config=True, download_directory_path=pai_rag_model_dir
            )
            download_models.load_model(model="chinese-clip-vit-large-patch14")
            logger.info(
                f"Embedding model {embed_config.model} downloaded to {pai_model_path}."
            )
        embed_model = CnClipEmbedding(
            model_name=embed_config.model,
            embed_batch_size=embed_config.embed_batch_size,
            callback_manager=Settings.callback_manager,
            model_path=pai_model_path,
        )
        logger.info(
            f"Initialized CnClip embedding model {embed_config.model} with {embed_config.embed_batch_size} batch size."
        )

    else:
        raise ValueError(f"Unknown Embedding source: {embed_config}")

    return embed_model
