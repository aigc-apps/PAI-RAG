from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
    DashScopeEmbeddingConfig,
    OpenAIEmbeddingConfig,
    HuggingFaceEmbeddingConfig,
    CnClipEmbeddingConfig,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pai_rag.integrations.embeddings.clip.cnclip_embedding import CnClipEmbedding
import os
import logging

logger = logging.getLogger(__name__)


def create_embedding(embed_config: PaiBaseEmbeddingConfig):
    if isinstance(embed_config, OpenAIEmbeddingConfig):
        embed_model = OpenAIEmbedding(
            api_key=embed_config.api_key,
            embed_batch_size=embed_config.embed_batch_size,
        )
        logger.info(
            f"Initialized Open AI embedding model with {embed_config.embed_batch_size} batch size."
        )
    elif isinstance(embed_config, DashScopeEmbeddingConfig):
        embed_model = DashScopeEmbedding(
            api_key=embed_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
            embed_batch_size=embed_config.embed_batch_size,
        )
        logger.info(
            f"Initialized DashScope embedding model with {embed_config.embed_batch_size} batch size."
        )
    elif isinstance(embed_config, HuggingFaceEmbeddingConfig):
        pai_model_dir = os.getenv("PAI_RAG_MODEL_DIR", "./model_repository")
        embed_model = HuggingFaceEmbedding(
            model_name=os.path.join(pai_model_dir, embed_config.model),
            embed_batch_size=embed_config.embed_batch_size,
            trust_remote_code=True,
        )

        logger.info(
            f"Initialized HuggingFace embedding model {embed_config.model} from model_dir_path {pai_model_dir} with {embed_config.embed_batch_size} batch size."
        )

    elif isinstance(embed_config, CnClipEmbeddingConfig):
        embed_model = CnClipEmbedding(
            model_name=embed_config.model,
            embed_batch_size=embed_config.embed_batch_size,
        )
        logger.info(
            f"Initialized CnClip embedding model {embed_config.model} with {embed_config.embed_batch_size} batch size."
        )

    else:
        raise ValueError(f"Unknown Embedding source: {embed_config}")

    return embed_model
