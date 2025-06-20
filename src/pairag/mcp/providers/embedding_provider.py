from typing import Dict
from sqlmodel import select
from pairag.db.encrypt_utils import decrypt_key
from pairag.db.models.knowledgebase.embedding import EmbeddingModelEntity, EmbeddingType
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pairag.utils.cuda_utils import infer_cuda_device
from loguru import logger

from pairag.utils.modelscope_utils import download_model_to_directory


def create_embedding_model(embedding_config: EmbeddingModelEntity) -> BaseEmbedding:
    if embedding_config.type == EmbeddingType.OPENAI_LIKE:
        logger.info(
            f"Creating OpenAI like embedding model  {embedding_config.model_name}"
        )
        return OpenAILikeEmbedding(
            api_key=decrypt_key(embedding_config.encrypted_api_key),
            model_name=embedding_config.model_name,
            dimensions=embedding_config.dimension,
            embed_batch_size=embedding_config.embed_batch_size,
            api_base=embedding_config.endpoint,
        )
    elif embedding_config.type == EmbeddingType.LOCAL:
        pai_model_path = download_model_to_directory(embedding_config.model_name)
        logger.info(
            f"Creating local embedding model {embedding_config.model_name} with path {pai_model_path}."
        )

        return HuggingFaceEmbedding(
            model_name=pai_model_path,
            embed_batch_size=embedding_config.embed_batch_size,
            device=infer_cuda_device(),
        )
    else:
        logger.error(f"Unknown embedding type: {embedding_config.type}.")
        raise ValueError(f"Unknown embedding type: {embedding_config.type}.")


@with_async_db_session
async def fetch_embedding_models(session: AsyncSession):
    logger.info("[EmbeddingProvider] Start fetching embedding models.")
    sql_results = await session.exec(select(EmbeddingModelEntity))
    embedding_results = sql_results.all()

    embedding_config_map = {
        embedding.model_name: embedding for embedding in embedding_results
    }
    logger.info(
        f"[EmbeddingProvider] fetched {len(embedding_results)} embedding models."
    )
    embedding_models_map = {
        embedding_config.model_name: create_embedding_model(embedding_config)
        for embedding_config in embedding_results
    }
    return embedding_config_map, embedding_models_map


class EmbeddingProvider:
    def __init__(self):
        self.embedding_config_map: Dict[str, EmbeddingModelEntity] = {}
        self.embedding_models_map: Dict[str, BaseEmbedding] = {}

    async def refresh(self):
        logger.info("[EmbeddingProvider] Start refreshing embedding models.")
        (
            self.embedding_config_map,
            self.embedding_models_map,
        ) = await fetch_embedding_models()
        logger.info(
            f"[EmbeddingProvider]refreshed {len(self.embedding_models_map)} llm models."
        )

    def get_embedding_config(self, model_name: str) -> EmbeddingModelEntity:
        assert (
            model_name in self.embedding_config_map
        ), f"Embedding model '{model_name}' not found"
        return self.embedding_config_map[model_name]

    def get_embedding_model(self, model_name: str) -> OpenAILikeEmbedding:
        assert (
            model_name in self.embedding_models_map
        ), f"Embedding model '{model_name}' not found"
        return self.embedding_models_map[model_name]


embedding_provider = EmbeddingProvider()
