import asyncio
import requests
from pai_rag.knowledgebase.index.pai.utils.vector_store_utils import create_vector_store
from pai_rag.knowledgebase.index.pai.vector_store_config import (
    BaseVectorStoreConfig,
    SupportedVectorStoreType,
)
from pai_rag.knowledgebase.models import KnowledgeBase
from loguru import logger


def get_vector_store_config(
    rag_endpoint: str, rag_api_key: str, knowledgebase: str
) -> BaseVectorStoreConfig:
    try:
        response = requests.get(
            f"{rag_endpoint}/api/v1/knowledgebases/{knowledgebase}",
            headers={"Authorization": f"Bearer {rag_api_key}"},
        )
        knowledgebase: KnowledgeBase = KnowledgeBase.model_validate(response.json())
        return knowledgebase.vector_store_config
    except Exception as e:
        logger.error(f"Failed to get vector store config: {e}")
        raise e


def get_vector_store(
    rag_endpoint: str,
    rag_api_key: str,
    knowledgebase: str,
    embed_dims: int,
):
    vector_store_config = get_vector_store_config(
        rag_endpoint=rag_endpoint, rag_api_key=rag_api_key, knowledgebase=knowledgebase
    )

    logger.info(f"Creating vector store from config {vector_store_config}.")
    assert (
        vector_store_config.type != SupportedVectorStoreType.faiss
    ), "FAISS is not supported."

    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    vector_store = create_vector_store(
        vectordb_config=vector_store_config,
        embed_dims=embed_dims,
    )

    logger.info(
        f"""[PaiVectorStore] init finished with following parameters:
                    config: {vector_store_config}
                    embed_dims: {embed_dims}
        """
    )

    return vector_store
