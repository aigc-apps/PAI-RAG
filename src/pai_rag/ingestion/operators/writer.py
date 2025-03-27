from typing import List, Optional

import requests
from pai_rag.ingestion.operators.base import BaseOperator, OperatorName
from pai_rag.ingestion.utils.formatters import convert_dict_to_node
from pai_rag.integrations.index.pai.utils.vector_store_utils import create_vector_store
from pai_rag.integrations.index.pai.vector_store_config import (
    BaseVectorStoreConfig,
    SupportedVectorStoreType,
)
from pai_rag.knowledgebase.models import KnowledgeBase
import asyncio

import ray
from loguru import logger


def get_vector_store_config(
    rag_endpoint: str, rag_key: str, knowledgebase: str
) -> BaseVectorStoreConfig:
    try:
        response = requests.get(
            f"{rag_endpoint}/api/v1/knowledgebases/{knowledgebase}",
            headers={"Authorization": f"Bearer {rag_key}"},
        )
        knowledgebase: KnowledgeBase = KnowledgeBase.model_validate(response.json())
        return knowledgebase.vector_store_config
    except Exception as e:
        logger.error(f"Failed to get vector store config: {e}")
        raise e


@ray.remote
class Writer(BaseOperator):
    def __init__(
        self,
        name: str = OperatorName.WRITER,
        batch_size: int = 10,
        device: str = "cpu",
        num_cpus: float = 1,
        num_gpus: Optional[float] = None,
        model_dir: str = None,
        output_filename: str = None,
        rag_endpoint: str = None,
        rag_key: str = None,
        knowledgebase: str = "default",
        embed_dims: int = 1024,
        **kwargs,
    ):
        super().__init__(
            name=name,
            batch_size=batch_size,
            device=device,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            model_dir=model_dir,
            output_filename=output_filename,
            **kwargs,
        )
        vector_store_config = get_vector_store_config(
            rag_endpoint=rag_endpoint, rag_key=rag_key, knowledgebase=knowledgebase
        )
        assert (
            vector_store_config.type != SupportedVectorStoreType.faiss
        ), "FAISS is not supported."

        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

        self.vector_store = create_vector_store(
            vectordb_config=vector_store_config,
            embed_dims=embed_dims,
        )

        logger.info(
            f"""Sinker [PaiVectorStore] init finished with following parameters:
                        config: {vector_store_config}
                        embed_dims: {embed_dims}
            """
        )

    def process(self, chunks: List[dict]) -> List[dict]:
        nodes = [convert_dict_to_node(chunk) for chunk in chunks]
        self.vector_store.add(nodes)
        logger.info(f"Inserted {len(nodes)} nodes into vector store.")
        return True
