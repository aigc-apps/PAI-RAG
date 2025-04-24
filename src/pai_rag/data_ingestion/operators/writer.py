from typing import List, Optional

from pai_rag.data_ingestion.operators.base import BaseOperator
from pai_rag.data_ingestion.models.config.operator import OperatorName

import ray
from loguru import logger

from pai_rag.data_ingestion.utils.vectordb_utils import get_vector_store


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

        self.vector_store = get_vector_store(
            rag_endpoint=rag_endpoint,
            rag_key=rag_key,
            knowledgebase=knowledgebase,
            embed_dims=embed_dims,
        )

        logger.info("Writer init successfully.")

    def process(self, chunks: List[dict]) -> List[dict]:
        nodes = [convert_dict_to_node(chunk) for chunk in chunks]
        self.vector_store.add(nodes)
        logger.info(f"Inserted {len(nodes)} nodes into vector store.")
        return True
