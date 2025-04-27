import traceback
from typing import Dict
import numpy as np
import pandas as pd

from pai_rag.data_ingestion.operators.base import BaseOperator
from pai_rag.data_ingestion.models.config.operator import WriterConfig
from pai_rag.data_ingestion.utils.vectordb_utils import get_vector_store
from pai_rag.data_ingestion.utils.node_utils import metadata_dict_to_node_v2
from loguru import logger


class Writer(BaseOperator):
    def __init__(
        self,
        config: WriterConfig,
    ):
        logger.info(f"Writer init started with {config}.")
        super().__init__(
            name=config.name,
            num_cpus=config.num_cpus,
            num_gpus=config.num_gpus,
            memory=config.memory,
        )

        self.vector_store = get_vector_store(
            rag_endpoint=config.rag_endpoint,
            rag_key=config.rag_key,
            knowledgebase=config.knowledgebase,
            embed_dims=config.embed_dims,
        )

        logger.info(f"Writer init successfully with {config}.")

    def __call__(self, row_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        logger.info("Start writing op.")
        chunks_df = pd.DataFrame(row_batch)
        logger.info(f"Start saving {len(chunks_df)} nodes...")

        add_chunks = chunks_df[chunks_df.operation == "add"].to_dict(orient="records")
        node_ids_to_delete = chunks_df[chunks_df.operation == "delete"].id.tolist()

        logger.info(f"Got {len(chunks_df)} in total, {len(add_chunks)} to add and {len(node_ids_to_delete)} to delete.")
        try:
            if len(node_ids_to_delete) > 0:
                self.vector_store.delete_nodes(node_ids_to_delete)
                logger.info(f"Deleted {len(node_ids_to_delete)} nodes successfully.")

            if len(add_chunks) > 0:
                add_nodes = [metadata_dict_to_node_v2(chunk) for chunk in add_chunks]
                self.vector_store.add(add_nodes)
                logger.info(f"Successfully saved {len(add_nodes)} nodes.")

            logger.info(f"Finished processing {len(chunks_df)} chunks.")
            logger.info("Finished writing op.")
            return {"write_sucess": np.array([True])}
        except Exception as e:
            logger.error(f"Error saving nodes: {traceback.format_exc()}")
            raise