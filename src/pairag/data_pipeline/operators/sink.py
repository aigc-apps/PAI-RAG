import traceback
from typing import Dict
import numpy as np
import pandas as pd

from pairag.data_pipeline.operators.base import BaseOperator
from pairag.data_pipeline.models.config.operator import SinkConfig
from pairag.data_pipeline.utils.vectordb_utils import get_vector_store
from pairag.data_pipeline.utils.node_utils import metadata_dict_to_node_v2
from loguru import logger


class Sinker(BaseOperator):
    def __init__(
        self,
        config: SinkConfig,
    ):
        logger.info(f"Sinker init started with {config}.")
        super().__init__(
            name=config.name,
            num_cpus=config.num_cpus,
            num_gpus=config.num_gpus,
            memory=config.memory,
        )

        self.vector_store = get_vector_store(
            rag_endpoint=config.pairag_endpoint,
            rag_api_key=config.pairag_token,
            knowledgebase=config.pairag_knowledgebase,
            embed_dims=config.pairag_embed_dims,
        )

        logger.info(f"Sinker init successfully with {config}.")

    def __call__(self, row_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        logger.info("Start data_sink op.")
        chunks_df = pd.DataFrame(row_batch)
        logger.info(f"Start saving {len(chunks_df)} nodes...")

        node_id_array = chunks_df.id.to_numpy()
        op_type_array = chunks_df.operation.to_numpy()
        file_name_array = chunks_df.file_name.to_numpy()

        add_chunks = chunks_df[chunks_df.operation == "add"].to_dict(orient="records")
        node_ids_to_delete = chunks_df[chunks_df.operation == "delete"].id.tolist()

        logger.info(
            f"Got {len(chunks_df)} in total, {len(add_chunks)} to add and {len(node_ids_to_delete)} to delete."
        )
        try:
            if len(node_ids_to_delete) > 0:
                self.vector_store.delete_nodes(node_ids_to_delete)
                logger.info(f"Deleted {node_ids_to_delete} nodes successfully.")

            if len(add_chunks) > 0:
                nodes_to_add = []
                for chunk in add_chunks:
                    nodes_to_add.append(metadata_dict_to_node_v2(chunk))

                self.vector_store.add(nodes_to_add)
                logger.info(f"Successfully saved {len(nodes_to_add)} nodes.")

            logger.info(f"Finished processing {len(chunks_df)} chunks.")
            logger.info("Finished data_sink op.")
            return {
                "id": node_id_array,
                "operation": op_type_array,
                "file_name": file_name_array,
            }
        except Exception:
            logger.error(f"Error saving nodes: {traceback.format_exc()}")
            raise
