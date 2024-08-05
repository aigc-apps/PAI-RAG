import logging
from typing import Dict, List, Any

from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.index.pai_bm25_index import PaiBm25Index


logger = logging.getLogger(__name__)


class BM25IndexModule(ConfigurableModule):
    """Class for managing indices.

    RagIndex to manage vector indices for RagApplication.
    When initializing, the index is empty or load from existing index.
    User can add nodes to index when needed.
    """

    @staticmethod
    def get_dependencies() -> List[str]:
        return ["IndexModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        index = new_params["IndexModule"]
        if (
            index.vectordb_type == "elasticsearch"
            or index.vectordb_type == "milvus"
            or index.vectordb_type == "postgresql"
        ):
            logger.info(f"Do not use local BM25 Index for {index.vectordb_type}.")
            return None
        else:
            logger.info("Using BM25 Index.")
            return PaiBm25Index(index.persist_path)
