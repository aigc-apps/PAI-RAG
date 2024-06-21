from typing import Any, Dict, List
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.data.rag_dataloader import RagDataLoader
import logging

logger = logging.getLogger(__name__)


class DataLoaderModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return [
            "OssCacheModule",
            "DataReaderFactoryModule",
            "NodeParserModule",
            "IndexModule",
            "BM25IndexModule",
        ]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        oss_cache = new_params["OssCacheModule"]
        data_reader_factory = new_params["DataReaderFactoryModule"]
        node_parser = new_params["NodeParserModule"]
        index = new_params["IndexModule"]
        bm25_index = new_params["BM25IndexModule"]

        return RagDataLoader(
            data_reader_factory, node_parser, index, bm25_index, oss_cache
        )
