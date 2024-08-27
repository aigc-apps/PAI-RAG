from typing import Any, Dict, List
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.data.rag_dataloader import RagDataLoader
from pai_rag.data.rag_oss_dataloader import OssDataLoader
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
            "NodesEnhancementModule",
        ]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.loader_config = new_params[MODULE_PARAM_CONFIG]
        oss_cache = new_params["OssCacheModule"]
        data_reader_factory = new_params["DataReaderFactoryModule"]
        node_parser = new_params["NodeParserModule"]
        index = new_params["IndexModule"]
        bm25_index = new_params["BM25IndexModule"]
        node_enhance = new_params["NodesEnhancementModule"]

        if self.loader_config["type"].lower() == "local":
            return RagDataLoader(
                data_reader_factory,
                node_parser,
                index,
                bm25_index,
                oss_cache,
                node_enhance,
            )
        elif self.loader_config["type"].lower() == "oss":
            return OssDataLoader(
                data_reader_factory,
                node_parser,
                index,
                bm25_index,
                oss_cache,
                node_enhance,
            )
