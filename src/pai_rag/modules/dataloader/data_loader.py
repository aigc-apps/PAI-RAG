from typing import Any, Dict, List
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.core.rag_data_loader import RagDataLoader
import logging

logger = logging.getLogger(__name__)


class DataLoaderModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return [
            "DataReaderModule",
            "NodeParserModule",
            "IndexModule",
            "NodesEnhancementModule",
            "EmbeddingModule",
            "MultiModalEmbeddingModule",
        ]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        data_reader = new_params["DataReaderModule"]
        node_parser = new_params["NodeParserModule"]
        index = new_params["IndexModule"]
        node_enhance = new_params["NodesEnhancementModule"]
        embed_model = new_params["EmbeddingModule"]
        multi_modal_embed_modal = new_params["MultiModalEmbeddingModule"]

        return RagDataLoader(
            data_reader=data_reader,
            node_parser=node_parser,
            vector_index=index,
            raptor_processor=node_enhance,
            embed_model=embed_model,
            multi_modal_embed_modal=multi_modal_embed_modal,
        )
