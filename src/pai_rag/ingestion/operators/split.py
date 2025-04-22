from typing import Any, Dict, List, Optional
from pai_rag.ingestion.models.config.base import SplitterConfig
from pai_rag.ingestion.models.file.event import NodeOperationType
from pai_rag.ingestion.operators.base import BaseOperator, OperatorName
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pai_rag.ingestion.utils.node_utils import metadata_dict_to_node_v2
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import (
    NodeParserConfig,
    PaiNodeParser,
)
import ray
from loguru import logger


class Splitter(BaseOperator):
    def __init__(
        self,
        config: SplitterConfig
    ):
        super().__init__(
            name=config.name,
            num_cpus=config.num_cpus,
            num_gpus=config.num_gpus,
            model_dir=config.model_dir,
        )

        self.node_parser_config = NodeParserConfig(
            type=config.node_parser_type,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.node_parser = PaiNodeParser(
            parser_config=self.node_parser_config,
        )
        logger.info(
            f"""SplitterActor init finished with following parameters: {config}"""
        )

    def process_delete(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [row]

    def process_add(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = metadata_dict_to_node_v2(row)
        nodes = []

        splitted_nodes = self.node_parser.get_nodes_from_documents([doc])
        for node in splitted_nodes:
            node_dict = node_to_metadata_dict(doc)
            node_dict["operation"] = row.get("operation")
            node_dict["operation_reason"] = row.get("operation_reason")
            nodes.append(node_dict)

        return nodes

    def __call__(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        if row.get("operation") == NodeOperationType.DELETE:
            return self.process_delete(row)
        else:
            return self.process_add(row)