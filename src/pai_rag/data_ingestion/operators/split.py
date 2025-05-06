from typing import Any, Dict, List
from llama_index.core.schema import TextNode
from pai_rag.data_ingestion.models.config.operator import SplitterConfig
from pai_rag.data_ingestion.models.file.event import NodeOperationType
from pai_rag.data_ingestion.operators.base import BaseOperator
from pai_rag.data_ingestion.utils.node_utils import (
    metadata_dict_to_node_v2,
    node_to_metadata_dict_v2,
)
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import (
    NodeParserConfig,
    PaiNodeParser,
)
from loguru import logger


class Splitter(BaseOperator):
    def __init__(self, config: SplitterConfig):
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
            # 去掉文本内容为空的分块
            if isinstance(node, TextNode) and not node.text:
                continue
            node_dict = node_to_metadata_dict_v2(node)
            node_dict["operation"] = row.get("operation")
            node_dict["operation_reason"] = row.get("operation_reason")
            nodes.append(node_dict)

        logger.info(f"Split {len(splitted_nodes)} nodes from {doc.node_id}.")
        return nodes

    def __call__(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        if row.get("operation") == NodeOperationType.DELETE:
            return self.process_delete(row)
        else:
            return self.process_add(row)
