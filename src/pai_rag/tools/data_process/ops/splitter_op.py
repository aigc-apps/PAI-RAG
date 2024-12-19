import ray
from loguru import logger
from pai_rag.core.rag_module import resolve
from pai_rag.tools.data_process.ops.base_op import BaseOP
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import NodeParserConfig
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.tools.data_process.utils.formatters import convert_list_to_documents
from pai_rag.tools.data_process.utils.formatters import convert_nodes_to_list


@ray.remote
class Splitter(BaseOP):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    _accelerator = "cpu"
    _batched_op = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_name = "pai_rag_splitter"
        self.node_parser_config = NodeParserConfig(
            type=kwargs.get("type", None),
            chunk_size=kwargs.get("chunk_size", None),
            chunk_overlap=kwargs.get("chunk_overlap", None),
            enable_multimodal=kwargs.get("enable_multimodal", None),
        )
        logger.info("Splitter init finished.")

    def process(self, documents):
        node_parser = resolve(cls=PaiNodeParser, parser_config=self.node_parser_config)
        format_documents = convert_list_to_documents(documents)
        nodes = node_parser.get_nodes_from_documents(format_documents)
        return convert_nodes_to_list(nodes)
