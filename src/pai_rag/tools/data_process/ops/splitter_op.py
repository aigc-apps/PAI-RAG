import ray
from pai_rag.core.rag_module import resolve
from pai_rag.tools.data_process.ops.base_op import BaseOP, OPERATORS
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import NodeParserConfig
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.tools.data_process.utils.formatters import convert_list_to_documents
from pai_rag.tools.data_process.utils.formatters import convert_nodes_to_list

OP_NAME = "rag_splitter"


@OPERATORS.register_module(OP_NAME)
@ray.remote
class Splitter(BaseOP):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    def __init__(
        self,
        type: str = "Token",
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        enable_multimodal: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.node_parser_config = NodeParserConfig(
            type=type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_multimodal=enable_multimodal,
        )
        self.node_parser = resolve(
            cls=PaiNodeParser, parser_config=self.node_parser_config
        )
        self.logger.info(
            f"""SplitterActor [PaiNodeParser] init finished with following parameters:
                        type: {type}
                        chunk_size: {chunk_size}
                        chunk_overlap: {chunk_overlap}
                        enable_multimodal: {enable_multimodal}
            """
        )

    def process(self, documents):
        format_documents = convert_list_to_documents(documents)
        nodes = self.node_parser.get_nodes_from_documents(format_documents)
        return convert_nodes_to_list(nodes)
