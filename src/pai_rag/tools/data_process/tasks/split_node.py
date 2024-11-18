from pai_rag.core.rag_module import resolve
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.tools.data_process.utils.format_document import dict_to_document
from pai_rag.tools.data_process.utils.format_node import text_node_to_dict


def split_node_task(document, config_file):
    config = RagConfigManager.from_file(config_file).get_value()
    parser_config = config.node_parser
    node_parser = resolve(cls=PaiNodeParser, parser_config=parser_config)
    format_document = dict_to_document(document)
    nodes = node_parser([format_document])
    return [text_node_to_dict(node) for node in nodes]
