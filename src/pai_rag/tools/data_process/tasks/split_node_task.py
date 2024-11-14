from pai_rag.core.rag_module import resolve
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from llama_index.core.schema import Document
from datetime import datetime
from pai_rag.core.rag_config_manager import RagConfigManager


def text_node_to_dict(node):
    return {
        "id": node.id_,
        "embedding": node.embedding,
        "metadata": {
            k: str(v) if isinstance(v, datetime) else v
            for k, v in node.metadata.items()
        },
        "text": node.text,
        "mimetype": node.mimetype,
        "start_char_idx": node.start_char_idx,
        "end_char_idx": node.end_char_idx,
        "text_template": node.text_template,
        "metadata_template": node.metadata_template,
        "metadata_seperator": node.metadata_seperator,
    }


def split_node_task(document, config_file):
    config = RagConfigManager.from_file(config_file).get_value()
    parser_config = config.node_parser
    node_parser = resolve(cls=PaiNodeParser, parser_config=parser_config)
    format_doc = Document(
        doc_id=document["doc_id"],
        text=document["data"]["content"],
        metadata=document["data"]["meta_data"],
    )
    nodes = node_parser([format_doc])
    return [text_node_to_dict(node) for node in nodes]
