import os
from pai_rag.core.rag_module import resolve
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.tools.data_process.utils.format_node import (
    text_node_to_dict,
    dict_to_text_node,
)
from pai_rag.utils.constants import DEFAULT_MODEL_DIR
from pai_rag.utils.download_models import ModelScopeDownloader


def embed_node_task(node, config_file):
    config = RagConfigManager.from_file(config_file).get_value()
    ModelScopeDownloader().load_rag_models()
    os.environ["PAI_RAG_MODEL_DIR"] = DEFAULT_MODEL_DIR

    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
    format_node = dict_to_text_node(node)
    embed_nodes = embed_model([format_node])
    nodes_dict = text_node_to_dict(embed_nodes[0])
    return nodes_dict
