import os
from pai_rag.core.rag_module import resolve
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.tools.data_process.utils.format_node import (
    text_node_to_dict,
    dict_to_text_node,
)
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)

RAY_ENV_MODEL_DIR = "/PAI-RAG/pai_rag_model_repository"
os.environ["PAI_RAG_MODEL_DIR"] = RAY_ENV_MODEL_DIR


def embed_node_task(node, config_file):
    config = RagConfigManager.from_file(config_file).get_value()
    download_models = ModelScopeDownloader(download_directory_path=RAY_ENV_MODEL_DIR)
    download_models.load_models()

    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
    format_node = dict_to_text_node(node)
    embed_nodes = embed_model([format_node])
    sparse_embedding = None
    if config.embedding.enable_sparse:
        sparse_embed_model = BGEM3SparseEmbeddingFunction()
        sparse_embedding = sparse_embed_model.encode_documents([embed_nodes[0].text])[0]
    nodes_dict = text_node_to_dict(embed_nodes[0], sparse_embedding)
    return nodes_dict
