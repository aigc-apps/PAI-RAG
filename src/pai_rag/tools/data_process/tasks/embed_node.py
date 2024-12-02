import os
from pai_rag.core.rag_module import resolve
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.tools.data_process.utils.format_node import (
    node_to_dict,
    dict_to_node,
)
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
from pai_rag.integrations.embeddings.pai.pai_multimodal_embedding import (
    PaiMultiModalEmbedding,
)
from llama_index.core.schema import TextNode, ImageNode

RAY_ENV_MODEL_DIR = "/PAI-RAG/pai_rag_model_repository"
os.environ["PAI_RAG_MODEL_DIR"] = RAY_ENV_MODEL_DIR


def embed_node_task(node, config_file):
    config = RagConfigManager.from_file(config_file).get_value()
    download_models = ModelScopeDownloader(
        fetch_config=True, download_directory_path=RAY_ENV_MODEL_DIR
    )
    format_node = dict_to_node(node)
    if type(format_node) is TextNode:
        embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
        embed_nodes = embed_model([format_node])
        sparse_embedding = None
        if config.embedding.enable_sparse:
            download_models.load_model(model="bge-m3")
            sparse_embed_model = BGEM3SparseEmbeddingFunction(
                model_name_or_path=RAY_ENV_MODEL_DIR
            )
            sparse_embedding = sparse_embed_model.encode_documents(
                [embed_nodes[0].text]
            )[0]
        nodes_dict = node_to_dict(embed_nodes[0], sparse_embedding)
    elif type(format_node) is ImageNode:
        multimodal_embed_model = resolve(
            cls=PaiMultiModalEmbedding,
            multimodal_embed_config=config.multimodal_embedding,
        )
        embed_nodes = multimodal_embed_model([format_node])
        nodes_dict = node_to_dict(embed_nodes[0], None)
    else:
        raise ValueError(f"Invalid node type: {type(format_node)}")
    return nodes_dict
