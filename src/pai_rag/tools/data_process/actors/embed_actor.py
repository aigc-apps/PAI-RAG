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
from loguru import logger


class EmbedActor:
    def __init__(self, config_file):
        logger.info("Init EmbedActor.")
        # RAY_ENV_MODEL_DIR = "/PAI-RAG/pai_rag_model_repository"
        RAY_ENV_MODEL_DIR = "/home/xiaowen/xiaowen/github_code/PAI-RAG/model_repository"
        os.environ["PAI_RAG_MODEL_DIR"] = RAY_ENV_MODEL_DIR
        config = RagConfigManager.from_file(config_file).get_value()
        download_models = ModelScopeDownloader(
            fetch_config=True, download_directory_path=RAY_ENV_MODEL_DIR
        )
        self.embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
        logger.info(f"Dense embed model loaded {config.embedding}.")
        self.sparse_embed_model = None
        if config.embedding.enable_sparse:
            download_models.load_model(model="bge-m3")
            self.sparse_embed_model = BGEM3SparseEmbeddingFunction(
                model_name_or_path=RAY_ENV_MODEL_DIR
            )
            logger.info("Sparse embed model loaded.")
        self.multimodal_embed_model = resolve(
            cls=PaiMultiModalEmbedding,
            multimodal_embed_config=config.multimodal_embedding,
        )
        logger.info(f"Multi-modal embed model loaded {config.multimodal_embedding}.")
        logger.info("EmbedActor init finished.")

    def __call__(self, node):
        format_node = dict_to_node(node)
        if type(format_node) is TextNode:
            embed_nodes = self.embed_model([format_node])
            sparse_embedding = None
            if self.sparse_embed_model:
                sparse_embedding = self.sparse_embed_model.encode_documents(
                    [embed_nodes[0].text]
                )[0]
            nodes_dict = node_to_dict(embed_nodes[0], sparse_embedding)
        elif type(format_node) is ImageNode:
            embed_nodes = self.multimodal_embed_model([format_node])
            nodes_dict = node_to_dict(embed_nodes[0], None)
        else:
            raise ValueError(f"Invalid node type: {type(format_node)}")
        return nodes_dict
