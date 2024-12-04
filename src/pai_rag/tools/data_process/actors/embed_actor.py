import os
from pai_rag.core.rag_module import resolve
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
from loguru import logger
import numpy as np


class EmbedActor:
    def __init__(self, working_dir, config_file):
        RAY_ENV_MODEL_DIR = os.path.join(working_dir, "model_repository")
        os.environ["PAI_RAG_MODEL_DIR"] = RAY_ENV_MODEL_DIR
        logger.info(f"Init EmbedActor with working dir: {RAY_ENV_MODEL_DIR}.")
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

        logger.info("EmbedActor init finished.")

    def __call__(self, nodes):
        text_contents = list(nodes["text"])
        embeddings = self.embed_model.get_text_embedding_batch(text_contents)
        nodes["embedding"] = np.array(embeddings)
        if self.sparse_embed_model:
            sparse_embeddings = self.sparse_embed_model.encode_documents(text_contents)
            nodes["sparse_embedding"] = sparse_embeddings
        return nodes
