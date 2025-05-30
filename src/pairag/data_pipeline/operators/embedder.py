import traceback
from typing import Dict, List

import numpy as np
from pairag.data_pipeline.models.config.operator import EmbedderConfig
from pairag.data_pipeline.operators.base import BaseOperator
from pairag.data_pipeline.utils.download_utils import download_models_via_lock
from pairag.integrations.embeddings.pai.embedding_utils import create_embedding
from pairag.integrations.embeddings.pai.pai_embedding_config import parse_embed_config
from pairag.knowledgebase.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
from loguru import logger


class Embedder(BaseOperator):
    def __init__(
        self,
        config: EmbedderConfig,
    ):
        super().__init__(
            name=config.name,
            num_cpus=config.num_cpus,
            memory=config.memory,
            num_gpus=config.num_gpus,
            model_dir=config.model_dir,
        )
        self.embedder_cfg = parse_embed_config(
            {
                "source": config.source,
                "model": config.model,
                "enable_sparse": config.enable_sparse,
                "embed_batch_size": config.batch_size,
            }
        )
        # Init model download list
        self.download_model_list = []
        if self.embedder_cfg.source.lower() == "huggingface":
            self.download_model_list.append(self.embedder_cfg.model)
        if self.embedder_cfg.enable_sparse:
            self.download_model_list.append("bge-m3")
        for model_name in self.download_model_list:
            download_models_via_lock(self.model_dir, model_name)

        # Init embedding models
        self.embed_model = create_embedding(self.embedder_cfg)
        if self.embedder_cfg.enable_sparse:
            self.sparse_embed_model = BGEM3SparseEmbeddingFunction(
                model_name_or_path=self.model_dir
            )

        logger.info(
            f"""Embedder [PaiEmbedding] init finished with following parameters: {config}"""
        )

    def calc_embeddings(self, texts: List[str]):
        embeddings = self.embed_model.get_text_embedding_batch(texts)
        return embeddings

    def calc_sparse_embeddings(self, texts: List[str]):
        if self.embedder_cfg.enable_sparse:
            sparse_embeddings = self.sparse_embed_model.encode_documents(texts)
        else:
            sparse_embeddings = [None] * len(texts)
        return sparse_embeddings

    def __call__(self, nodes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        try:
            node_texts = nodes.get("text", [])
            logger.info(f"Start embedding {len(node_texts)} nodes...")

            if len(node_texts) == 0:
                logger.warning("No nodes to embed, directly returning...")
                return nodes

            nodes["embedding"] = self.calc_embeddings(node_texts)
            nodes["sparse_embedding"] = self.calc_sparse_embeddings(node_texts)

            logger.info(
                f"Successfully calculated embeddings for {len(node_texts)} nodes."
            )
            return nodes
        except Exception:
            logger.error(
                f"Error calculating embeddings for nodes: {traceback.format_exc()}"
            )
            raise
