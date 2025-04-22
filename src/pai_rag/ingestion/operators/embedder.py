import os
from typing import Any, Dict, List
from pai_rag.ingestion.models.config.base import EmbedderConfig
from pai_rag.ingestion.models.file.event import NodeOperationType
from pai_rag.ingestion.operators.base import BaseOperator
from pai_rag.ingestion.utils.download_utils import download_models_via_lock
from pai_rag.ingestion.utils.node_utils import metadata_dict_to_node_v2
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pai_rag.integrations.embeddings.pai.embedding_utils import create_embedding
from pai_rag.integrations.embeddings.pai.pai_embedding_config import parse_embed_config
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
import ray
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

    def calc_embedings(self, texts: List[str]) -> Dict[str, List[float]]:
        embeddings = self.embed_model.get_text_embedding_batch(texts)
        return dict(zip(texts, embeddings))

    def calc_sparse_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
        if self.embedder_cfg.enable_sparse:
            sparse_embeddings = self.sparse_embed_model.encode_documents(texts)
        else:
             sparse_embeddings = [None] * len(texts)
        return dict(zip(texts, sparse_embeddings))
        

    def __call__(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        node_texts = [node["text"] for node in nodes 
            if node.get("operation") != NodeOperationType.DELETE]
        
        embedding_dict = self.calc_embedings(node_texts)
        sparse_embedding_dict = self.calc_sparse_embeddings(node_texts)

        for node in nodes:
            if node.get("operation") != NodeOperationType.DELETE:
                node["embedding"] = embedding_dict.get(node["text"])
                node["sparse_embedding"] = sparse_embedding_dict.get(node["text"])

        return nodes