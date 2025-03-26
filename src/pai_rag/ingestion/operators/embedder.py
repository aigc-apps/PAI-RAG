from typing import List, Optional
from pai_rag.ingestion.operators.base import BaseOperator, OperatorName
from pai_rag.ingestion.utils.download_utils import download_models_via_lock
from pai_rag.integrations.embeddings.pai.embedding_utils import create_embedding
from pai_rag.integrations.embeddings.pai.pai_embedding_config import parse_embed_config
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
import ray
import numpy as np
from loguru import logger


@ray.remote
class Embedder(BaseOperator):
    def __init__(
        self,
        name: str = OperatorName.SPLITTER,
        batch_size: int = 10,
        device: str = "cpu",
        num_cpus: float = 1,
        num_gpus: Optional[float] = None,
        model_dir: str = None,
        output_filename: str = None,
        source: str = None,
        model: str = None,
        enable_sparse: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            batch_size=batch_size,
            device=device,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            model_dir=model_dir,
            output_filename=output_filename,
            **kwargs,
        )
        self.embedder_cfg = parse_embed_config(
            {
                "source": source,
                "model": model,
                "enable_sparse": enable_sparse,
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
            f"""Embedder [PaiEmbedding] init finished with following parameters:
                        source: {source}
                        model: {model}
                        enable_sparse: {enable_sparse}
            """
        )

    def process_extra_metadata(self, nodes):
        excluded_embed_metadata_keys = nodes["excluded_embed_metadata_keys"]
        nodes["excluded_embed_metadata_keys"] = np.array(
            [list(a) for a in excluded_embed_metadata_keys]
        )
        excluded_llm_metadata_keys = nodes["excluded_llm_metadata_keys"]
        nodes["excluded_llm_metadata_keys"] = np.array(
            [list(a) for a in excluded_llm_metadata_keys]
        )
        nodes["start_char_idx"] = np.nan_to_num(nodes["start_char_idx"]).astype(int)
        nodes["end_char_idx"] = np.nan_to_num(nodes["start_char_idx"]).astype(int)
        return nodes

    def process(self, chunks: List[dict]) -> List[dict]:
        chunks = [node for node in chunks if node["type"] == "text"]

        if len(chunks) > 0:
            text_contents = [node["text"] for node in chunks]
            embeddings = self.embed_model.get_text_embedding_batch(text_contents)
            if self.embedder_cfg.enable_sparse:
                sparse_embeddings = self.sparse_embed_model.encode_documents(
                    text_contents
                )
            else:
                sparse_embeddings = [None] * len(text_contents)
            # 回填embedding字段
            for node, embedding, sparse_embedding in zip(
                chunks, embeddings, sparse_embeddings
            ):
                node["embedding"] = embedding
                node["sparse_embedding"] = sparse_embedding
        else:
            logger.info("No nodes to process.")

        self.persist(chunks)
        return chunks
