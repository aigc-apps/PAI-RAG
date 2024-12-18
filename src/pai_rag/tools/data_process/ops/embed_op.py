import ray
import asyncio
import numpy as np
from loguru import logger
from pai_rag.tools.data_process.ops.base_op import BaseOP, OPERATORS
from pai_rag.utils.embed_utils import download_url
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.integrations.embeddings.pai.pai_embedding_config import parse_embed_config
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
from pai_rag.integrations.embeddings.pai.embedding_utils import create_embedding

OP_NAME = "pai_rag_embedder"


@OPERATORS.register_module(OP_NAME)
@ray.remote
class Embedder(BaseOP):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    _accelerator = "cpu"
    _batched_op = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder_cfg = parse_embed_config(
            {
                "source": kwargs.get("source", None),
                "model": kwargs.get("model", None),
                "enable_sparse": kwargs.get("enable_sparse", None),
            }
        )
        self.download_model_list = []
        if self.embedder_cfg.source.lower() == "huggingface":
            self.download_model_list.append(self.embedder_cfg.model)
        if self.embedder_cfg.enable_sparse:
            self.download_model_list.append("bge-m3")
        self.enable_multimodal = (
            True if kwargs.get("enable_multimodal", None) else False
        )
        if self.enable_multimodal:
            self.mm_embedder_cfg = parse_embed_config(
                {"source": kwargs.get("multimodal_source", None)}
            )
            self.download_model_list.append("chinese-clip-vit-large-patch14")
        self.load_models(self.download_model_list, kwargs)
        logger.info("Embedder init finished.")

    def load_models(self, model_list, kwargs):
        logger.info(f"Downloading models {model_list}.")
        download_models = ModelScopeDownloader(
            fetch_config=True,
            download_directory_path=self.model_dir,
        )
        for model_name in model_list:
            download_models.load_model(model=model_name)

        logger.info("Loading models.")
        self.embed_model = create_embedding(self.embedder_cfg)
        logger.info(f"Dense embed model loaded {self.embedder_cfg}.")
        self.sparse_embed_model = None
        if self.embedder_cfg.enable_sparse:
            self.sparse_embed_model = BGEM3SparseEmbeddingFunction(
                model_name_or_path=self.model_dir
            )
            logger.info("Sparse embed model loaded.")

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

    async def load_images_from_nodes(self, iamge_urls):
        tasks = [download_url(url) for url in iamge_urls]
        results = await asyncio.gather(*tasks)
        return results

    async def process(self, nodes):
        text_nodes = [node for node in nodes if node["type"] == "text"]
        image_nodes = [node for node in nodes if node["type"] == "image"]
        if len(text_nodes) > 0:
            text_contents = [node["text"] for node in text_nodes]
            embeddings = self.embed_model.get_text_embedding_batch(text_contents)
            if self.sparse_embed_model:
                sparse_embeddings = self.sparse_embed_model.encode_documents(
                    text_contents
                )
            else:
                sparse_embeddings = [None] * len(text_contents)
            # 回填embedding字段
            for node, embedding, sparse_embedding in zip(
                text_nodes, embeddings, sparse_embeddings
            ):
                node["embedding"] = embedding
                node["sparse_embedding"] = sparse_embedding
        else:
            logger.info("No text nodes to process.")

        if len(image_nodes) > 0:
            multimodal_embed_model = create_embedding(
                self.mm_embedder_cfg, pai_rag_model_dir=self.model_dir
            )
            image_urls = [node["image_url"] for node in image_nodes]
            images = await self.load_images_from_nodes(image_urls)
            image_embeddings = multimodal_embed_model.get_image_embedding_batch(images)
            for node, emb in zip(image_nodes, image_embeddings):
                node["embedding"] = emb
        else:
            logger.info("No image nodes to process.")

        return text_nodes + image_nodes
