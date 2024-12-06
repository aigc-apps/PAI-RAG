import os
import asyncio
import numpy as np
from loguru import logger
from pai_rag.core.rag_module import resolve
from pai_rag.utils.embed_utils import download_url
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding
from pai_rag.integrations.index.pai.utils.sparse_embed_function import (
    BGEM3SparseEmbeddingFunction,
)
from pai_rag.integrations.embeddings.pai.pai_multimodal_embedding import (
    PaiMultiModalEmbedding,
)
from pai_rag.utils.constants import DEFAULT_MODEL_DIR


class EmbedActor:
    def __init__(self, working_dir, config_file):
        RAY_ENV_MODEL_DIR = os.path.join(working_dir, "model_repository")
        os.environ["PAI_RAG_MODEL_DIR"] = RAY_ENV_MODEL_DIR
        logger.info(f"Init EmbedActor with model dir: {RAY_ENV_MODEL_DIR}.")
        self.config = RagConfigManager.from_file(config_file).get_value()

        self.download_model_list = []
        if self.config.embedding.source == "HuggingFace":
            self.download_model_list.append(self.config.embedding.model)
        if self.config.embedding.enable_sparse:
            self.download_model_list.append("bge-m3")
        if self.config.index.enable_multimodal:
            self.download_model_list.append("chinese-clip-vit-large-patch14")
        self.load_models(self.download_model_list)
        logger.info("EmbedActor init finished.")

    def load_models(self, model_list):
        logger.info(f"Downloading models {model_list}.")
        download_models = ModelScopeDownloader(
            fetch_config=True,
            download_directory_path=os.getenv("PAI_RAG_MODEL_DIR", DEFAULT_MODEL_DIR),
        )
        for model_name in model_list:
            download_models.load_model(model=model_name)

        logger.info("Loading models.")
        self.embed_model = resolve(cls=PaiEmbedding, embed_config=self.config.embedding)
        logger.info(f"Dense embed model loaded {self.config.embedding}.")
        self.sparse_embed_model = None
        if self.config.embedding.enable_sparse:
            self.sparse_embed_model = BGEM3SparseEmbeddingFunction(
                model_name_or_path=os.getenv("PAI_RAG_MODEL_DIR", DEFAULT_MODEL_DIR)
            )
            logger.info("Sparse embed model loaded.")

        self.multimodal_embed_model = None
        if self.config.index.enable_multimodal:
            self.multimodal_embed_model = resolve(
                cls=PaiMultiModalEmbedding,
                multimodal_embed_config=self.config.multimodal_embedding,
            )
            logger.info("Multimodal embed model loaded.")

    def __call__(self, nodes):
        text_indices = np.where(nodes["type"] == "text")[0]
        image_indices = np.where(nodes["type"] == "image")[0]
        if text_indices.size > 0:
            text_contents = nodes["text"][text_indices]
            text_embeddings = self.embed_model.get_text_embedding_batch(text_contents)
            for idx, emb in zip(text_indices, text_embeddings):
                nodes["embedding"][idx] = np.array(emb)
            if self.sparse_embed_model:
                sparse_embeddings = self.sparse_embed_model.encode_documents(
                    list(text_contents)
                )
                for idx, emb in zip(text_indices, sparse_embeddings):
                    nodes["sparse_embedding"][idx] = emb
        else:
            logger.info("No image nodes to process.")

        if image_indices.size > 0 and self.multimodal_embed_model:
            images = asyncio.run(
                self.load_images_from_nodes(list(nodes["image_url"][image_indices]))
            )
            image_embeddings = self.multimodal_embed_model.get_image_embedding_batch(
                images
            )
            for idx, emb in zip(image_indices, image_embeddings):
                nodes["embedding"][idx] = np.array(emb)
        else:
            logger.info("No image nodes to process.")

        nodes["embedding"] = np.array(nodes["embedding"], dtype=object)
        return self.process_extra_metadata(nodes)

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
