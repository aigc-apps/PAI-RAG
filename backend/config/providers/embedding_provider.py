import traceback
from typing import Dict, Type

from sqlmodel import SQLModel
from db.encrypt_utils import decrypt_key
from db.models.knowledgebase.embedding import EmbeddingModelEntity, EmbeddingType
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config.providers.base_provider import BaseConfigProvider
from utils.cuda_utils import infer_cuda_device
from pydantic import Field
from utils.modelscope_utils import download_model_to_directory
from loguru import logger



def create_embedding_model(config: EmbeddingModelEntity) -> BaseEmbedding:
    if config.type == EmbeddingType.OPENAI_LIKE:
        logger.info(
            f"Creating OpenAI like embedding model  {config.model_name} with {config}."
        )
        return OpenAILikeEmbedding(
            api_key=decrypt_key(config.encrypted_api_key),
            model_name=config.model_name,
            dimensions=config.dimension,
            embed_batch_size=config.embed_batch_size,
            api_base=config.endpoint,
        )
    elif config.type == EmbeddingType.LOCAL:
        pai_model_path = download_model_to_directory(config.model_name)
        logger.info(
            f"Creating local embedding model {config.model_name} with path {pai_model_path}."
        )

        return HuggingFaceEmbedding(
            model_name=pai_model_path,
            embed_batch_size=config.embed_batch_size,
            device=infer_cuda_device(),
        )
    else:
        logger.error(f"Unknown embedding type: {config.type}.")
        raise ValueError(f"Unknown embedding type: {config.type}.")



class EmbeddingProvider(BaseConfigProvider):
    config_map: Dict[str, EmbeddingModelEntity] = Field(default={})
    model_id_to_entry_id: Dict[str, str] = Field(default={})
    entity_class: Type[SQLModel] = EmbeddingModelEntity

    def add(self, entry: EmbeddingModelEntity):
        super().add(entry)
        self.model_id_to_entry_id[entry.model_id] = entry.id

    def update(self, entry: EmbeddingModelEntity):
        super().update(entry)
        self.model_id_to_entry_id[entry.model_id] = entry.id

    def delete(self, entry_id: str):
        super().delete(entry_id)
        try:
            for k, v in self.model_id_to_entry_id.items():
                if v == entry_id:
                    del self.model_id_to_entry_id[k]
                    break
        except Exception:
            logger.warning(f"Failed to delete entry with entry_id {entry_id}. error: {traceback.format_exc()}.")

    def _load_entries(self, entries):
        super()._load_entries(entries)
        for entry_id, entry in self.config_map.items():
            self.model_id_to_entry_id[entry.model_id] = entry_id

    def _create_instance(self, config: EmbeddingModelEntity) -> BaseEmbedding:
        return create_embedding_model(config=config)

    def get_embedding_model(self, model_id: str):
        if model_id not in self.model_id_to_entry_id:
            raise ValueError(f"`{model_id}` not found. available model_ids: {self.model_id_to_entry_id}")
        entry_id = self.model_id_to_entry_id[model_id]
        if self.config_map[entry_id].is_ready or self.config_map[entry_id].type == EmbeddingType.OPENAI_LIKE:
            return self.get_instance(entry_id)
        else:
            raise ValueError(f"Embedding model {model_id} is still downloading.")

    def get_embedding_config(self, model_id: str):
        if model_id not in self.model_id_to_entry_id:
            raise ValueError(f"`{model_id}` not found. available model_ids: {self.model_id_to_entry_id}")
        entry_id = self.model_id_to_entry_id[model_id]
        return self.config_map[entry_id]


embedding_provider = EmbeddingProvider()
