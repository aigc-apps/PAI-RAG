import traceback
from typing import Dict, Type
from sqlmodel import Field, SQLModel
from common.encrypt_utils import decrypt_key
from db.models.knowledgebase.reranker import RerankerModelEntity
from config.providers.base_provider import BaseConfigProvider
from rag.rerank.reranker import OpenAICompatibleReranker
from loguru import logger


def create_reranker_model(reranker_config: RerankerModelEntity) -> OpenAICompatibleReranker:
    logger.info(
        f"Creating OpenAI compatible reranker model  {reranker_config.model_name} with {reranker_config}."
    )
    return OpenAICompatibleReranker(
        api_key=decrypt_key(reranker_config.encrypted_api_key),
        model=reranker_config.model_name,
        base_url=reranker_config.base_url,
    )


class RerankerProvider(BaseConfigProvider):
    model_id_to_entry_id: Dict[str, str] = Field(default={})
    entity_class: Type[SQLModel] = RerankerModelEntity

    def add(self, entry: RerankerModelEntity):
        super().add(entry)
        self.model_id_to_entry_id[entry.model_id] = entry.id

    def update(self, entry: RerankerModelEntity):
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

    def _create_instance(self, config):
        return create_reranker_model(config)

    def get_reranker_model(self, model_id: str) -> OpenAICompatibleReranker:
        id = self.model_id_to_entry_id.get(model_id)
        assert (
            id in self.config_map
        ), f"Reranker model '{id}' not found"
        return self.get_instance(id)


reranker_provider = RerankerProvider()
