import traceback
from typing import Dict, Optional, Type
from chat.llm.llm_model import PaiLlm
from sqlmodel import Field, SQLModel
from loguru import logger
from common.encrypt_utils import decrypt_key
from config.providers.base_provider import BaseConfigProvider
from db.models.llm import LlmModelEntity


class LlmProvider(BaseConfigProvider):
    model_id_to_entry_id: Dict[str, str] = Field(default={})
    entity_class: Type[SQLModel] = LlmModelEntity

    def _load_entries(self, entries):
        super()._load_entries(entries)
        for entry_id, entry in self.config_map.items():
            self.model_id_to_entry_id[entry.model_id] = entry_id

    def add(self, entry: LlmModelEntity):
        super().add(entry)
        self.model_id_to_entry_id[entry.model_id] = entry.id

    def update(self, entry: LlmModelEntity):
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


    def _create_instance(self, config: LlmModelEntity):
        return PaiLlm(
            api_base=config.base_url,
            api_key=decrypt_key(config.encrypted_api_key),
            model=config.model,
            enable_thinking=config.enable_thinking,
            vision_support=config.vision_support,
            temperature=config.temperature,
            context_window=config.context_window,
        )


    def get_llm_model(self, model_id: str) -> PaiLlm:
        assert model_id in self.model_id_to_entry_id, f"Model {model_id} not found."
        return self.get_instance(self.model_id_to_entry_id[model_id])

    # 获取多模态大模型，如果没找到，直接返回None
    def get_multimodal_llm(self, model_id: str | None = None) -> Optional[PaiLlm]:
        if model_id is None:
            for llm in self.config_map.values():
                if llm.vision_support:
                    logger.info(f"[LLMProvider] found multimodal llm {llm.model_id}.")
                    return self.get_llm_model(llm.model_id)
            logger.info(
                "[LLMProvider] No multimodal llm available. Will not process images in knowledgebase files."
            )
            return None
        else:
            return self.get_llm_model(llm.model_id)


llm_provider = LlmProvider()
