from typing import Dict, Optional
from sqlmodel import Field
from llama_index.llms.openai_like import OpenAILike
from loguru import logger
from pairag.mcp.providers.base_provider import BaseConfigProvider


class LlmProvider(BaseConfigProvider):
    model_id_to_entry_id: Dict[str, str] = Field(default={})

    def _load_entries(self, entries):
        super()._load_entries(entries)
        for entry_id, entry in self.config_map.items():
            self.model_id_to_entry_id[entry.model_id] = entry_id

    def get_llm_model(self, model_id: str) -> OpenAILike:
        assert model_id in self.model_id_to_entry_id, f"Model {model_id} not found."
        return self.get_instance(self.model_id_to_entry_id[model_id])

    # 获取多模态大模型，如果没找到，直接返回None
    def get_multimodal_llm(self, model_id: str | None = None) -> Optional[OpenAILike]:
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
