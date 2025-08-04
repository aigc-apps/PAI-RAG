from loguru import logger
from typing import Type
from sqlmodel import SQLModel
from db.models.prompt import PromptModelEntity, PromptModel
from config.providers.base_provider import BaseConfigProvider
DEFAULT_PROMPT_ID = "default_prompt_id"



class PromptProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = PromptModelEntity

    def _load_entries(self, entries):
        super()._load_entries(entries)

    def update(self, entry):
        super().update(entry)

    def get_prompts(self, id: str=DEFAULT_PROMPT_ID):
        prompt_entity = self.config_map.get(id)

        if prompt_entity:
            return prompt_entity
        else:
            logger.warning("No prompt configuration found, using defaults")
            return PromptModel()



prompt_provider = PromptProvider()
