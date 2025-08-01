from loguru import logger
from typing import Type
from sqlmodel import SQLModel, Field
from db.models.prompt import PromptModelEntity, PromptModel
from config.providers.base_provider import BaseConfigProvider



class PromptProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = PromptModelEntity
    id: str = Field(default="default_prompt_id")

    def _load_entries(self, entries):
        super()._load_entries(entries)

    def update(self, entry):
        super().update(entry)
        self.id = entry.id

    def get_prompts(self):
        prompt_entity = self.config_map.get(self.id)

        if prompt_entity:
            return prompt_entity
        else:
            logger.warning("No prompt configuration found, using defaults")
            return PromptModel()



prompt_provider = PromptProvider()
