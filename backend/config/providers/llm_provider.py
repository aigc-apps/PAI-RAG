import traceback
import os
from db.db_context import with_async_db_session
import openai
from typing import Dict, Optional, Type
from chat.llm.llm_model import PaiLlm
from sqlmodel import Field, SQLModel
from loguru import logger
from common.encrypt_utils import decrypt_key, encrypt_key
from config.providers.base_provider import BaseConfigProvider
from db.models.llm import LlmModelEntity
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import select

def try_get_initial_model_from_env():
    endpoint = os.environ.get("PAIRAG_RAG__LLM__endpoint")
    if not endpoint:
        return None

    if not endpoint.endswith("/v1"):
        endpoint = endpoint.rstrip("/") + "/v1"

    token = os.environ.get("PAIRAG_RAG__LLM__token") or "abc"

    client = openai.OpenAI(api_key=token, base_url=endpoint)
    try:
        logger.info(f"Try to load models from {endpoint}:{token}.")
        models = client.models.list()
        if len(models.data) > 0:
            logger.info(f"Loaded default llm model {models.data[0].id}")
            return LlmModelEntity.model_validate({
                "base_url": endpoint,
                "encrypted_api_key": encrypt_key(token),
                "model": models.data[0].id,
                "model_id": models.data[0].id,
                "source": "OpenAI-Compatible",
            })
    except Exception as ex:
        logger.warning(f"Load model list failed: {ex}")
        pass

    return None


class LlmProvider(BaseConfigProvider):
    model_id_to_entry_id: Dict[str, str] = Field(default={})
    entity_class: Type[SQLModel] = LlmModelEntity

    def _load_entries(self, entries):
        super()._load_entries(entries)
        for entry_id, entry in self.config_map.items():
            self.model_id_to_entry_id[entry.model_id] = entry_id

    @with_async_db_session
    async def full_load_from_db_async(self, session: AsyncSession):
        entries = [LlmModelEntity.model_validate(entry) for entry in (await session.exec(select(LlmModelEntity))).all()]
        default_model = try_get_initial_model_from_env()
        if default_model and all([entry.base_url != default_model.base_url and entry.model != default_model.model for entry in entries]):
            logger.info("Default model not initialized, inserting into db.")
            try:
                session.add(default_model)
                await session.commit()
            except Exception as ex:
                logger.warning(f"Failed to add default model: {ex}.")
                await session.rollback()
            entries = [default_model] + entries

        self._load_entries(entries)

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
