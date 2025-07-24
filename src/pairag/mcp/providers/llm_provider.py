from typing import Dict, List, Optional
from sqlmodel import select
from pairag.db.encrypt_utils import decrypt_key
from pairag.db.models.llm import LlmModelEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from llama_index.llms.openai_like import OpenAILike
from loguru import logger


@with_async_db_session
async def fetch_llm_models(session: AsyncSession):
    logger.info("[LlmProvider] Start fetching llm model.")
    sql_results = await session.exec(select(LlmModelEntity))
    llm_results = sql_results.all()

    logger.info(f"[LlmProvider] fetched {len(llm_results)} llm models.")

    return llm_results


class LlmProvider:
    def __init__(self):
        self.llm_models_map: Dict[str, OpenAILike] = {}
        self.llm_configs: List[LlmModelEntity] = []

    async def refresh(self):
        logger.info("[LlmProvider] Start refreshing llm models.")
        self.llm_configs = await fetch_llm_models()
        self.llm_models_map = {
            llm.model_id: OpenAILike(
                model=llm.model,
                api_base=llm.base_url,
                api_key=decrypt_key(llm.encrypted_api_key),
                temperature=llm.temperature,
                max_tokens=llm.context_window,
                is_chat_model=True,
                is_function_calling_model=True,
            )
            for llm in self.llm_configs
        }
        logger.info(f"[LlmProvider]refreshed {len(self.llm_models_map)} llm models.")

    def get_llm_model(self, model_id: str) -> OpenAILike:
        assert model_id in self.llm_models_map, f"Model {model_id} not found."
        return self.llm_models_map[model_id]

    # 获取多模态大模型，如果没找到，直接返回None
    def get_multimodal_llm(self, model_id: str | None = None) -> Optional[OpenAILike]:
        if model_id is None:
            for llm in self.llm_configs:
                if llm.vision_support:
                    logger.info(f"[LLMProvider] found multimodal llm {llm.model_id}.")
                    return self.get_llm_model(llm.model_id)
            logger.info(
                "[LLMProvider] No multimodal llm available. Will not process images in knowledgebase files."
            )
            return None
        else:
            assert model_id in self.llm_models_map, f"Model {model_id} not found."
            return self.llm_models_map[model_id]


llm_provider = LlmProvider()
