from typing import Dict
from sqlmodel import select
from pairag.db.encrypt_utils import decrypt_key
from pairag.db.models.llm import LlmModelEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from llama_index.llms.openai_like import OpenAILike
from loguru import logger


@with_async_db_session
async def fetch_llm_models(session: AsyncSession):
    logger.info("[LlmProvider] Start fetching mcp servers.")
    sql_results = await session.exec(select(LlmModelEntity))
    llm_results = sql_results.all()

    logger.info(f"[LlmProvider] fetched {len(llm_results)} llm models.")
    llm_map = {
        llm.model_id: OpenAILike(
            model=llm.model,
            api_base=llm.base_url,
            api_key=decrypt_key(llm.encrypted_api_key),
            temperature=llm.temperature,
            max_tokens=llm.context_window,
            is_chat_model=True,
            is_function_calling_model=True,
        )
        for llm in llm_results
    }

    return llm_map


class LlmProvider:
    def __init__(self):
        self.llm_models_map: Dict[str, OpenAILike] = {}

    async def refresh(self):
        logger.info("[LlmProvider] Start refreshing llm models.")
        self.llm_models_map = await fetch_llm_models()
        logger.info(f"[LlmProvider]refreshed {len(self.llm_models_map)} llm models.")

    def get_llm_model(self, model_id: str) -> OpenAILike:
        assert model_id in self.llm_models_map, f"Model {model_id} not found."
        return self.llm_models_map[model_id]


llm_provider = LlmProvider()
