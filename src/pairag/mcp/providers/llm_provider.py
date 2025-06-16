from typing import Dict
from sqlmodel import select
from pairag.db.encrypt_utils import decrypt_key
from pairag.db.models.llm import LlmModelEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from openai import AsyncClient
from loguru import logger


class OpenAILlmEntry:
    model: str
    client: AsyncClient

    def __init__(self, model: str, client: AsyncClient):
        self.model = model
        self.client = client


@with_async_db_session
async def fetch_llm_models(session: AsyncSession):
    logger.info("[LlmProvider] Start fetching mcp servers.")
    sql_results = await session.exec(select(LlmModelEntity))
    llm_results = sql_results.all()

    logger.info(f"[LlmProvider] fetched {len(llm_results)} llm models.")
    llm_map = {
        llm.model_id: OpenAILlmEntry(
            model=llm.model,
            client=AsyncClient(
                api_key=decrypt_key(llm.encrypted_api_key),
                base_url=llm.base_url,
                max_retries=3,
                timeout=60,
            ),
        )
        for llm in llm_results
    }

    return llm_map


class LlmProvider:
    def __init__(self):
        self.llm_models_map: Dict[str, OpenAILlmEntry] = {}

    async def refresh(self):
        logger.info("[LlmProvider] Start refreshing llm models.")
        self.llm_models_map = await fetch_llm_models()
        logger.info(f"[LlmProvider]refreshed {len(self.llm_models_map)} llm models.")

    def get_llm_model(self, model_id: str) -> OpenAILlmEntry:
        assert model_id in self.llm_models_map, f"Model {model_id} not found."
        return self.llm_models_map[model_id]


llm_provider = LlmProvider()
