from typing import Dict
from sqlmodel import select
from pairag.db.encrypt_utils import decrypt_key
from pairag.db.models.knowledgebase.reranker import RerankerModelEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.mcp.rag.model.reranker import OpenAICompatibleReranker
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


@with_async_db_session
async def fetch_reranker_models(session: AsyncSession):
    logger.info("[RerankerProvider] Start fetching reranker models.")
    sql_results = await session.exec(select(RerankerModelEntity))
    reranker_results = sql_results.all()

    reranker_config_map = {
        reranker.model_name: reranker for reranker in reranker_results
    }
    logger.info(
        f"[RerankerProvider] fetched {len(reranker_results)} reranker models."
    )
    reranker_models_map = {
        reranker_config.model_name: create_reranker_model(reranker_config)
        for reranker_config in reranker_results
    }
    return reranker_config_map, reranker_models_map


class RerankerProvider:
    def __init__(self):
        self.reranker_config_map: Dict[str, RerankerModelEntity] = {}
        self.reranker_models_map: Dict[str, OpenAICompatibleReranker] = {}

    async def refresh(self):
        logger.info("[RerankerProvider] Start refreshing reranker models.")
        (
            self.reranker_config_map,
            self.reranker_models_map,
        ) = await fetch_reranker_models()
        logger.info(
            f"[RerankerProvider]refreshed {len(self.reranker_models_map)} reranker models."
        )

    def get_reranker_config(self, model_name: str) -> RerankerModelEntity:
        assert (
            model_name in self.reranker_config_map
        ), f"Reranker model '{model_name}' not found"
        return self.reranker_config_map[model_name]

    def get_reranker_model(self, model_name: str) -> OpenAICompatibleReranker:
        assert (
            model_name in self.reranker_models_map
        ), f"Reranker model '{model_name}' not found"
        return self.reranker_models_map[model_name]


reranker_provider = RerankerProvider()
