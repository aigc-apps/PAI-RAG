from chat.openai.openai_like import OpenAILike
from loguru import logger
from sqlmodel import select, and_
from common.encrypt_utils import decrypt_key
from db.models.llm import LlmModelEntity
from db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from chat.llm.llm_model import PaiLlm

@with_async_db_session
async def get_llm_from_db(
    session: AsyncSession, model_id: str
) -> PaiLlm:
    config = (await session.exec(
        select(LlmModelEntity).where(LlmModelEntity.model_id == model_id)
    )).first()

    if not config:
        raise ValueError(f"Llm model {model_id} not found.")

    return PaiLlm(
        api_base=config.base_url,
        api_key=decrypt_key(config.encrypted_api_key),
        model=config.model,
        enable_thinking=config.enable_thinking,
        vision_support=config.vision_support,
        temperature=config.temperature,
        context_window=config.context_window,
    )

@with_async_db_session
async def get_multimodal_llm_from_db(
    session: AsyncSession,
) -> OpenAILike:
    config = (await session.exec(
        select(LlmModelEntity).where(and_(
                LlmModelEntity.vision_support,
                LlmModelEntity.enabled
            ))
    )).first()

    if not config:
        logger.warning("No multimodal LLM model found.")
        return None
    return OpenAILike(
        model=config.model,
        api_base=config.base_url,
        api_key=decrypt_key(config.encrypted_api_key),
        temperature=config.temperature,
        max_tokens=config.context_window,
        is_chat_model=True,
        is_function_calling_model=True,
    )
