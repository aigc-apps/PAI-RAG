### Prompt configuration API ###

from fastapi import APIRouter, Depends
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.prompt import PromptModelEntity
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from db.models.change_event import ChangeEventSource, ChangeEventType
from config.providers.config_change_manager import config_change_manager
from config.providers.prompt_provider import prompt_provider
from api.response_model import success_response, error_response, ResponseModel

from loguru import logger


prompt_router = APIRouter()

@prompt_router.post("", response_model=ResponseModel[PromptModelEntity])
async def set_prompt_config(
    new_prompt_config: PromptModelEntity,
    session: AsyncSession = Depends(get_session),
):
    prompt_config = (await session.exec(select(PromptModelEntity))).first()
    if prompt_config is None:
        logger.info(f"Adding new prompt config {prompt_config}")

        prompt_config = PromptModelEntity.model_validate(
            new_prompt_config,
        )
    else:
        new_prompts = {**prompt_config.prompts,**new_prompt_config.prompts}

        prompt_config.prompts = new_prompts


    session.add(prompt_config)
    try:
        await session.commit()
        await session.refresh(prompt_config)
        prompt_provider.update(prompt_config)
        await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.PROMPT,
        source_id=prompt_config.id,
        event_type=ChangeEventType.UPDATE
    )
        return success_response(data=prompt_config, message="add prompt config successfully")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add prompt config: {e.orig}")
        await session.rollback()
        return error_response(
            code=400, message=f"IntegrityError occurred when add prompt config: {e.orig}"
        )
    except Exception as e:
        await session.rollback()
        return error_response(
            code=400, message=f"Failed to add prompt config: {str(e)}"
        )


@prompt_router.get("", response_model=ResponseModel[PromptModelEntity])
async def get_prompt_config(
    session: AsyncSession = Depends(get_session),
):
    prompt_config_results = await session.exec(select(PromptModelEntity))
    prompt_config = prompt_config_results.first()
    if not prompt_config:
        logger.warning("No prompt config found.")
        return success_response(data=PromptModelEntity(), message="get prompt config successfully")

    return success_response(data=prompt_config, message="get prompt config successfully")
