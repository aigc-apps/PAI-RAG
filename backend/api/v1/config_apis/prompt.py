### Trace configuration API ###

### Prompt configuration API ###

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.prompt import PromptModel, PromptModelEntity
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from db.models.change_event import ChangeEventSource, ChangeEventType
from config.providers.config_change_manager import config_change_manager
from config.providers.prompt_provider import prompt_provider

from loguru import logger


prompt_router = APIRouter()


@prompt_router.post("", response_model=PromptModel)
async def set_prompt_config(
    new_prompt_config: PromptModel,
    session: AsyncSession = Depends(get_session),
):
    prompt_config = (await session.exec(select(PromptModelEntity))).first()
    if prompt_config is None:
        logger.info(f"Adding new prompt config {prompt_config}")

        prompt_config = PromptModelEntity.model_validate(
            new_prompt_config,
        )
    else:
        prompt_config.system_prompt = new_prompt_config.system_prompt or prompt_config.system_prompt
        prompt_config.search_web_tool_prompt = new_prompt_config.search_web_tool_prompt or prompt_config.search_web_tool_prompt
        prompt_config.thinking_tool_prompt = new_prompt_config.thinking_tool_prompt or prompt_config.thinking_tool_prompt
        prompt_config.attachments_tool_prompt = new_prompt_config.attachments_tool_prompt or prompt_config.attachments_tool_prompt
        prompt_config.knowledgebase_tool_prompt = new_prompt_config.knowledgebase_tool_prompt or prompt_config.knowledgebase_tool_prompt
        prompt_config.without_tools_prompt = new_prompt_config.without_tools_prompt or prompt_config.without_tools_prompt


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
        return prompt_config
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add prompt config: {e.orig}")
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=400, detail=f"Failed to add prompt config: {str(e)}"
        )


@prompt_router.get("", response_model=PromptModel)
async def get_prompt_config(
    session: AsyncSession = Depends(get_session),
):
    prompt_config_results = await session.exec(select(PromptModelEntity))
    prompt_config = prompt_config_results.first()
    if not prompt_config:
        logger.warning("No prompt config found.")
        return PromptModel()

    return prompt_config
