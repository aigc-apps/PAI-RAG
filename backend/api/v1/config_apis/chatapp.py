### Embedding configuration API ###

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.chatbot import (
    ChatBotCreate,
    ChatBotEntity,
)
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from api.response_model import PagedResult, ResponseModel, success_response, error_response
from config.providers.config_change_manager import config_change_manager
from config.providers.chatbot_provider import chatbot_provider
from api.v1.utils.paginate import get_pagination_meta
from loguru import logger

app_router = APIRouter()


@app_router.post("", response_model=ResponseModel[ChatBotEntity])
async def create_chatbot(
    chatbot_create: ChatBotCreate, session: AsyncSession = Depends(get_session)
):
    chatbot = ChatBotEntity.model_validate(chatbot_create)

    try:
        session.add(chatbot)
        await session.commit()
        await session.refresh(chatbot)

        chatbot_provider.add(chatbot)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.EMBEDDING,
            source_id=chatbot.id,
            event_type=ChangeEventType.ADD
        )
        return success_response(data=chatbot, message="创建应用成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add chatbot: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    code=400, message=f"创建应用失败: '{chatbot.app_id}'已存在."
                ),
            )
        else:
            return JSONResponse(
                status_code=400,
                content=error_response(code=400, message=f"创建应用失败: '{e}'."),
            )
    except Exception as e:
        await session.rollback()
        return JSONResponse(
            status_code=400,
            content=error_response(code=400, message=f"创建应用失败: '{e}'."),
        )


@app_router.get("")
async def get_chatbots(
    app_id: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    if not app_id:
        total_results = await session.exec(
            select(func.count()).select_from(
                select(ChatBotEntity)
            )
        )
        total_num = total_results.one_or_none()
        pagination = get_pagination_meta(page, size, total_num)
        sql_results = await session.exec(select(ChatBotEntity).offset(pagination.offset).limit(size))
        chatbot_entities = sql_results.all()

        return success_response(
            data=PagedResult(
                items=chatbot_entities,
                total=pagination.total,
                pages=pagination.pages,
                page=pagination.page,
                size=pagination.size,
            ),message="查询应用列表成功")
    else:
        statement = select(ChatBotEntity).where(
            ChatBotEntity.app_id == app_id
        )
        app = (await session.exec(statement)).first()
        if not app:
            return JSONResponse(
                content=error_response(
                    code=404, message=f"查询应用失败: '{app_id}'不存在。"
                ),
                status_code=404,
            )

        return success_response(data=app, message="查询应用成功。")

@app_router.put("/{id}", response_model=ResponseModel[ChatBotEntity])
async def update_chatbot(
    id: str,
    new_chatbot: ChatBotCreate,
    session: AsyncSession = Depends(get_session),
):
    chatbot = await session.get(ChatBotEntity, id)
    if not chatbot:
        return JSONResponse(
            content=error_response(
                code=404, message=f"查询应用失败: '{id}'不存在。"
            ),
            status_code=404,
        )

    logger.info(f"正在更新应用 {id} to {new_chatbot}.")
    chatbot.app_id = new_chatbot.app_id or chatbot.app_id
    chatbot.model_id = new_chatbot.model_id or chatbot.model_id
    chatbot.enable_search = new_chatbot.enable_search
    chatbot.enable_agent = new_chatbot.enable_agent
    chatbot.kb_ids = new_chatbot.kb_ids
    chatbot.mcp_ids = new_chatbot.mcp_ids
    chatbot.description = new_chatbot.description
    chatbot.enable_vision = new_chatbot.enable_vision
    chatbot.enable_input_guardrail = new_chatbot.enable_input_guardrail
    chatbot.enable_output_guardrail = new_chatbot.enable_output_guardrail
    chatbot.guardrail_hint = new_chatbot.guardrail_hint

    session.add(chatbot)
    await session.commit()
    await session.refresh(chatbot)


    chatbot_provider.update(chatbot)

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.EMBEDDING,
        source_id=chatbot.id,
        event_type=ChangeEventType.UPDATE,
    )

    logger.info(f"应用 {id} 已更新至： {chatbot}.")

    return success_response(data=chatbot, message="应用更新成功。")


@app_router.delete("/{id}")
async def delete_chatbot(
    id: str,
    session: AsyncSession = Depends(get_session),
):
    chatbot = await session.get(ChatBotEntity, id)
    if not chatbot:
        return JSONResponse(
            content=error_response(
                code=404, message=f"删除应用失败: 应用'{id}'不存在。"
            ),
            status_code=404,
        )

    await session.delete(chatbot)
    await session.commit()
    chatbot_provider.delete(id)
    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.EMBEDDING,
        source_id=id,
        event_type=ChangeEventType.DELETE,
    )

    logger.info(f"应用 '{id}' 已删除。")
    return success_response(message=f"应用'{id}'删除成功。")
