### Embedding configuration API ###

from fastapi import APIRouter, Depends, Query
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.chatbot import (
    ChatBotCreate,
    ChatBotEntity,
)
from db.db_context import get_db_session
from sqlalchemy.exc import IntegrityError
from common.chat.response_model import PagedResult, ResponseModel, success_response
from api.v1.utils.paginate import get_pagination_meta
from service.injection import get_chatapp_service, get_tenant_id
from service.tool.chatapp_service import ChatappService
from api.api_exception import ApiException
import traceback
from loguru import logger

app_router = APIRouter()


@app_router.post("", response_model=ResponseModel[ChatBotEntity])
async def create_chatbot(
    chatbot_create: ChatBotCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
):
    try:
        chatbot = await chatapp_service.create_chatapp(app_data=chatbot_create, tenant_id=tenant_id)
        return success_response(data=chatbot, message="创建应用成功。")
    except ValueError as e:
        logger.error(f"Failed to create chatapp: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to create chatapp: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"创建应用失败: {traceback.format_exc()}")


@app_router.get("")
async def get_chatbots(
    app_id: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
):
    try:
        if not app_id:
            chatbots = await chatapp_service.list_chatapps(page=page, size=size, tenant_id=tenant_id)
            return success_response(data=chatbots, message="查询应用列表成功。")
        else:
            chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
            if not chatbot:
                raise ApiException(code=404, message=f"查询应用失败: '{app_id}'不存在。")
            return success_response(data=chatbot, message="查询应用成功。")
    except ValueError as e:
        logger.error(f"Failed to list chatapps: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to list chatapps: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"查询应用列表失败: {traceback.format_exc()}")


@app_router.put("/{id}", response_model=ResponseModel[ChatBotEntity])
async def update_chatbot(
    id: str,
    new_chatbot: ChatBotCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
):
    try:
        chatbot = await chatapp_service.update_chatapp(id=id, update_data=new_chatbot, tenant_id=tenant_id)
        return success_response(data=chatbot, message="更新应用成功。")
    except ValueError as e:
        logger.error(f"Failed to update chatapp: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to update chatapp: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"更新应用失败: {traceback.format_exc()}")


@app_router.delete("/{id}")
async def delete_chatbot(
    id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
):
    try:
        await chatapp_service.delete_chatapp(id=id, tenant_id=tenant_id)
        return success_response(message=f"应用'{id}'删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete chatapp: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete chatapp: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"删除应用失败: {traceback.format_exc()}")
