### Embedding configuration API ###

from datetime import datetime, timezone
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
from service.injection import get_chatapp_service, get_tenant_id, get_faq_config_service, get_faq_item_service
from service.tool.chatapp_service import ChatappService
from service.tool.faq_config_service import FAQConfigService
from service.tool.faq_item_service import FAQItemService
from api.api_exception import ApiException
import traceback
from loguru import logger

app_router = APIRouter()

# Import FAQ dependencies
from db.models.faq_config import FAQConfigCreate, FAQConfigEntity
from db.models.faq_item import FAQItemCreate, FAQItemEntity

# FAQ routes - MUST be defined before /{id} routes to avoid route conflicts
# FastAPI matches routes in order, so more specific routes must come first
@app_router.post("/{app_id}/faqs", response_model=ResponseModel[FAQItemEntity], tags=["FAQ"])
async def create_faq_item(
    app_id: str,
    faq_item_create: FAQItemCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    faq_item_service: FAQItemService = Depends(get_faq_item_service),
):
    logger.info(f"Creating FAQ item for app_id: {app_id}")
    try:
        # Get chatbot by app_id to get chatbot_id
        chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
        if not chatbot:
            raise ApiException(code=404, message=f"应用 '{app_id}' 不存在。")

        if not chatbot.faq_id:
            raise ApiException(code=400, message="请先启用FAQ功能。")

        faq_item = await faq_item_service.create_faq_item(
            chatbot_id=chatbot.id,
            faq_id=chatbot.faq_id,
            faq_item_data=faq_item_create,
            tenant_id=tenant_id,
        )
        await session.commit()
        await session.refresh(faq_item)
        return success_response(data=faq_item, message="创建FAQ成功。")
    except ValueError as e:
        logger.error(f"Failed to create FAQ item: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to create FAQ item: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"创建FAQ失败: {traceback.format_exc()}")

@app_router.get("/{app_id}/faqs", tags=["FAQ"])
async def list_faq_items(
    app_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=100, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    faq_item_service: FAQItemService = Depends(get_faq_item_service),
):
    logger.info(f"Listing FAQ items for app_id: {app_id}")
    try:
        # Get chatbot by app_id to get chatbot_id
        chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
        if not chatbot:
            raise ApiException(code=404, message=f"应用 '{app_id}' 不存在。")

        faq_items = await faq_item_service.list_faq_items(
            chatbot_id=chatbot.id,
            faq_id=chatbot.faq_id,
            tenant_id=tenant_id,
            page=page,
            size=size,
        )
        return success_response(data=faq_items, message="查询FAQ列表成功。")
    except ValueError as e:
        logger.error(f"Failed to list FAQ items: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to list FAQ items: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"查询FAQ列表失败: {traceback.format_exc()}")

@app_router.put("/{app_id}/faqs/{faq_item_id}", response_model=ResponseModel[FAQItemEntity], tags=["FAQ"])
async def update_faq_item(
    app_id: str,
    faq_item_id: str,
    faq_item_update: FAQItemCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    faq_item_service: FAQItemService = Depends(get_faq_item_service),
):
    try:
        faq_item = await faq_item_service.update_faq_item(
            id=faq_item_id, update_data=faq_item_update, tenant_id=tenant_id
        )
        await session.commit()
        await session.refresh(faq_item)
        return success_response(data=faq_item, message="更新FAQ成功。")
    except ValueError as e:
        logger.error(f"Failed to update FAQ item: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to update FAQ item: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"更新FAQ失败: {traceback.format_exc()}")

@app_router.delete("/{app_id}/faqs/{faq_item_id}", tags=["FAQ"])
async def delete_faq_item(
    app_id: str,
    faq_item_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    faq_item_service: FAQItemService = Depends(get_faq_item_service),
):
    try:
        await faq_item_service.delete_faq_item(id=faq_item_id, tenant_id=tenant_id)
        await session.commit()
        return success_response(message=f"FAQ'{faq_item_id}'删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete FAQ item: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete FAQ item: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"删除FAQ失败: {traceback.format_exc()}")

@app_router.get("/{app_id}/faq-config", response_model=ResponseModel[FAQConfigEntity], tags=["FAQ"])
async def get_faq_config(
    app_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    faq_config_service: FAQConfigService = Depends(get_faq_config_service),
):
    """Get FAQ config for an app."""
    try:
        # Get chatbot by app_id to get chatbot_id
        chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
        if not chatbot:
            raise ApiException(code=404, message=f"应用 '{app_id}' 不存在。")

        faq_config = await faq_config_service.get_or_create_faq_config(
            chatbot_id=chatbot.id, tenant_id=tenant_id
        )
        await session.commit()
        await session.refresh(faq_config)
        return success_response(data=faq_config, message="获取FAQ配置成功。")
    except ValueError as e:
        logger.error(f"Failed to get FAQ config: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to get FAQ config: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"获取FAQ配置失败: {traceback.format_exc()}")

@app_router.put("/{app_id}/faq-config", response_model=ResponseModel[FAQConfigEntity], tags=["FAQ"])
async def update_faq_config(
    app_id: str,
    faq_config_data: FAQConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    faq_config_service: FAQConfigService = Depends(get_faq_config_service),
):
    """Update FAQ config for an app."""
    try:
        # Get chatbot by app_id to get chatbot_id
        chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
        if not chatbot:
            raise ApiException(code=404, message=f"应用 '{app_id}' 不存在。")

        # Get or create FAQ config
        faq_config = await faq_config_service.get_or_create_faq_config(
            chatbot_id=chatbot.id, tenant_id=tenant_id
        )

        # Update FAQ config using the service method which handles individual fields
        updated_faq_config = await faq_config_service.update_faq_config(
            id=faq_config.id,
            update_data=faq_config_data,
            tenant_id=tenant_id
        )

        await session.commit()
        await session.refresh(updated_faq_config)
        return success_response(data=updated_faq_config, message="更新FAQ配置成功。")
    except ValueError as e:
        logger.error(f"Failed to update FAQ config: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to update FAQ config: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"更新FAQ配置失败: {traceback.format_exc()}")


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
