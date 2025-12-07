### Reranker configuration API ###

from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.reranker import (
    RerankerModelCreate,
    RerankerModelRead,
)
from db.db_context import get_db_session
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException
from service.model.reranker_service import RerankerService
from service.injection import get_reranker_service
from loguru import logger

reranker_router = APIRouter()


@reranker_router.post("", response_model=ResponseModel[RerankerModelRead])
async def create_reranker(
    reranker_create: RerankerModelCreate,
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    try:
        reranker = await reranker_service.create_reranker(reranker_create)
        return success_response(data=reranker, message="创建reranker模型成功。")
    except ValueError as e:
        logger.error(f"Failed to create reranker model: {str(e)}")
        raise ApiException(code=400, message=f"创建reranker模型失败: '{e}'.")
    except Exception as e:
        logger.error(f"Failed to create reranker model: {str(e)}")
        raise ApiException(code=500, message=f"创建reranker模型失败: '{e}'.")


@reranker_router.get("")
async def get_rerankers(
    model_name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    try:
        if not model_name:
            reranker_models = await reranker_service.list_rerankers(page=page, size=size)
            return success_response(data=reranker_models, message="查询reranker模型列表成功。")
        else:
            reranker_model = await reranker_service.get_reranker_by_model_name(model_name=model_name)
            return success_response(data=reranker_model, message="查询reranker模型成功。")
    except ValueError as e:
        logger.error(f"Failed to get reranker model: {str(e)}")
        raise ApiException(code=400, message=f"查询reranker模型失败: '{e}'.")
    except Exception as e:
        logger.error(f"Failed to get reranker model: {str(e)}")
        raise ApiException(code=500, message=f"查询reranker模型失败: '{e}'.")


@reranker_router.put("/{reranker_id}", response_model=ResponseModel[RerankerModelRead])
async def update_reranker(
    reranker_id: str,
    new_reranker: RerankerModelCreate,
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    try:
        reranker_model = await reranker_service.update_reranker(reranker_id=reranker_id, update_data=new_reranker)
        return success_response(data=reranker_model, message="更新reranker模型成功。")
    except ValueError as e:
        logger.error(f"Failed to update reranker model: {str(e)}")
        raise ApiException(code=400, message=f"更新reranker模型失败: '{e}'.")
    except Exception as e:
        logger.error(f"Failed to update reranker model: {str(e)}")
        raise ApiException(code=500, message=f"更新reranker模型失败: '{e}'.")


@reranker_router.delete("/{reranker_id}")
async def delete_reranker(
    reranker_id: str,
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    try:
        await reranker_service.delete_reranker(reranker_id=reranker_id)
        return success_response(message="删除reranker模型成功。")
    except ValueError as e:
        logger.error(f"Failed to delete reranker model: {str(e)}")
        raise ApiException(code=400, message=f"删除reranker模型失败: '{e}'.")
    except Exception as e:
        logger.error(f"Failed to delete reranker model: {str(e)}")
        raise ApiException(code=500, message=f"删除reranker模型失败: '{e}'.")
