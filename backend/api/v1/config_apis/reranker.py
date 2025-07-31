### Reranker configuration API ###

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.knowledgebase.reranker import (
    RerankerModelCreate,
    RerankerModelEntity,
    RerankerModelRead,
)
from db.db_context import get_session
from db.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from api.response_model import PagedResult, ResponseModel, success_response, error_response
from api.v1.utils.paginate import get_pagination_meta

from loguru import logger

reranker_router = APIRouter()


@reranker_router.post("", response_model=ResponseModel[RerankerModelRead])
async def create_reranker(
    reranker_create: RerankerModelCreate, session: AsyncSession = Depends(get_session)
):
    encrypted_api_key = encrypt_key(reranker_create.api_key)
    reranker = RerankerModelEntity.model_validate(
        reranker_create, update={"encrypted_api_key": encrypted_api_key}
    )

    session.add(reranker)
    try:
        await session.commit()
        await session.refresh(reranker)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.RERANK,
            source_id=reranker.id,
            event_type=ChangeEventType.ADD
        )

        return success_response(data=reranker, message="创建reranker模型成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add reranker: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    code=400, message=f"创建reranker模型失败: '{reranker.model_name}'已存在."
                ),
            )
        else:
            return JSONResponse(
                status_code=400,
                content=error_response(code=400, message=f"创建reranker模型失败: '{e}'."),
            )
    except Exception as e:
        await session.rollback()
        return JSONResponse(
            status_code=400,
            content=error_response(code=400, message=f"创建reranker模型失败: '{e}'."),
        )


@reranker_router.get("")
async def get_rerankers(
    model_name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    if not model_name:
        total_results = await session.exec(
            select(func.count()).select_from(
                select(RerankerModelEntity)
            )
        )
        total_num = total_results.one_or_none()
        pagination = get_pagination_meta(page, size, total_num)
        sql_results = await session.exec(select(RerankerModelEntity).offset(pagination.offset).limit(size))
        reranker_entities = sql_results.all()
        reranker_models = [
            RerankerModelRead.model_validate(reranker)
            for reranker in reranker_entities
        ]

        return success_response(
            data=PagedResult(
                items=reranker_models,
                total=pagination.total,
                pages=pagination.pages,
                page=pagination.page,
                size=pagination.size,
            ),message="查询reranker模型列表成功")
    else:
        statement = select(RerankerModelEntity).where(
            RerankerModelEntity.model_name == model_name
        )
        reranker_model = (await session.exec(statement)).first()
        if not reranker_model:
            return JSONResponse(
                content=error_response(
                    code=404, message=f"查询reranker模型失败: 模型'{model_name}'不存在。"
                ),
                status_code=404,
            )

        return success_response(data=reranker_model, message="查询reranker模型成功。")


@reranker_router.patch("/{reranker_id}", response_model=ResponseModel[RerankerModelRead])
async def update_reranker(
    reranker_id: str,
    new_reranker: RerankerModelCreate,
    session: AsyncSession = Depends(get_session),
):
    reranker_model = await session.get(RerankerModelEntity, reranker_id)

    if not reranker_model:
        return JSONResponse(
            content=error_response(
                code=404, message=f"查询reranker失败: 模型'{reranker_id}'不存在。"
            ),
            status_code=404,
        )

    logger.info(f"Updating Reranker {reranker_id} to {new_reranker}.")
    reranker_model.model_id = new_reranker.model_id or reranker_model.model_id
    reranker_model.model_name = new_reranker.model_name or reranker_model.model_name
    reranker_model.base_url = new_reranker.base_url or reranker_model.base_url
    reranker_model.encrypted_api_key = (
        encrypt_key(new_reranker.api_key)
        if new_reranker.api_key
        else reranker_model.encrypted_api_key
    )

    session.add(reranker_model)
    await session.commit()
    await session.refresh(reranker_model)

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.RERANK,
        source_id=reranker_model.id,
        event_type=ChangeEventType.UPDATE
    )
    logger.info(f"Reranker {reranker_id} updated to {reranker_model}.")

    return success_response(data=reranker_model, message="Reranker模型更新成功。")


@reranker_router.delete("/{reranker_id}")
async def delete_reranker(
    reranker_id: str,
    session: AsyncSession = Depends(get_session),
):
    reranker_model = await session.get(RerankerModelEntity, reranker_id)
    if not reranker_model:
        return JSONResponse(
            content=error_response(
                code=404, message=f"删除reranker失败: 模型'{reranker_id}'不存在。"
            ),
            status_code=404,
        )
    await session.delete(reranker_model)
    await session.commit()
    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.RERANK,
        source_id=reranker_id,
        event_type=ChangeEventType.DELETE
    )

    logger.info(f"Reranker {reranker_id} deleted.")
    return success_response(message=f"Reranker模型{reranker_id}删除成功。")
