import traceback
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models.change_event import ChangeEventSource, ChangeEventType
from pairag.db.models.llm import LlmModelCreate, LlmModelRead, LlmModelEntity
from pairag.db.db_context import get_session
from pairag.db.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.config_change_manager import config_change_manager
from pairag.mcp.providers.llm_provider import llm_provider
from pairag.api.agent.utils.paginate import get_pagination_meta
from pairag.api.response_model import PagedResult, success_response, error_response, ResponseModel
from loguru import logger
from fastapi.responses import JSONResponse

### LLM Configuration API ###
llm_router = APIRouter()


llm_url_group_map = {
    "https://dashscope.aliyuncs.com/compatible-mode/v1": "通义千问",
    "https://api.openai.com/v1": "OpenAI",
}


@llm_router.post("", response_model=ResponseModel[LlmModelRead])
async def create_llm(
    llm_data: LlmModelCreate, session: AsyncSession = Depends(get_session)
):
    encrypted_api_key = encrypt_key(llm_data.api_key)
    llm = LlmModelEntity.model_validate(
        llm_data, update={"encrypted_api_key": encrypted_api_key}
    )
    llm.source = llm_url_group_map.get(llm.base_url, "OpenAI-Compatible")
    session.add(llm)
    try:
        llm_provider.add(llm)
        await session.commit()
        await session.refresh(llm)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.LLM,
            source_id=llm.id,
            event_type=ChangeEventType.ADD,
        )

        return success_response(data=llm, message="LLM创建成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add llm: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return JSONResponse(
                content=error_response(code=400, message=f"Model_id {llm_data.model_id} already exists."),
                status_code=400,
            )
        else:
            return JSONResponse(
                content=error_response(code=400, message=f"Failed to add llm config: {str(e)}"),
                status_code=400,
            )
    except Exception as e:
        logger.error(f"Failed to add llm config: {traceback.format_exc()}")
        await session.rollback()
        return JSONResponse(
            content=error_response(code=400, message=f"Failed to add llm config: {str(e)}"),
            status_code=400,
        )


@llm_router.get("/groups")
async def get_llm_groups(
    session: AsyncSession = Depends(get_session),
):
    llm_results = await session.exec(select(LlmModelEntity))
    llms = llm_results.all()

    grouped_results = {}
    for llm in llms:
        if not llm.model:
            continue

        group_name = llm_url_group_map.get(llm.base_url, "OpenAI-Compatible")
        if group_name not in grouped_results:
            grouped_results[group_name] = {
                "id": len(grouped_results),
                "label": group_name,
                "models": [],
            }

        grouped_results[group_name]["models"].append(llm)

    return {"groups": list(grouped_results.values())}


@llm_router.get("")
async def get_llms(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    total_results = await session.exec(
        select(func.count()).select_from(
            select(LlmModelEntity)
        )
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    sql_results = await session.exec(select(LlmModelEntity).offset(pagination.offset).limit(size))
    llm_entities = sql_results.all()
    llm_models = [
        LlmModelRead.model_validate(
            llm,
            update={"source": llm_url_group_map.get(llm.base_url, "OpenAI-Compatible")},
        )
        for llm in llm_entities
    ]

    return success_response(
        data=PagedResult(
            items=llm_models,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取LLM模型列表成功")


@llm_router.get("/{llm_id}", response_model=LlmModelRead)
async def read_llm(llm_id: str, session: AsyncSession = Depends(get_session)):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {llm_id} not found.")

    return llm


@llm_router.patch("/{llm_id}", response_model=ResponseModel[LlmModelRead])
async def update_llm(
    llm_id: str,
    update_llm: LlmModelCreate,
    session: AsyncSession = Depends(get_session),
):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        return JSONResponse(
            content=error_response(code=404, message=f"Failed to update llm config {llm_id}"),
            status_code=400,
        )
    logger.info(f"update_llm {update_llm}.")
    llm.model_id = update_llm.model_id or llm.model_id
    llm.base_url = update_llm.base_url or llm.base_url
    llm.context_window = update_llm.context_window or llm.context_window
    llm.model = update_llm.model or llm.model
    llm.temperature = update_llm.temperature or llm.temperature
    llm.encrypted_api_key = (
        encrypt_key(update_llm.api_key) if update_llm.api_key else llm.encrypted_api_key
    )
    llm.enabled = update_llm.enabled

    logger.info(f"Updating LLM {llm_id} to {llm}.")
    session.add(llm)

    llm_provider.update(llm)
    await session.commit()
    await session.refresh(llm)

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.LLM,
        source_id=llm.id,
        event_type=ChangeEventType.UPDATE,
    )



    return success_response(data=llm, message="LLM更新成功。")


@llm_router.delete("/{llm_id}")
async def delete_llm(
    llm_id: str,
    session: AsyncSession = Depends(get_session),
):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {llm_id} not found.")

    llm_provider.delete(llm_id)
    await session.delete(llm)
    await session.commit()
    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.LLM,
        source_id=llm_id,
        event_type=ChangeEventType.DELETE,
    )


    logger.info(f"LLM {llm_id} deleted.")
    return {"message": f"LLM {llm_id} deleted."}
