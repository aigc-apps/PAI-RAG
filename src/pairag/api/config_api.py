from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models import LlmModelEntity, LlmModelCreate, LlmModelRead
from pairag.db.db_context import db_context
from pairag.db.encrypt_utils import encrypt_key


config_router = APIRouter()


### Configuration API ###


@config_router.post("/llms/", response_model=LlmModelRead)
async def create_llm(
    llm_data: LlmModelCreate, session: AsyncSession = Depends(db_context.get_session)
):
    encrypted_api_key = encrypt_key(llm_data.api_key)
    llm = LlmModelEntity.model_validate(
        llm_data, update={"encrypted_api_key": encrypted_api_key}
    )
    session.add(llm)
    await session.commit()
    await session.refresh(llm)
    return llm


@config_router.get("/llms/", response_model=List[LlmModelRead])
async def get_llms(
    session: AsyncSession = Depends(db_context.get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    llm_results = await session.exec(select(LlmModelEntity).offset(offset).limit(limit))
    return llm_results.all()


@config_router.get("/llms/{llm_id}", response_model=LlmModelRead)
async def read_llm(
    llm_id: int, session: AsyncSession = Depends(db_context.get_session)
):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {llm_id} not found.")

    return llm


@config_router.patch("/llms/{llm_id}", response_model=LlmModelRead)
async def update_llm(
    llm_id: int,
    update_llm: LlmModelCreate,
    session: AsyncSession = Depends(db_context.get_session),
):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {llm_id} not found.")

    llm.base_url = update_llm.base_url or llm.base_url
    llm.context_window = update_llm.context_window or llm.context_window
    llm.model = update_llm.model or llm.model
    llm.temperature = update_llm.temperature or llm.temperature
    llm.encrypted_api_key = (
        encrypt_key(update_llm.api_key) if update_llm.api_key else llm.encrypted_api_key
    )

    await session.add(llm)
    await session.commit()
    await session.refresh(llm)

    return llm


@config_router.delete("/llms/{llm_id}")
async def delete_llm(
    llm_id: int,
    session: AsyncSession = Depends(db_context.get_session),
):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {llm_id} not found.")
    await session.delete(llm)
    await session.commit()
    return {"message": f"LLM {llm_id} deleted."}
