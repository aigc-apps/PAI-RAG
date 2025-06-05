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
    llm: LlmModelCreate, session: AsyncSession = Depends(db_context.get_session)
):
    encrypted_api_key = encrypt_key(llm.api_key)
    llm_entity = LlmModelEntity.model_validate(
        llm, update={"encrypted_api_key": encrypted_api_key}
    )
    await session.add(llm)
    await session.commit()
    await session.refresh(llm_entity)
    return llm_entity


@config_router.get("/llms/", response_model=List[LlmModelRead])
async def get_llms(
    session: AsyncSession = Depends(db_context.get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    llms = await session.exec(select(LlmModelEntity).offset(offset).limit(limit)).all()
    return llms


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
    llm: LlmModelCreate,
    session: AsyncSession = Depends(db_context.get_session),
):
    return


async def get_models():
    return {
        "data": [
            {
                "id": "default",
                "object": "model",
                "created": 1739298766,
                "owned_by": "pai",
                "permission": [],
            }
        ]
    }
