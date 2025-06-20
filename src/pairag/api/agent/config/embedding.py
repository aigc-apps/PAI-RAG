### Embedding configuration API ###

import asyncio
from typing import List
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models.knowledgebase.embedding import (
    EmbeddingModel,
    EmbeddingModelCreate,
    EmbeddingModelEntity,
)
from pairag.db.db_context import get_session
from pairag.db.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.embedding_provider import embedding_provider
from pairag.api.response_model import ResponseModel, success_response, error_response
from loguru import logger

embedding_router = APIRouter()


@embedding_router.post("", response_model=ResponseModel[EmbeddingModel])
async def create_embedding(
    embedding_create: EmbeddingModelCreate, session: AsyncSession = Depends(get_session)
):
    encrypted_api_key = encrypt_key(embedding_create.api_key)
    embedding = EmbeddingModelEntity.model_validate(
        embedding_create, update={"encrypted_api_key": encrypted_api_key}
    )

    session.add(embedding)
    try:
        await session.commit()
        await session.refresh(embedding)
        asyncio.create_task(embedding_provider.refresh())

        return success_response(data=embedding, message="创建embedding模型成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add llm: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    code=400, message=f"创建embedding模型失败: '{embedding.model_name}'已存在."
                ),
            )
        else:
            return JSONResponse(
                status_code=400,
                content=error_response(code=400, message=f"创建embedding模型失败: '{e}'."),
            )
    except Exception as e:
        await session.rollback()
        return JSONResponse(
            status_code=400,
            content=error_response(code=400, message=f"创建embedding模型失败: '{e}'."),
        )


@embedding_router.get(
    "",
    response_model=ResponseModel[List[EmbeddingModel]] | ResponseModel[EmbeddingModel],
)
async def get_embeddings(
    session: AsyncSession = Depends(get_session),
    model_name: str = None,
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    if not model_name:
        sql_results = await session.exec(
            select(EmbeddingModelEntity).offset(offset).limit(limit)
        )
        embedding_entities = sql_results.all()
        embedding_models = [
            EmbeddingModel.model_validate(embedding) for embedding in embedding_entities
        ]

        return success_response(data=embedding_models, message="查询embedding模型列表成功")
    else:
        statement = select(EmbeddingModelEntity).where(
            EmbeddingModelEntity.model_name == model_name
        )
        embedding_model = (await session.exec(statement)).first()
        if not embedding_model:
            return JSONResponse(
                content=error_response(
                    code=404, message=f"查询embedding模型失败: 模型'{model_name}'不存在。"
                ),
                status_code=404,
            )

        return success_response(data=embedding_model, message="查询embedding模型成功。")


@embedding_router.patch("", response_model=ResponseModel[EmbeddingModel])
async def update_embedding(
    model_name: str,
    new_embedding: EmbeddingModelCreate,
    session: AsyncSession = Depends(get_session),
):
    statement = select(EmbeddingModelEntity).where(
        EmbeddingModelEntity.model_name == model_name
    )
    embedding_model = (await session.exec(statement)).first()
    if not embedding_model:
        return JSONResponse(
            content=error_response(
                code=404, message=f"查询embedding失败: 模型'{model_name}'不存在。"
            ),
            status_code=404,
        )

    embedding_model.dimension = new_embedding.dimension or embedding_model.dimension
    embedding_model.type = new_embedding.type
    embedding_model.endpoint = new_embedding.endpoint or embedding_model.endpoint
    embedding_model.encrypted_api_key = (
        encrypt_key(new_embedding.api_key)
        if new_embedding.api_key
        else embedding_model.encrypted_api_key
    )

    session.add(embedding_model)
    await session.commit()
    await session.refresh(embedding_model)

    asyncio.create_task(embedding_provider.refresh())
    logger.info(f"Embedding {model_name} updated to {embedding_model}.")

    return success_response(data=embedding_model, message="Embedding模型更新成功。")


@embedding_router.delete("")
async def delete_embedding(
    model_name: str,
    session: AsyncSession = Depends(get_session),
):
    statement = select(EmbeddingModelEntity).where(
        EmbeddingModelEntity.model_name == model_name
    )
    embedding_model = (await session.exec(statement)).first()
    if not embedding_model:
        return JSONResponse(
            content=error_response(
                code=404, message=f"删除embedding失败: 模型'{model_name}'不存在。"
            ),
            status_code=404,
        )

    await session.delete(embedding_model)
    await session.commit()
    asyncio.create_task(embedding_provider.refresh())

    logger.info(f"Embedding model {model_name} deleted.")
    return success_response(message=f"Embedding模型{model_name}删除成功。")
