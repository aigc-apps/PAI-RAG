### Embedding configuration API ###

from typing import List
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models.change_event import ChangeEventSource, ChangeEventType
from pairag.db.models.knowledgebase.embedding import (
    EmbeddingModel,
    EmbeddingModelCreate,
    EmbeddingModelEntity,
    EmbeddingModelRead,
    EmbeddingType,
)
from pairag.db.db_context import get_session
from pairag.db.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.embedding_provider import embedding_provider
from pairag.api.response_model import ResponseModel, success_response, error_response
from pairag.mcp.providers.config_change_manager import config_change_manager
from loguru import logger

embedding_router = APIRouter()


@embedding_router.post("", response_model=ResponseModel[EmbeddingModel])
async def create_embedding(
    embedding_create: EmbeddingModelCreate, session: AsyncSession = Depends(get_session)
):
    encrypted_api_key = encrypt_key(embedding_create.api_key)
    embedding_create.is_ready = embedding_create.type == EmbeddingType.OPENAI_LIKE

    embedding = EmbeddingModelEntity.model_validate(
        embedding_create, update={"encrypted_api_key": encrypted_api_key}
    )

    try:
        embedding_provider.add(embedding)
        session.add(embedding)
        await session.commit()
        await session.refresh(embedding)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.EMBEDDING,
            source_id=embedding.id,
            event_type=ChangeEventType.ADD
        )
        if embedding.type == EmbeddingType.LOCAL:
            import pairag.mcp.rag.file_worker as worker
            worker.download_model.delay(model_id=embedding.id, model_name=embedding.model_name)
        return success_response(data=embedding, message="创建embedding模型成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add embedding: {e.orig}")
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
    response_model=ResponseModel[List[EmbeddingModelRead]]
    | ResponseModel[EmbeddingModelRead],
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
            EmbeddingModelRead.model_validate(embedding)
            for embedding in embedding_entities
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


@embedding_router.patch("/{emb_id}", response_model=ResponseModel[EmbeddingModelRead])
async def update_embedding(
    emb_id: str,
    new_embedding: EmbeddingModelCreate,
    session: AsyncSession = Depends(get_session),
):
    embedding_model = await session.get(EmbeddingModelEntity, emb_id)

    if not embedding_model:
        return JSONResponse(
            content=error_response(
                code=404, message=f"查询embedding失败: 模型'{emb_id}'不存在。"
            ),
            status_code=404,
        )

    logger.info(f"Updating Embedding {emb_id} to {new_embedding}.")
    embedding_model.model_name = new_embedding.model_name or embedding_model.model_name
    embedding_model.dimension = new_embedding.dimension or embedding_model.dimension
    embedding_model.type = new_embedding.type
    embedding_model.endpoint = new_embedding.endpoint or embedding_model.endpoint
    embedding_model.is_ready = new_embedding.is_ready
    embedding_model.encrypted_api_key = (
        encrypt_key(new_embedding.api_key)
        if new_embedding.api_key
        else embedding_model.encrypted_api_key
    )

    embedding_provider.update(embedding_model)

    session.add(embedding_model)
    await session.commit()
    await session.refresh(embedding_model)

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.EMBEDDING,
        source_id=embedding_model.id,
        event_type=ChangeEventType.UPDATE,
    )

    logger.info(f"Embedding {emb_id} updated to {embedding_model}.")

    return success_response(data=embedding_model, message="Embedding模型更新成功。")


@embedding_router.delete("/{emb_id}")
async def delete_embedding(
    emb_id: str,
    session: AsyncSession = Depends(get_session),
):
    embedding_model = await session.get(EmbeddingModelEntity, emb_id)
    if not embedding_model:
        return JSONResponse(
            content=error_response(
                code=404, message=f"删除embedding失败: 模型'{emb_id}'不存在。"
            ),
            status_code=404,
        )
    embedding_provider.delete(emb_id)

    await session.delete(embedding_model)
    await session.commit()
    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.EMBEDDING,
        source_id=emb_id,
        event_type=ChangeEventType.DELETE,
    )

    logger.info(f"Embedding {emb_id} deleted.")
    return success_response(message=f"Embedding模型{emb_id}删除成功。")
