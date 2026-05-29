from typing import List
from api.api_exception import ApiException, handle_api_exceptions
from common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL
from fastapi import APIRouter
from pydantic import BaseModel
from openai.types.embedding import Embedding
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse,
    Usage as EmbeddingUsage,
)
from service.factory.model_factory import create_embedding_model
from service.model.embedding_service import EmbeddingService
from service.injection import get_embedding_service
from service.injection import get_tenant_id
from db.db_context import get_db_session
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

embedding_router = APIRouter()


class EmbeddingInput(BaseModel):
    input: str | List[str] = None
    model: str = DEFAULT_EMBEDDING_MODEL


@embedding_router.post("")
@handle_api_exceptions(action="embed")
async def aembed(
    embedding_input: EmbeddingInput,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> CreateEmbeddingResponse:
    if embedding_input.input is None:
        raise ApiException(code=400, message=f"Embedding的Input输入'{embedding_input.input}'不可以为空。")

    text_inputs = []
    if isinstance(embedding_input.input, str):
        text_inputs = [embedding_input.input]
    elif isinstance(embedding_input.input, list):
        text_inputs = embedding_input.input
        if not all(
            item is not None and isinstance(item, str)
            for item in text_inputs
        ):
            raise ApiException(code=400, message="Embedding的Input列表元素必须都是非null的字符串。")
    else:
        raise ApiException(code=400, message="Embedding的Input输入必须是字符串或者字符串数组。")


    logger.info(f"Start embedding: {text_inputs}.")
    if embedding_input.model == "bge-m3":
        embedding_input.model = DEFAULT_EMBEDDING_MODEL

    embedding_entity = await embedding_service.get_embedding_by_model_id(embedding_input.model, tenant_id=tenant_id)
    if not embedding_entity:
        raise ApiException(code=400, message=f"Embedding model {embedding_input.model} not found.")

    embed_model = create_embedding_model(embedding_entity)
    text_embeddings = await embed_model.aget_text_embedding_batch(text_inputs)
    embedding_data_list = [
        Embedding(
            embedding=embedding,
            index=i,
            object="embedding",
        )
        for i, embedding in enumerate(text_embeddings)
    ]
    logger.info(f"aembed: finished embedding {len(embedding_data_list)} texts.")
    return CreateEmbeddingResponse(
        object="list",
        data=embedding_data_list,
        model=embedding_input.model,
        usage=EmbeddingUsage(
            prompt_tokens=0,
            total_tokens=0,
        ),
    )
