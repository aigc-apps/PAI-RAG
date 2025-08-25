import traceback
from typing import List
from api.response_model import error_response
from common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL
from config.providers.embedding_provider import embedding_provider
from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger
from openai.types.embedding import Embedding
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse,
    Usage as EmbeddingUsage,
)

embedding_router = APIRouter()


class EmbeddingInput(BaseModel):
    input: str | List[str] = None
    model: str = DEFAULT_EMBEDDING_MODEL



@embedding_router.post("")
async def aembed(
    embedding_input: EmbeddingInput,
) -> CreateEmbeddingResponse:
    if embedding_input.input is None:
        return error_response(code=400, message=f"Embedding的Input输入'{embedding_input.input}'不可以为空。")

    text_inputs = []
    if isinstance(embedding_input.input, str):
        text_inputs = [embedding_input.input]
    elif isinstance(embedding_input.input, list):
        text_inputs = embedding_input.input
        if not all(
            item is not None and isinstance(item, str)
            for item in text_inputs
        ):
            return error_response(code=400, message="Embedding的Input列表元素必须都是非null的字符串。")
    else:
        return error_response(code=400, message="Embedding的Input输入必须是字符串或者字符串数组。")


    logger.info(f"Start embedding: {text_inputs}.")
    if embedding_input.model == "bge-m3":
        embedding_input.model = DEFAULT_EMBEDDING_MODEL

    try:
        embed_model = embedding_provider.get_embedding_model(embedding_input.model)
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
    except ValueError as ve:
        logger.warning(f"Embedding failed due to value error: {traceback.format_exc()}")
        return error_response(code=400, message=f"Embedding失败: {ve}")
    except Exception as ex:
        logger.error(f"Embedding failed: {traceback.format_exc()}")
        return error_response(code=500, message=f"Embedding失败: {ex}")

