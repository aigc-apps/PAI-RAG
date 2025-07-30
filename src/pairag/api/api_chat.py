from fastapi import APIRouter
from openai.types.create_embedding_response import CreateEmbeddingResponse
from sse_starlette import EventSourceResponse
from pairag.chat.models import ChatCompletionRequest, EmbeddingInput
from pairag.core.chat_service import chat_service


router_openai = APIRouter()
router_chat = APIRouter()


### OpenAI API ###


@router_openai.get("/models")
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


@router_openai.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.stream:
        response = await chat_service.achat(request)
        return response
    else:
        response = await chat_service.astream_chat(request)
        return EventSourceResponse(
            response,
            media_type="text/event-stream",
        )


@router_openai.post("/embeddings")
async def aembed(
    embedding_input: EmbeddingInput,
) -> CreateEmbeddingResponse:
    return await chat_service.aembed(embedding_input)


@router_chat.post("/intent")
async def recognize_intent(request: ChatCompletionRequest):
    return await chat_service.arecognize_intent(request)
