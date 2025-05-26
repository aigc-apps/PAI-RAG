from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pairag.chat.models import ChatCompletionRequest
from pairag.core.chat_service import chat_service

router_openai = APIRouter()


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
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )
