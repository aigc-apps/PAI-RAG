from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pai_rag.app.api.models import ChatCompletionRequest
from pai_rag.core.rag_service import rag_service

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
    response = await rag_service.achat(request)
    if not request.stream:
        return response
    else:
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )
