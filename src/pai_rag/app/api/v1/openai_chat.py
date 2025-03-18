from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pai_rag.app.api.models import ChatCompletionRequest
from pai_rag.core.rag_service import rag_service
from pai_rag.app.api.v1.home import templates

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
        response = await rag_service.achat(request)
        return response
    else:
        response = await rag_service.astream_chat(request)
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_openai.get("", response_class=HTMLResponse)
async def homepage(request: Request):
    # Render the index.html template with a message variable
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": "Hi, welcome to PAI-RAG!"}
    )
