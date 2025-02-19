from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
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


# Set up Jinja2 template rendering
templates = Jinja2Templates(directory="templates")


@router_openai.get("/v1", response_class=HTMLResponse)
async def homepage(request: Request):
    # Render the index.html template with a message variable
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": "Hi, welcome to PAI-RAG!"}
    )
