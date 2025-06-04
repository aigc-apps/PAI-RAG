from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai.types.create_embedding_response import CreateEmbeddingResponse
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
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_openai.post("/embeddings")
async def aembed(
    embedding_input: EmbeddingInput,
) -> CreateEmbeddingResponse:
    return await chat_service.aembed(embedding_input)


### 原子能力API ###


@router_chat.post("/knowledgebase/v1/chat/completions")
async def chat_knowledgebase(request: ChatCompletionRequest):
    if not request.stream:
        response = await chat_service.achat_knowledgebase_atomic(request)
        return response
    else:
        response = await chat_service.astream_knowledgebase_atomic(request)
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_chat.post("/web/v1/chat/completions")
async def chat_web(request: ChatCompletionRequest):
    if not request.stream:
        response = await chat_service.achat_web_atomic(request)
        return response
    else:
        response = await chat_service.astream_web_atomic(request)
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_chat.post("/llm/v1/chat/completions")
async def chat_llm(request: ChatCompletionRequest):
    if not request.stream:
        response = await chat_service.achat_llm_atomic(request)
        return response
    else:
        response = await chat_service.astream_llm_atomic(request)
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_chat.post("/news/v1/chat/completions")
async def chat_news_agent(request: ChatCompletionRequest):
    if not request.stream:
        response = await chat_service.achat_news_agent_atomic(request)
        return response
    else:
        response = await chat_service.astream_news_agent_atomic(request)
        return StreamingResponse(
            response,
            media_type="text/event-stream",
        )


@router_chat.post("/intent")
async def recognize_intent(request: ChatCompletionRequest):
    return await chat_service.arecognize_intent(request)
