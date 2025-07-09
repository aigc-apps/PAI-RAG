from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from pairag.mcp.agent_loop import AgentLoop
from pairag.mcp.models import ChatAgentRequest
from pairag.utils.openai_response_converter import OpenAIChatCompletionChunkConverter
import traceback
from loguru import logger


chat_agent_router = APIRouter()


@chat_agent_router.post("")
async def chat(chat_request: ChatAgentRequest):
    logger.info(f"Chat agent body: {chat_request}")
    try:
        agent_loop = AgentLoop()
        async_response_gen = await agent_loop.arun(chat_request=chat_request)
        openai_converter = OpenAIChatCompletionChunkConverter(chat_request)
        return StreamingResponse(
            openai_converter.aconvert_to_openai_chat_completion_chunk(
                async_response_gen
            ),
            media_type="text/event-stream",
        )

    except Exception:
        logger.exception(f"Error in /api/chat: {traceback.format_exc()}")
        return Response(content="Internal Server Error", status_code=500)
