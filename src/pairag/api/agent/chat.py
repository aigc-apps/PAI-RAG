from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from pairag.mcp.agent_loop import AgentLoop
from pairag.mcp.models import ChatAgentRequest
from pairag.mcp.stream_text import VercelAiDataStreamWriter
import traceback
from loguru import logger


chat_agent_router = APIRouter()


@chat_agent_router.post("")
async def chat(chat_request: ChatAgentRequest):
    logger.info(f"Chat agent body: {chat_request}")
    try:
        agent_loop = AgentLoop()
        async_response_gen = await agent_loop.arun(chat_request=chat_request)

        data_stream_writer = VercelAiDataStreamWriter()
        return StreamingResponse(
            data_stream_writer.astream_text(async_response_gen),
            media_type="text/event-stream",
            headers={"x-vercel-ai-data-stream": "v1"},
        )

    except Exception:
        logger.exception(f"Error in /api/chat: {traceback.format_exc()}")
        return Response(content="Internal Server Error", status_code=500)
