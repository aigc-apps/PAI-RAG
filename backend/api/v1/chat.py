from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from chat.agent_loop import AgentLoop
from common.chat.models import ChatAgentRequest
from utils.openai_response_converter import OpenAIChatCompletionChunkConverter
from chat.stream_text import AgentFinalAnswerWriter
import traceback
from loguru import logger
import time


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


agent_answer_dump_router = APIRouter()


@agent_answer_dump_router.post("")
async def get_final_answer(chat_request: ChatAgentRequest):
    logger.info(f"Chat agent body: {chat_request}")
    try:
        start_time = time.time()
        agent_loop = AgentLoop()
        async_response_gen = await agent_loop.arun(chat_request=chat_request)

        final_answer_writer = AgentFinalAnswerWriter()
        final_answer, step = await final_answer_writer.astream_text(async_response_gen)
        end_time = time.time()
        run_time = end_time - start_time
        logger.info(
            f"Final answer: {final_answer}, Step: {step}, Run time: {run_time:.1f}s"
        )
        final_answer_dict = {
            "answer": final_answer,
            "step": step,
            "run_time": round(run_time, 1),
        }
        return final_answer_dict
    except Exception:
        logger.exception(f"Error in /api/chat: {traceback.format_exc()}")
        return Response(content="Internal Server Error", status_code=500)
