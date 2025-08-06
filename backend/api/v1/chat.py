from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from api.response_model import error_response
from chat.agent_loop import AgentLoop
from common.chat.models import ChatAgentRequest
from utils.openai_response_converter import OpenAIChatCompletionChunkConverter
from chat.stream_text import AgentFinalAnswerWriter
from config.providers.llm_provider import llm_provider
from config.providers.chatbot_provider import chatbot_provider
import traceback
from loguru import logger
import time


chat_agent_router = APIRouter()


def parse_chat_request(chat_request: ChatAgentRequest) -> ChatAgentRequest:
    if chat_request.model in llm_provider.model_id_to_entry_id:
        return chat_request

    if chat_request.model in chatbot_provider.app_id_to_entry_id:
        chatbot = chatbot_provider.get_chatbot(chat_request.model)
        new_chat_request = ChatAgentRequest(
            model=chatbot.model_id,
            messages=chat_request.messages,
            stream=chat_request.stream,
            mcp_ids=chatbot.mcp_ids,
            enable_search=chatbot.enable_search,
            enable_thinking=chatbot.enable_agent,
            enable_attachments=chat_request.enable_attachments,
            kb_ids=chatbot.kb_ids,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
        )
        logger.info(f"正在调用应用{chat_request.model}: {new_chat_request}")
        return new_chat_request

    raise ValueError(f"Unknown model id: {chat_request.model}")


@chat_agent_router.post("")
async def chat(chat_request: ChatAgentRequest):
    logger.info(f"Chat agent body: {chat_request}")

    try:
        agent_loop = AgentLoop()
        new_chat_request = parse_chat_request(chat_request) # 利用chatbot信息
        async_response_gen = await agent_loop.arun(chat_request=new_chat_request)
        openai_converter = OpenAIChatCompletionChunkConverter(chat_request)
        return StreamingResponse(
            openai_converter.aconvert_to_openai_chat_completion_chunk(
                async_response_gen
            ),
            media_type="text/event-stream",
        )
    except ValueError as ve:
        logger.exception(f"Chat failed: {traceback.format_exc()}")
        return error_response(code=400, message=f"Chat failed: {ve}")
    except Exception:
        logger.exception(f"Error in /api/chat: {traceback.format_exc()}")
        return error_response(message="Internal Server Error: {ex}", code=500)


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
