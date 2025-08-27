from fastapi import APIRouter
from sse_starlette import EventSourceResponse
from chat.agent_loop import AgentLoop
from common.chat.models import DEFAULT_GUARDRAIL_ADVICE, ChatAgentRequest
from utils.openai_response_converter import OpenAIChatCompletionConverter
from config.providers.llm_provider import llm_provider
from config.providers.chatbot_provider import chatbot_provider
import traceback
from loguru import logger


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
            enable_agent=chatbot.enable_agent,
            kb_ids=chatbot.kb_ids,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            enable_input_guardrail=chatbot.enable_input_guardrail,
            enable_output_guardrail=chatbot.enable_output_guardrail,
            guardrail_hint=chatbot.guardrail_hint or DEFAULT_GUARDRAIL_ADVICE,
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
        openai_converter = OpenAIChatCompletionConverter(chat_request)
        if chat_request.stream:
            return EventSourceResponse(
                openai_converter.astream_convert(
                    async_response_gen
                ),
                media_type="text/event-stream",
            )
        else:
            return await openai_converter.aconvert(async_response_gen)
    except ValueError as ve:
        logger.exception(f"Chat failed: {traceback.format_exc()}")
        raise ValueError(f"Chat failed: {ve}")
    except Exception as ex:
        logger.exception(f"Error in /api/chat: {traceback.format_exc()}")
        raise ValueError(f"Chat failed: {ex}")
