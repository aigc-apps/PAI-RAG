from chat.agent.state import AgentState
from chat.agent_builder import build_agent
from chat.llm.utils import convert_gen_to_stream_chat_completions, convert_gen_to_chat_completions
from fastapi import APIRouter
from sse_starlette import EventSourceResponse
from common.chat.models import DEFAULT_GUARDRAIL_ADVICE, ChatAgentRequest
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
            prompts=chatbot.prompts,
        )
        logger.info(f"正在调用应用{chat_request.model}: {new_chat_request}")
        return new_chat_request

    raise ValueError(f"Unknown model id: {chat_request.model}")


@chat_agent_router.post("")
async def chat(chat_request: ChatAgentRequest):
    logger.info(f"Chat agent body: {chat_request}")

    try:
        new_chat_request = parse_chat_request(chat_request) # 利用chatbot信息
        agent = await build_agent(new_chat_request)
        state = AgentState.from_messages(
            messages=new_chat_request.messages,
            enable_agent=new_chat_request.enable_agent,
        )
        async_response_gen = await agent.run_async(state=state)
        if chat_request.stream:
            return EventSourceResponse(
                convert_gen_to_stream_chat_completions(
                    new_chat_request.model,
                    async_response_gen
                ),
                media_type="text/event-stream",
            )
        else:
            return await convert_gen_to_chat_completions(
                new_chat_request.model,
                async_response_gen
            )
    except ValueError as ve:
        logger.exception(f"Chat failed: {traceback.format_exc()}")
        raise ValueError(f"Chat failed: {ve}")
    except Exception as ex:
        logger.exception(f"Error in /api/chat: {traceback.format_exc()}")
        raise ValueError(f"Chat failed: {ex}")
