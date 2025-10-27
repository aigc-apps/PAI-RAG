from chat.agent.state import AgentState
from chat.agent_builder import build_agent
from chat.llm.models import ChatResponseGenerator
from chat.llm.utils import convert_gen_to_stream_chat_completions, convert_gen_to_chat_completions, error_chunk_gen
from config.providers.guardrail_provider import guardrail_provider
from fastapi import APIRouter
from sse_starlette import EventSourceResponse
from common.chat.models import DEFAULT_GUARDRAIL_ADVICE, ChatAgentRequest
from config.providers.llm_provider import llm_provider
from config.providers.chatbot_provider import chatbot_provider
from openai.types.chat import ChatCompletionMessageParam
import traceback
from loguru import logger
from concurrent.futures import ThreadPoolExecutor


_EXECUTOR = ThreadPoolExecutor(max_workers=10)


chat_agent_router = APIRouter()



def extract_user_message(raw_msg: ChatCompletionMessageParam) -> str:
    content = raw_msg.get("content", "")
    logger.info(f"Extracted {content} from {raw_msg}.")
    if isinstance(content, str):
        return content
    else:
        user_content = ""
        for block in content:
            user_content += block.get("text", "")
        return user_content

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
            enable_chatdb=chatbot.enable_chatdb or False,
            kb_ids=chatbot.kb_ids,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            enable_input_guardrail=chatbot.enable_input_guardrail,
            enable_output_guardrail=chatbot.enable_output_guardrail,
            guardrail_hint=chatbot.guardrail_hint or DEFAULT_GUARDRAIL_ADVICE,
            prompts=chatbot.prompts,
            user_id=chat_request.user_id,
        )
        logger.info(f"正在调用应用{chat_request.model}: {new_chat_request}")
        return new_chat_request

    raise ValueError(f"Unknown model id: {chat_request.model}")


async def generate_reponse(
    chunk_gen: ChatResponseGenerator,
    model: str,
    stream: bool,
    enable_output_check: bool = False,
    guardrail_hint: str | None = None,
):
    if stream:
        return EventSourceResponse(
            convert_gen_to_stream_chat_completions(
                model,
                chunk_gen,
                enable_output_check,
                guardrail_hint,
            ),
            media_type="text/event-stream",
        )
    else:
        return await convert_gen_to_chat_completions(
            model,
            chunk_gen,
            enable_output_check,
            guardrail_hint,
        )


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

        # 创建审核器
        checker = guardrail_provider.get_checker()

        # 输入护栏检测
        if new_chat_request.enable_input_guardrail:
            user_message = extract_user_message(chat_request.messages[-1])
            check_result = await checker.acheck_input(text=user_message)
            if check_result.reject:
                return await generate_reponse(
                    error_chunk_gen(message=check_result.advice or new_chat_request.guardrail_hint),
                    model=chat_request.model,
                    stream=chat_request.stream,
                )

        async_response_gen = await agent.run_async(state=state)
        return await generate_reponse(
            async_response_gen,
            model=chat_request.model,
            stream=chat_request.stream,
            enable_output_check=new_chat_request.enable_output_guardrail,
            guardrail_hint=new_chat_request.guardrail_hint,
        )
    except ValueError as ve:
        logger.exception(f"Chat failed: {traceback.format_exc()}")
        raise ValueError(f"Chat failed: {ve}")
    except Exception as ex:
        logger.exception(f"Error in /api/chat: {traceback.format_exc()}")
        raise ValueError(f"Chat failed: {ex}")
