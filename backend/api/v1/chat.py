from agent.state import AgentState
from common.llm.models import ChatResponseGenerator
from common.llm.utils import convert_gen_to_stream_chat_completions, convert_gen_to_chat_completions, error_chunk_gen
from fastapi import APIRouter, Response
from sse_starlette import EventSourceResponse

from common.chat.models import ChatAgentRequest
from openai.types.chat import ChatCompletionMessageParam
import traceback
from service.tool.guardrail_service import GuardrailService
from db.db_context import get_db_session
from service.injection import get_agent_service, get_guardrail_service, get_tenant_id
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from service.factory.extension_factory import create_guardrail_checker
from service.agent.agent_service import AgentService
from extensions.guardrail.guardrail_check import GuardrailChecker
from service.cache.session_history_manager import session_history_manager
from loguru import logger


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


async def generate_reponse(
    chunk_gen: ChatResponseGenerator,
    model: str,
    stream: bool,
    enable_output_check: bool = False,
    guardrail_hint: str | None = None,
    checker: GuardrailChecker | None = None,
    session: AsyncSession = None,
    user_id: str = None,
    session_id: str = None,
    user_message: ChatCompletionMessageParam = None,
):
    if stream:
        return EventSourceResponse(
            convert_gen_to_stream_chat_completions(
                model=model,
                response_generator=chunk_gen,
                enable_output_check=enable_output_check,
                guardrail_hint=guardrail_hint,
                checker=checker,
                session=session,
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
            ),
            media_type="text/event-stream",
        )
    else:
        return await convert_gen_to_chat_completions(
            model=model,
            response_generator=chunk_gen,
            enable_output_check=enable_output_check,
            guardrail_hint=guardrail_hint,
            checker=checker,
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
        )


@chat_agent_router.post("")
async def chat(
    chat_request: ChatAgentRequest,
    response: Response,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
    agent_service: AgentService = Depends(get_agent_service),
):
    logger.info(f"Chat agent body: {chat_request}.")
    if not chat_request.messages:
        return await generate_reponse(
            chunk_gen=error_chunk_gen(message="Hi, how can I help you."),
            model=chat_request.model,
            stream=chat_request.stream,
        )

    # 如果 messages 只有一条消息且 user_id和sesion_id 存在，尝试从 Redis 恢复历史上下文
    if chat_request.session_id and chat_request.user_id and len(chat_request.messages) == 1:
        logger.info(
            f"Messages has only one item, attempting to load history from Redis: "
            f"user_id={chat_request.user_id},"
            f"session_id={chat_request.session_id}"
        )
        history_messages = await session_history_manager.get_history_messages(
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
        )
        if history_messages:
            # 将历史消息添加到当前消息前面
            chat_request.messages = history_messages + chat_request.messages
            logger.info(
                f"Restored {len(history_messages)} history messages, "
                f"total messages: {len(chat_request.messages)}"
            )

    # 保存当前用户消息（用于后续保存到 Redis）
    current_user_message = chat_request.messages[-1]

    try:
        # 创建审核器
        checker = None
        guardrail_config = await guardrail_service.get_guardrail_config_or_create(tenant_id=tenant_id)
        if guardrail_config:
            checker = create_guardrail_checker(guardrail_config)

        agent_state = AgentState.from_messages(messages=chat_request.messages)
        async with agent_service.create_agent(chat_request, tenant_id=tenant_id) as agent:
            # 输入护栏检测
            if chat_request.enable_input_guardrail:
                logger.info("Trying to check input.")
                user_message = extract_user_message(chat_request.messages[-1])
                if not checker:
                    raise ValueError("Guardrail checker config not found.")
                check_result = await checker.acheck_input(text=user_message)
                if check_result.reject:
                    return await generate_reponse(
                        chunk_gen=error_chunk_gen(message=check_result.advice or chat_request.guardrail_hint),
                        model=chat_request.model,
                        stream=chat_request.stream,
                        session=session,
                        user_id=chat_request.user_id,
                        session_id=chat_request.session_id,
                        user_message=current_user_message,
                    )

            async_response_gen = await agent.run_async(
                state=agent_state
            )
            response = await generate_reponse(
                chunk_gen=async_response_gen,
                model=chat_request.model,
                stream=chat_request.stream,
                enable_output_check=chat_request.enable_output_guardrail,
                guardrail_hint=chat_request.guardrail_hint,
                checker=checker,
                session=session,
                user_id=chat_request.user_id,
                session_id=chat_request.session_id,
                user_message=current_user_message,
            )

            return response
    except ValueError as ve:
        logger.exception(f"Chat failed: {traceback.format_exc()}")
        return await generate_reponse(
            chunk_gen=error_chunk_gen(message=f"Request failed: {ve}"),
            model=chat_request.model,
            stream=chat_request.stream,
            session=session,
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
            user_message=current_user_message,
        )
    except Exception as ex:
        logger.exception(f"Error in /v1/chat: {traceback.format_exc()}")
        return await generate_reponse(
            chunk_gen=error_chunk_gen(message=f"Unknown error: {ex}"),
            model=chat_request.model,
            stream=chat_request.stream,
            session=session,
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
            user_message=current_user_message,
        )
