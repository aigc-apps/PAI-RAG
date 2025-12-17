from agent.state import AgentState
from common.llm.models import ChatResponseGenerator
from common.llm.utils import convert_gen_to_stream_chat_completions, convert_gen_to_chat_completions, error_chunk_gen
from fastapi import APIRouter
from sse_starlette import EventSourceResponse

from common.chat.models import ChatAgentRequest
from openai.types.chat import ChatCompletionMessageParam
import traceback
from service.tool.codesandbox_service import CodesandboxService
from service.tool.chatapp_service import ChatappService
from service.tool.guardrail_service import GuardrailService
from db.db_context import get_db_session
from service.injection import get_agent_service, get_chatapp_service, get_guardrail_service, get_llm_service, get_codesandbox_service, get_tenant_id
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from service.factory.extension_factory import create_guardrail_checker
from loguru import logger
from service.agent.agent_service import AgentService
from service.model.llm_service import LlmService
from extensions.guardrail.guardrail_check import GuardrailChecker


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
):
    if stream:
        return EventSourceResponse(
            convert_gen_to_stream_chat_completions(
                model,
                chunk_gen,
                enable_output_check,
                guardrail_hint,
                checker,
            ),
            media_type="text/event-stream",
        )
    else:
        return await convert_gen_to_chat_completions(
            model,
            chunk_gen,
            enable_output_check,
            guardrail_hint,
            checker,
        )


@chat_agent_router.post("")
async def chat(
    chat_request: ChatAgentRequest,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    code_sandbox_service: CodesandboxService = Depends(get_codesandbox_service),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
    llm_service: LlmService = Depends(get_llm_service),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    agent_service: AgentService = Depends(get_agent_service),
):
    logger.info(f"Chat agent body: {chat_request}.")
    try:
        agent = await agent_service.create_agent(chat_request, tenant_id=tenant_id)
        # 创建审核器
        checker = None
        guardrail_config = await guardrail_service.get_guardrail_config_or_create(tenant_id=tenant_id)
        if guardrail_config:
            checker = create_guardrail_checker(guardrail_config)

        # 输入护栏检测
        if chat_request.enable_input_guardrail:
            user_message = extract_user_message(chat_request.messages[-1])
            if not checker:
                raise ValueError("Guardrail checker config not found.")
            check_result = await checker.acheck_input(text=user_message)
            if check_result.reject:
                return await generate_reponse(
                    error_chunk_gen(message=check_result.advice or chat_request.guardrail_hint),
                    model=chat_request.model,
                    stream=chat_request.stream,
                )


        async_response_gen = await agent.run_async(
            state=AgentState.from_messages(
                messages=chat_request.messages,
                enable_agent=chat_request.enable_agent,
            )
        )
        response = await generate_reponse(
            async_response_gen,
            model=chat_request.model,
            stream=chat_request.stream,
            enable_output_check=chat_request.enable_output_guardrail,
            guardrail_hint=chat_request.guardrail_hint,
        )

        return response
    except ValueError as ve:
        logger.exception(f"Chat failed: {traceback.format_exc()}")
        return await generate_reponse(
            error_chunk_gen(message=f"请求失败: {ve}"),
            model=chat_request.model,
            stream=chat_request.stream,
        )
    except Exception as ex:
        logger.exception(f"Error in /api/chat: {traceback.format_exc()}")
        return await generate_reponse(
            error_chunk_gen(message=f"未知错误: {ex}"),
            model=chat_request.model,
            stream=chat_request.stream,
        )
