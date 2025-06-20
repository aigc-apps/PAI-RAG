import os
from fastapi import APIRouter, Depends, Request, Response
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models.websearch import (
    WebSearchConfigEntity,
)
from pairag.db.db_context import get_session
from pairag.db.encrypt_utils import decrypt_key
from pairag.mcp.chat import handle_chat
from pairag.mcp.tools.think.think_and_planning_tool import aget_simple_think_tool
from pairag.mcp.prompts import (
    PROMPT_WITH_DEEP_RESEARCH,
    PROMPT_WITHOUT_DEEP_RESEARCH,
    PROMPT_WITHOUT_TOOLS,
)
from pairag.mcp.utils.message_utils import convert_to_chat_messages
from pairag.mcp.utils.time_utils import get_prompt_current_time_str
from pairag.mcp.tools.search.aliyun_search_tool import aget_aliyun_search_tool
from pairag.mcp.providers.mcp_tool_provider import mcp_provider
from pairag.mcp.providers.llm_provider import llm_provider
from loguru import logger


chat_agent_router = APIRouter()


@chat_agent_router.post("")
async def chat(request: Request, session: AsyncSession = Depends(get_session)):
    try:
        # 解析请求体
        data = await request.json()
        messages = data.get("messages", [])

        # 从 headers 中获取模型参数
        model_id = request.headers.get("X-Model-Id")
        llm = llm_provider.get_llm_model(model_id=model_id)

        x_options = (
            request.headers.get("X-Options").split(",")
            if request.headers.get("X-Options")
            else []
        )
        system = data.get("system", x_options_to_prompt_mode(x_options))

        mcp_tools = []
        # 获取思考工具
        think_cache = []
        think_tool = await aget_simple_think_tool(think_cache=think_cache)
        mcp_tools.append(think_tool)

        if "search" in x_options:
            sql_result = await session.exec(select(WebSearchConfigEntity))
            search_entity = sql_result.first()

            if search_entity is None:
                logger.error("Search config not exists.")
                raise

            os.environ["WEBSEARCH_ACCESS_KEY_ID"] = decrypt_key(
                search_entity.encrypted_access_key_id
            )
            os.environ["WEBSEARCH_ACCESS_KEY_SECRET"] = decrypt_key(
                search_entity.encrypted_access_key_secret
            )
            search_tool = await aget_aliyun_search_tool()
            mcp_tools.append(search_tool)
        if "mcp" in x_options:
            active_mcp_names = (
                request.headers.get("X-MCP-NAMES").split(",")
                if request.headers.get("X-MCP-NAMES")
                else []
            )
            logger.info(f"[Model] selected mcp_ids: {active_mcp_names}")

            mcp_tools.extend(mcp_provider.get_mcp_tools(active_mcp_names))
            logger.info(f"[Model] mcp_tools: {mcp_tools}")

        # 构建chat_messages
        full_messages = [{"role": "system", "content": system}] + messages
        full_messages = convert_to_chat_messages(full_messages)

        return await handle_chat(
            llm=llm,
            messages=full_messages,
            tools=mcp_tools,
        )
    except Exception as e:
        logger.exception(f"Error in /api/chat: {str(e)}")
        return Response(content="Internal Server Error", status_code=500)


def x_options_to_prompt_mode(x_options):
    if "search" in x_options or "mcp" in x_options:
        if "thinking" in x_options:
            system_prompt = PROMPT_WITH_DEEP_RESEARCH.format(
                current_datetime=get_prompt_current_time_str()
            )
        else:
            system_prompt = PROMPT_WITHOUT_DEEP_RESEARCH.format(
                current_datetime=get_prompt_current_time_str()
            )
    else:
        system_prompt = PROMPT_WITHOUT_TOOLS.format(
            current_datetime=get_prompt_current_time_str()
        )
    return system_prompt
