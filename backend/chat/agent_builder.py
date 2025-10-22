from typing import List
from chat.agent.planner import PlanAgentPromptSet, Planner
from chat.llm.llm_model import PaiLlm
from common.chat.models import ChatAgentRequest
from config.providers.llm_provider import llm_provider
from config.providers.mcp_tool_provider import mcp_provider
from config.providers.websearch_provider import websearch_provider
from config.providers.chatdb_provider import chatdb_provider
from llama_index.core.tools.function_tool import FunctionTool
from loguru import logger
from tools.knowledgebase.knowledgebase_tool import aget_knowledgebase_tool
from chat.tools.attachments.file_searcher import aget_file_searcher
from chat.tools.visit_webpage import aget_visit_webpage_tool
from utils.attachment_utils import is_attachment_truncated

async def aget_mcp_tools(chat_request: ChatAgentRequest, attachments: List[dict]=[]) -> List[FunctionTool]:
    mcp_tools = []

    if len(attachments) > 0:
        logger.info(f"Loading file searcher tool with attachments: {attachments}")
        file_searcher_tool = await aget_file_searcher(attachments=attachments)
        mcp_tools.append(file_searcher_tool)

    if chat_request.enable_search:
        # Add search web tool
        search_tools = websearch_provider.get_search_tools()
        mcp_tools.extend(search_tools)
        # Add visit webpage tool
        visit_webpage_tool = await aget_visit_webpage_tool()
        mcp_tools.append(visit_webpage_tool)

    if len(chat_request.mcp_ids) > 0:
        logger.info(f"[Model] selected mcp servers: {chat_request.mcp_ids}")
        mcp_tools.extend(await mcp_provider.get_mcp_tools_async(chat_request.mcp_ids))
        logger.info(f"[Model] mcp_tools: {mcp_tools}")

    if chat_request.enable_chatdb:
        mcp_tools.extend(chatdb_provider.get_db_tools())
        logger.info("Loaded chat_db tools.")

    mcp_tools.extend(await aget_kb_tools(chat_request))

    return mcp_tools


async def aget_kb_tools(chat_request: ChatAgentRequest) -> List[FunctionTool]:
    kb_tools = []

    kb_ids = chat_request.kb_ids or []
    for kb_id in kb_ids:
        kb_tools.append(await aget_knowledgebase_tool(kb_id=kb_id, user_id=chat_request.user_id))

    logger.info(f"Resolved {len(kb_tools)} knowledgebase tools.")
    return kb_tools


async def build_agent(chat_request: ChatAgentRequest) -> Planner:
    try:
        attachments = []
        for message in chat_request.messages:
            if message.get("role") == "user":
                for attachment in message.get("attachments", []):
                    if not str(attachment.get("contentType", "")).startswith(
                        "image/"
                    ):
                        is_truncated = await is_attachment_truncated(attachment.get("id"))
                        logger.info(f"Attachment {attachment.get('id')} truncated status is {is_truncated}.")
                        if is_truncated:
                            attachments.append(attachment)

        mcp_tools = await aget_mcp_tools(chat_request, attachments=attachments)
        llm: PaiLlm = llm_provider.get_llm_model(model_id=chat_request.model)

        prompt_set = PlanAgentPromptSet()
        if chat_request.prompts:
            prompt_set.plan_prompt = chat_request.prompts.get("plan") or prompt_set.plan_prompt
            prompt_set.act_prompt = chat_request.prompts.get("act") or prompt_set.act_prompt
            prompt_set.act_with_plan_prompt = chat_request.prompts.get("act_with_plan") or prompt_set.act_with_plan_prompt
            prompt_set.summary_prompt = chat_request.prompts.get("summary") or prompt_set.summary_prompt

        runner = Planner(
            llm=llm,
            prompt_set=prompt_set,
            tools=mcp_tools,
            name="Planner",
        )

        return runner
    except Exception as ex:
        logger.exception(f"Error in build_agent: {ex}")
        raise ex
