from typing import List
from chat.agent.planner import PlanAgentPromptSet, Planner
from chat.llm.llm_model import PaiLlm
from common.chat.models import ChatAgentRequest
from config.providers.llm_provider import llm_provider
from config.providers.mcp_tool_provider import mcp_provider
from config.providers.websearch_provider import websearch_provider
from config.providers.code_sandbox_provider import codesandbox_provider
from config.providers.chatdb_provider import chatdb_provider
from llama_index.core.tools.function_tool import FunctionTool
from loguru import logger
from tools.knowledgebase.knowledgebase_tool import aget_knowledgebase_tool
from chat.tools.attachments.file_searcher import aget_file_searcher
from chat.tools.visit_webpage import aget_visit_webpage_tool
from utils.attachment_utils import is_attachment_truncated
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import with_async_db_session
from db.models.knowledgebase.file import KbFileEntity
from pairag.file.store.file_store_helper import file_store
from utils.tool_utils import get_binary_io_from_oss_url
from sqlmodel import select
import asyncio
from concurrent.futures import ThreadPoolExecutor


_EXECUTOR = ThreadPoolExecutor(max_workers=10)

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

@with_async_db_session
async def aupload_upload_files_to_code_sandbox(session: AsyncSession, file_ids: List[str]):
    file_res = await session.exec(
        select(KbFileEntity).where(
            KbFileEntity.id.in_(file_ids)
        )
    )
    processed_file_entities = file_res.all()
    files = [(entity.file_path, entity.file_name) for entity in processed_file_entities]
    unique_kb_ids = list({entity.kb_id for entity in processed_file_entities})
    assert len(unique_kb_ids) == 1, "file_ids must be from the same knowledgebase"
    loop = asyncio.get_event_loop()
    for file_path, file_name in files:
        file_url = file_store.get_url(file_path)
        file_content = get_binary_io_from_oss_url(file_url)
        _ = await loop.run_in_executor(
                    _EXECUTOR,
                    codesandbox_provider.tool.upload_data_file_to_sandbox,
                    file_content,
                    file_name,

                )


async def build_agent(chat_request: ChatAgentRequest) -> Planner:
    try:
        attachments = []
        code_sandbox_attachments = []
        for message in chat_request.messages:
            if message.get("role") == "user":
                code_sandbox_attachments.extend(message.get("attachments", []))
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

        code_sandbox_ready: asyncio.Future = asyncio.Future()
        if codesandbox_provider.tool and codesandbox_provider.tool.enabled:
            loop = asyncio.get_event_loop()
            sandbox_init_task = loop.run_in_executor(_EXECUTOR, codesandbox_provider.tool.create_session_and_context)

            if code_sandbox_attachments:
                logger.info(f"[Model] scheduling upload of {len(code_sandbox_attachments)} code sandbox attachments.")

                async def _initialize_and_upload():
                    try:
                        await sandbox_init_task
                        # 2. 上传文件
                        file_ids = [att["id"] for att in code_sandbox_attachments]
                        await aupload_upload_files_to_code_sandbox(file_ids=file_ids)
                        # 3. 标记就绪
                        code_sandbox_ready.set_result(None)
                        logger.info("[Model] Code sandbox ready and files uploaded.")
                    except Exception as e:
                        logger.exception("[Model] Failed to prepare code sandbox.")
                        code_sandbox_ready.set_exception(e)

                # 启动后台任务
                asyncio.create_task(_initialize_and_upload())
            else:
                async def _wait_sandbox_only():
                    try:
                        await sandbox_init_task
                        code_sandbox_ready.set_result(None)
                    except Exception as e:
                        code_sandbox_ready.set_exception(e)
                asyncio.create_task(_wait_sandbox_only())

            code_interpreter_tool = codesandbox_provider.get_code_sandbox_tool(code_sandbox_ready)
            mcp_tools.append(code_interpreter_tool)

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
