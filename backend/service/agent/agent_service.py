from common.chat.models import ChatAgentRequest
from service.factory.model_factory import create_llm
from tools.knowledgebase.knowledgebase_tool import aget_knowledgebase_tool
from service.factory.tools import create_search_tools, create_chatdb_tools, create_codesandbox_tools
from service.factory.mcp_factory import create_mcp_tools_async
from tools.attachments.file_searcher import aget_file_searcher
from tools.search.visit_webpage import aget_visit_webpage_tool
from tools.attachments.file_reader import aget_file_reader
from tools.attachments.image_parser import aget_image_parser_tool
from pairag.file.store.file_store_helper import file_store
import os
from tools.code.code_sandbox_tool import DEFAULT_CODE_SANDBOX_DIR_PATH
from llama_index.core.tools.function_tool import FunctionTool
from sqlmodel.ext.asyncio.session import AsyncSession
from agent.base import BaseAgent
from agent.planner import PlanAgentPromptSet, Planner
from loguru import logger
from typing import List, Callable, Awaitable, Dict


def append_text(user_message: Dict, text: str):
    assert "content" in user_message, "Message必须包含content字段"

    if isinstance(user_message["content"], str):
        user_message["content"] += text
    else:
        for block in user_message["content"]:
            if block.get("type") == "text":
                block["text"] += text
                return


class AgentService:
    def __init__(
        self,
        session: AsyncSession,
        llm_service_getter: Callable[[], Awaitable],
        chatapp_service_getter: Callable[[], Awaitable],
        websearch_service_getter: Callable[[], Awaitable],
        mcpserver_service_getter: Callable[[], Awaitable],
        codesandbox_service_getter: Callable[[], Awaitable],
        chatdb_service_getter: Callable[[], Awaitable],
        rag_service_getter: Callable[[], Awaitable],
        file_service_getter: Callable[[], Awaitable],
    ):
        self.session = session
        self._get_llm_service = llm_service_getter
        self._get_chatapp_service = chatapp_service_getter
        self._get_websearch_service = websearch_service_getter
        self._get_codesandbox_service = codesandbox_service_getter
        self._get_chatdb_service = chatdb_service_getter
        self._get_mcpserver_service = mcpserver_service_getter
        self._get_rag_service = rag_service_getter
        self._get_file_service = file_service_getter

    async def create_agent(self, chat_request: ChatAgentRequest) -> BaseAgent:
        try:
            llm_service = await self._get_llm_service()
            llm_model = await llm_service.get_llm_by_model_id(chat_request.model)

            prompt_set = PlanAgentPromptSet()

            if llm_model:
                llm = create_llm(llm_model)
            else:
                chatapp_service = await self._get_chatapp_service()
                chatapp = await chatapp_service.get_chatapp_by_app_id(chat_request.model)
                if not chatapp:
                    raise ValueError(f"Model `{chat_request.model}` not found.")

                llm_model = await llm_service.get_llm_by_model_id(chatapp.model_id)
                if not llm_model:
                    raise ValueError(f"LLM model {chatapp.model_id} not found.")

                llm = create_llm(llm_model)

                if chatapp.prompts:
                    prompt_set.plan_prompt = chatapp.prompts.get("plan", prompt_set.plan_prompt)
                    prompt_set.act_prompt = chatapp.prompts.get("act", prompt_set.act_prompt)
                    prompt_set.act_with_plan_prompt = chatapp.prompts.get("act_with_plan", prompt_set.act_with_plan_prompt)
                    prompt_set.summary_prompt = chatapp.prompts.get("summary", prompt_set.summary_prompt)

            tools = await self.aget_tools(
                messages=chat_request.messages,
                enable_search=chat_request.enable_search,
                enable_chatdb=chat_request.enable_chatdb,
                mcp_ids=chat_request.mcp_ids,
                kb_ids=chat_request.kb_ids,
                user_id=chat_request.user_id,
            )

            file_service = await self._get_file_service()
            llm_service = await self._get_llm_service()
            runner = Planner(
                llm=llm,
                prompt_set=prompt_set,
                tools=tools,
                name="Planner",
                file_service=file_service,
                llm_service=llm_service,
            )

            return runner
        except Exception as ex:
            logger.exception(f"Error in build_agent: {ex}")
            raise ex


    async def aget_tools(
        self,
        messages: List[dict],
        enable_search: bool = False,
        enable_chatdb: bool = False,
        user_id: str = None,
        mcp_ids: List[str] = [],
        kb_ids: List[str] = [],
    ) -> List[FunctionTool]:
        tools = []

        # 知识库工具
        rag_service = await self._get_rag_service()
        for kb_id in kb_ids:
            tools.append(await aget_knowledgebase_tool(kb_id=kb_id, user_id=user_id, rag_service=rag_service))
        logger.info(f"Resolved {len(kb_ids)} knowledgebase tools.")

        # 搜索工具
        if enable_search:
            # Add search web tool
            websearch_service = await self._get_websearch_service()
            websearch_config = await websearch_service.get_websearch_config_or_create()
            if not websearch_config:
                raise ValueError("Websearch config not found.")

            search_tools = create_search_tools(websearch_config=websearch_config)
            tools.extend(search_tools)
            # Add visit webpage tool
            visit_webpage_tool = await aget_visit_webpage_tool()
            tools.append(visit_webpage_tool)

        if len(mcp_ids) > 0:
            mcpserver_service = await self._get_mcpserver_service()
            mcpserver_configs = await mcpserver_service.get_mcpserver_by_ids(ids=mcp_ids)

            for mcpserver_config in mcpserver_configs:
                tools.extend(await create_mcp_tools_async(config=mcpserver_config))

        if enable_chatdb:
            chatdb_service = await self._get_chatdb_service()
            chatdb_config = await chatdb_service.get_chatdb_config_or_create()
            chatdb_tools = create_chatdb_tools(chatdb_config=chatdb_config)
            tools.extend(chatdb_tools)
            logger.info(f"Loaded {len(chatdb_tools)} chat_db tools.")

        attachment_tools = await self.parse_attachment_tools(messages=messages)
        tools.extend(attachment_tools)
        logger.info(f"Loaded {len(attachment_tools)} attachment tools.")
        return tools


    async def parse_attachment_tools(
        self,
        messages: List[dict],
    ):
        file_service = await self._get_file_service()
        llm_service = await self._get_llm_service()
        rag_service = await self._get_rag_service()

        attachment_tools = []
        if not messages:
            return []

        user_message = messages[-1]
        if user_message.get("role") != "user":
            return []

        user_attachments = user_message.get("attachments", [])
        image_url_list = []
        file_ids_to_read = []

        for attachment in user_attachments:
            attachment_file_id = attachment.get("id")
            if str(attachment.get("contentType")).startswith("image/"):
                image_file_entity = await file_service.get_file_by_id(file_id=attachment.get("id"))
                if not image_file_entity:
                    logger.warning(f"Image file entity not found for attachment {attachment.get('id')}")
                    continue
                image_url = file_store.get_url(image_file_entity.file_path)
                image_url_list.append(image_url)
                attachment_content = attachment.get("content", "")

                if isinstance(attachment_content, List):
                    for content in attachment_content:
                        if isinstance(content, dict) and content.get("type") == "image":
                            image_url_list.append(content.get("image"))
                else:
                    image_url_list.append(image_url)

                attachment_tools.append(await aget_image_parser_tool(llm_service=llm_service))
            else:
                file_ids_to_read.append(attachment_file_id)
                attachment_tools.append(await aget_file_reader(file_service=file_service))


        # 文件搜索工具
        if len(file_ids_to_read) > 0:
            logger.info(f"Loading file searcher tool with file ids to read: {file_ids_to_read}")
            file_searcher_tool = await aget_file_searcher(rag_service=rag_service)
            attachment_tools.append(file_searcher_tool)


        # 只在user message最后追加列出文件结果，不使用 tool_call / tool 消息， 增加相关hint提示

        if len(image_url_list) > 0:
            reply_text = f"\n\n 可以参考的图片的URL列表: \n\n {image_url_list}"
            append_text(user_message, reply_text)

        if len(file_ids_to_read) > 0:
            reply_text = f"\n\n 可以阅读的文件的ID列表: \n\n {file_ids_to_read}"
            append_text(user_message, reply_text)

        # coding tool
        attachment_names_in_message = []
        attachment_ids_in_message = []
        for message in messages:
            if message.get("role") == "user":
                user_attachments = message.get("attachments", [])
                if len(user_attachments) > 0:
                    for attachment in user_attachments:
                        attachment_file_entity = await file_service.get_file_by_id(file_id=attachment.get("id"))
                        name = attachment_file_entity.file_name
                        if not name:
                            logger.warning("Attachment missing 'name' field, skipping: %s", attachment)
                            continue
                        if attachment_file_entity.file_extension not in [".xlsx", ".csv"]:
                            logger.info(f"Attachment {name} is not a spreadsheet file, skipping: {attachment_file_entity.file_extension}")
                            continue

                        attachment_names_in_message.append(name)
                        attachment_ids_in_message.append(attachment.get("id"))

        if len(attachment_names_in_message) > 0:
            codesandbox_service = await self._get_codesandbox_service()
            codesandbox_config = await codesandbox_service.get_codesandbox_config_or_create()
            if codesandbox_config and codesandbox_config.enabled:
                codesandbox_tools = create_codesandbox_tools(
                    codesandbox_config=codesandbox_config,
                    code_sandbox_attachments_ids=attachment_ids_in_message,
                    file_service=file_service,
                )
                logger.info(f"Loaded {len(codesandbox_tools)} codesandbox tools.")
                attachment_tools.extend(codesandbox_tools)

            attachment_names_in_message = [os.path.join(DEFAULT_CODE_SANDBOX_DIR_PATH, attachment_name) for attachment_name in attachment_names_in_message]
            attachment_names_in_message = ','.join(attachment_names_in_message)
            reply_text = f"\n\n 可以参考以下文件的本地路径回答：\n\n {attachment_names_in_message}"
            append_text(user_message, reply_text)

        return attachment_tools
