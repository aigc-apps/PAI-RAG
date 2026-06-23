from common.chat.models import ChatAgentRequest
from service.factory.model_factory import create_llm
from tools.knowledgebase.knowledgebase_tool import aget_knowledgebase_tool
from tools.knowledgebase.knowledgebase_file_tools import (
    aget_kb_catalog_tool,
    aget_kb_grep_tool,
    aget_kb_fetch_tool,
)
from tools.knowledgebase.faq_tool import aget_faq_tool
from service.factory.tools import create_search_tools, create_chatdb_tools, create_codesandbox_tools
from service.factory.mcp_factory import create_mcp_tools_async
from tools.attachments.file_chunk_searcher import aget_file_chunk_searcher
from tools.attachments.file_reader import aget_file_reader
from tools.attachments.multimodal_parser import aget_multimodal_parser_tool
import os
import traceback
from tools.code.code_sandbox_tool import DEFAULT_CODE_SANDBOX_DIR_PATH
from llama_index.core.tools.function_tool import FunctionTool
from sqlmodel.ext.asyncio.session import AsyncSession
from agent.react_agent import ReactAgent
from agent.prompts import REACT_PROMPT
from loguru import logger
from typing import List, Callable, Awaitable, Dict, Optional, AsyncIterator
from common.chat.models import MetadataFilteringCondition
from contextlib import asynccontextmanager

def append_text(user_message: Dict, text: str):
    """Append text to a user message, regardless of whether the content is
    stored as a string or as a list of content parts (OpenAI content-array
    format). If the list form has no text block yet — which happens when
    the user sends *only* attachments with no typed text — we add a fresh
    text block instead of silently dropping the append.
    """
    assert "content" in user_message, "Message必须包含content字段"
    content = user_message["content"]

    if content is None:
        user_message["content"] = text
        return
    if isinstance(content, str):
        user_message["content"] = content + text
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = (block.get("text") or "") + text
                return
        # No text block — append one so the hint isn't dropped.
        content.append({"type": "text", "text": text})
        return


def _user_message_has_text(user_message: Dict) -> bool:
    """True iff the user actually typed something (not just attached files)."""
    content = user_message.get("content")
    if not content:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                if (block.get("text") or "").strip():
                    return True
        return False
    return False


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
        file_resource_service_getter: Callable[[], Awaitable],
        faq_config_service_getter: Callable[[], Awaitable],
    ):
        self.session = session
        self._get_llm_service = llm_service_getter
        self._get_chatapp_service = chatapp_service_getter
        self._get_faq_config_service = faq_config_service_getter
        self._get_websearch_service = websearch_service_getter
        self._get_codesandbox_service = codesandbox_service_getter
        self._get_chatdb_service = chatdb_service_getter
        self._get_mcpserver_service = mcpserver_service_getter
        self._get_rag_service = rag_service_getter
        self._get_file_resource_service = file_resource_service_getter

    @asynccontextmanager
    async def create_agent(self, chat_request: ChatAgentRequest, tenant_id: str) -> AsyncIterator[ReactAgent]:
        sandbox_cleanup = None
        try:
            llm_service = await self._get_llm_service()
            llm_model = await llm_service.get_llm_by_model_id(chat_request.model, tenant_id=tenant_id)

            chatapp_id = None

            if llm_model:
                llm = create_llm(llm_model)
                system_prompt = REACT_PROMPT
                if chat_request.prompts:
                    system_prompt = chat_request.prompts.get("react", REACT_PROMPT)
            else:
                chatapp_service = await self._get_chatapp_service()
                chatapp = await chatapp_service.get_chatapp_by_app_id(chat_request.model, tenant_id=tenant_id)
                if not chatapp:
                    raise ValueError(f"Model `{chat_request.model}` not found.")

                chatapp_id = chatapp.app_id
                chat_request.model = chatapp.model_id
                chat_request.enable_faq = chatapp.enable_faq
                chat_request.mcp_ids = chatapp.mcp_ids
                chat_request.faq_config = chatapp.faq_config
                chat_request.kb_ids = chatapp.kb_ids
                chat_request.enable_search = chatapp.enable_search
                chat_request.enable_chatdb = chatapp.enable_chatdb
                chat_request.enable_agent = chatapp.enable_agent
                chat_request.vision_model_id = chatapp.vision_model_id
                chat_request.enable_input_guardrail = chatapp.enable_input_guardrail
                chat_request.enable_output_guardrail = chatapp.enable_output_guardrail
                chat_request.guardrail_hint = chatapp.guardrail_hint
                chat_request.enable_auto_metadata_filter = chatapp.enable_auto_metadata_filter

                llm_model = await llm_service.get_llm_by_model_id(chatapp.model_id, tenant_id=tenant_id)
                if not llm_model:
                    raise ValueError(f"LLM model {chatapp.model_id} not found.")

                llm = create_llm(llm_model)

                system_prompt = REACT_PROMPT
                if chatapp.prompts:
                    system_prompt = chatapp.prompts.get("react", REACT_PROMPT)

            tools, sandbox_cleanup = await self.aget_tools(
                messages=chat_request.messages,
                enable_search=chat_request.enable_search,
                enable_chatdb=chat_request.enable_chatdb,
                mcp_ids=chat_request.mcp_ids,
                kb_ids=chat_request.kb_ids,
                enable_faq=chat_request.enable_faq,
                faq_config=chat_request.faq_config,
                user_id=chat_request.user_id,
                chatapp_id=chatapp_id,
                metadata_condition=chat_request.metadata_condition,
                enable_auto_metadata_filter=chat_request.enable_auto_metadata_filter,
                vision_model_id=chat_request.vision_model_id,
                tenant_id=tenant_id,
            )

            tools_str = _build_tools_summary(tools)
            system_prompt = system_prompt.format(
                tools_str=tools_str,
                context_str="",  # backward-compat: old custom prompts may still have {context_str}
            )
            agent = ReactAgent(
                llm=llm,
                system_prompt=system_prompt,
                tools=tools,
            )

            yield agent
        except Exception as ex:
            logger.exception(f"Error in build_agent: {traceback.format_exc()}")
            raise ex
        finally:
            try:
                if sandbox_cleanup:
                    await sandbox_cleanup()
            except Exception:
                logger.exception(f"Error in agent cleanup: {traceback.format_exc()}")

    async def aget_tools(
        self,
        messages: List[dict],
        enable_search: bool = False,
        enable_chatdb: bool = False,
        enable_faq: bool = False,
        user_id: str = None,
        metadata_condition: Optional[MetadataFilteringCondition] = None,
        enable_auto_metadata_filter: bool = False,
        mcp_ids: List[str] = [],
        kb_ids: List[str] = [],
        tenant_id: str = None,
        chatapp_id: Optional[str] = None,
        faq_config: Optional[dict] = None,
        vision_model_id: Optional[str] = None,
    ) -> tuple[List[FunctionTool], Callable | None]:
        tools = []

        rag_service = await self._get_rag_service()
        chatapp_service = await self._get_chatapp_service()
        faq_config_service = await self._get_faq_config_service()

        if enable_faq and faq_config:
            tools.append(await aget_faq_tool(chatapp_id=chatapp_id, user_id=user_id, rag_service=rag_service, chatapp_service=chatapp_service, faq_config_service=faq_config_service, tenant_id=tenant_id))
            logger.info("Resolved FAQ tool (highest priority).")

        # 知识库工具
        for kb_id in kb_ids:
            tools.append(await aget_knowledgebase_tool(
                kb_id=kb_id, user_id=user_id, rag_service=rag_service,
                tenant_id=tenant_id, metadata_condition=metadata_condition,
                enable_auto_metadata_filter=enable_auto_metadata_filter,
            ))
            # General KB file tools (catalog / grep / fetch) over the whole KB,
            # complementing the semantic knowledgebase tool above.
            try:
                tools.append(await aget_kb_catalog_tool(
                    kb_id=kb_id, rag_service=rag_service, tenant_id=tenant_id,
                ))
                tools.append(await aget_kb_grep_tool(
                    kb_id=kb_id, rag_service=rag_service, tenant_id=tenant_id,
                ))
                tools.append(await aget_kb_fetch_tool(
                    kb_id=kb_id, rag_service=rag_service, tenant_id=tenant_id,
                ))
            except Exception:
                logger.exception(f"Failed to resolve knowledgebase file tools for kb {kb_id}")
        logger.info(f"Resolved {len(kb_ids)} knowledgebase tools.")

        # 搜索工具
        if enable_search:
            # Add search web tool
            websearch_service = await self._get_websearch_service()
            websearch_config = await websearch_service.get_websearch_config_or_create(tenant_id=tenant_id)
            if not websearch_config:
                raise ValueError("Websearch config not found.")

            search_tools = create_search_tools(websearch_config=websearch_config)
            tools.extend(search_tools)

        if len(mcp_ids) > 0:
            mcpserver_service = await self._get_mcpserver_service()
            mcpserver_configs = await mcpserver_service.get_mcpserver_by_ids(ids=mcp_ids, tenant_id=tenant_id)

            for mcpserver_config in mcpserver_configs:
                tools.extend(await create_mcp_tools_async(config=mcpserver_config))

        if enable_chatdb:
            chatdb_service = await self._get_chatdb_service()
            chatdb_config = await chatdb_service.get_chatdb_config_or_create(tenant_id=tenant_id)
            llm_service = await self._get_llm_service()
            llm_model = await llm_service.get_llm_by_model_id(chatdb_config.model_id, tenant_id=tenant_id)
            if not llm_model:
                raise ValueError(f"LLM model {chatdb_config.model_id} not found.")
            chatdb_llm = create_llm(llm_model)
            chatdb_tools = create_chatdb_tools(chatdb_config=chatdb_config, chatdb_llm=chatdb_llm)
            tools.extend(chatdb_tools)
            logger.info(f"Loaded {len(chatdb_tools)} chat_db tools.")

        attachment_tools, cleanup_code_sandbox = await self.parse_attachment_tools(
            messages=messages,
            tenant_id=tenant_id,
            vision_model_id=vision_model_id,
        )
        tools.extend(attachment_tools)
        logger.info(f"Loaded {len(attachment_tools)} attachment tools.")
        return tools, cleanup_code_sandbox


    async def parse_attachment_tools(
        self,
        messages: List[dict],
        tenant_id: str,
        vision_model_id: Optional[str] = None,
    ) -> tuple[List[FunctionTool], Callable | None]:
        file_service = await self._get_file_resource_service()
        llm_service = await self._get_llm_service()

        attachment_tools = []
        if not messages:
            return [], None

        user_message = messages[-1]
        if user_message.get("role") != "user":
            return [], None

        user_attachments = user_message.get("attachments", [])
        image_ids = []
        file_ids_to_read = []
        video_ids = []
        for attachment in user_attachments:
            attachment_file_id = attachment.get("id")
            attachment_content_type = attachment.get("contentType")
            if str(attachment_content_type).startswith("image/"):
                image_ids.append(attachment_file_id)
            elif str(attachment_content_type).startswith("video/"):
                video_ids.append(attachment_file_id)
            else:
                file_ids_to_read.append(attachment_file_id)

        image_base64_list = []
        video_base64_list = []
        if image_ids:
            image_base64_list = await file_service.get_file_base64_list(
                file_ids=image_ids, tenant_id=tenant_id,
            )
        if video_ids:
            # Videos ride as base64 data URIs too — the format verified on
            # the PAI-RAG feature branch and compatible with dashscope's
            # OpenAI-shim video_url shorthand. Keeps local dev working
            # without requiring an externally-reachable OSS endpoint.
            video_base64_list = await file_service.get_file_base64_list(
                file_ids=video_ids, tenant_id=tenant_id,
            )

        if image_base64_list or video_base64_list:
            attachment_tools.append(
                await aget_multimodal_parser_tool(
                    image_list=image_base64_list,
                    video_list=video_base64_list,
                    llm_service=llm_service,
                    tenant_id=tenant_id,
                    vision_model_id=vision_model_id,
                )
            )
            media_summary = []
            if image_ids:
                media_summary.append(f"{len(image_ids)} 张图片")
            if video_ids:
                media_summary.append(f"{len(video_ids)} 个视频")

            # Two cases:
            # - User typed something: short inline hint, let their query drive
            #   the agent as usual.
            # - User sent only media with no text: the chat LLM otherwise
            #   sees an empty prompt and has no reason to call any tool. Give
            #   it a direct instruction — understand the media first, then
            #   decide on further tools (search / KB / etc.) as needed.
            has_text = _user_message_has_text(user_message)
            if has_text:
                append_text(
                    user_message,
                    f"\n\n[已附件：{' + '.join(media_summary)}；"
                    f"如需分析其内容，请调用 `multimodal-parser` 工具。]",
                )
            else:
                append_text(
                    user_message,
                    f"用户上传了 {' + '.join(media_summary)}但没有文字提问。"
                    f"请先调用 `multimodal-parser` 工具理解附件内容，"
                    f"识别用户的真实意图；如果附件信息已足够回答，请直接给出回答；"
                    f"如果需要额外信息，再调用搜索 / 知识库等其他工具。",
                )

        if file_ids_to_read:
            # Register `read-file` for every text attachment regardless of
            # current extraction state. The tool itself does a fresh DB
            # lookup per invocation (and briefly polls if extraction is
            # still in flight), so it handles the upload→send race where
            # the worker is still parsing when parse_attachment_tools runs.
            read_files = await file_service.get_files(
                file_ids=file_ids_to_read, tenant_id=tenant_id,
            )
            read_file_names = [f.file_name for f in read_files if f.file_name]
            if read_file_names:
                attachment_tools.append(
                    await aget_file_reader(
                        file_ids=file_ids_to_read, tenant_id=tenant_id,
                    )
                )
                if _user_message_has_text(user_message):
                    append_text(
                        user_message,
                        f"\n\n 可以阅读的文件列表: \n\n {read_file_names}",
                    )
                else:
                    # User sent files with no typed question — prompt the
                    # agent to read + summarise + ask-for-clarification.
                    append_text(
                        user_message,
                        f"用户上传了以下文件但没有文字提问：{read_file_names}。"
                        f"请使用 `read-file` 工具读取文件内容，理解用户可能的意图，"
                        f"并给出摘要或基于内容的有用回答；若需要额外信息再调用其他工具。",
                    )

            # Only register search-file-chunks for files that already have
            # chunks — small files don't need a search tool, and the LLM
            # should just use read-file for them. Files still pending
            # chunking get picked up on the next chat turn; this is
            # acceptable because chunking only matters when the file is
            # large enough that inline reading would be truncated.
            large_files: List[dict] = []
            for fid in file_ids_to_read:
                chunk_count = await file_service.count_chunks(
                    file_id=fid, tenant_id=tenant_id
                )
                if chunk_count <= 0:
                    continue
                f_entity = await file_service.get_file(file_id=fid, tenant_id=tenant_id)
                if not f_entity:
                    continue
                large_files.append({
                    "file_id": fid,
                    "file_name": f_entity.file_name,
                    "chunk_count": chunk_count,
                })
            if large_files:
                attachment_tools.append(
                    await aget_file_chunk_searcher(
                        tenant_id=tenant_id,
                        files=large_files,
                    )
                )
                catalog_names = ", ".join(f["file_name"] for f in large_files)
                append_text(
                    user_message,
                    f"\n\n 对于较长的文件 [{catalog_names}] 可以调用 `search-file-chunks` "
                    f"工具按关键字检索。",
                )

        # coding tool
        attachment_names_in_message = []
        attachment_ids_in_message = []
        for message in messages:
            if message.get("role") == "user":
                user_attachments = message.get("attachments", [])
                if len(user_attachments) > 0:
                    for attachment in user_attachments:
                        attachment_file_entity = await file_service.get_file(file_id=attachment.get("id"), tenant_id=tenant_id)
                        if not attachment_file_entity:
                            logger.warning("Attachment file_id %s not found, skipping.", attachment.get("id"))
                            continue
                        name = attachment_file_entity.file_name
                        if not name:
                            logger.warning("Attachment missing 'name' field, skipping: %s", attachment)
                            continue
                        if attachment_file_entity.file_extension not in [".xlsx", ".csv", ".xls"]:
                            logger.info(f"Attachment {name} is not a spreadsheet file, skipping: {attachment_file_entity.file_extension}")
                            continue

                        attachment_names_in_message.append(name)
                        attachment_ids_in_message.append(attachment.get("id"))

        cleanup_code_sandbox = None
        if len(attachment_names_in_message) > 0:
            codesandbox_service = await self._get_codesandbox_service()
            codesandbox_config = await codesandbox_service.get_codesandbox_config_or_create(tenant_id=tenant_id)
            if codesandbox_config and codesandbox_config.enabled:
                codesandbox_tools, cleanup_code_sandbox = create_codesandbox_tools(
                    codesandbox_config=codesandbox_config,
                    code_sandbox_attachments_ids=attachment_ids_in_message,
                    file_service=file_service,
                    tenant_id=tenant_id,
                )
                logger.info(f"Loaded {len(codesandbox_tools)} codesandbox tools.")
                attachment_tools.extend(codesandbox_tools)

            attachment_names_in_message = [os.path.join(DEFAULT_CODE_SANDBOX_DIR_PATH, attachment_name) for attachment_name in attachment_names_in_message]
            attachment_names_in_message = ','.join(attachment_names_in_message)
            reply_text = f"\n\n 可以参考以下文件的本地路径回答：\n\n {attachment_names_in_message}"
            append_text(user_message, reply_text)

        return attachment_tools, cleanup_code_sandbox


def _build_tools_summary(tools: List[FunctionTool]) -> str:
    if not tools:
        return (
            "No tools are available in this session. "
            "Answer the user's questions directly using your own knowledge. "
            "Ignore all tool-related instructions above."
        )
    lines = [f"You have {len(tools)} tool(s). For every user query, pick the most relevant tool(s) to call:\n"]
    for tool in tools:
        name = tool.metadata.name
        desc = tool.metadata.description or ""
        first_line = desc.strip().split("\n")[0]
        lines.append(f"- **{name}**: {first_line}")
    lines.append("\nRemember: call at least one tool for any factual question.")
    return "\n".join(lines)
