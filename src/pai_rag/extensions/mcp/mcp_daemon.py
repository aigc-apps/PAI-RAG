import asyncio
import time
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_module import resolve_chat_llm
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from pai_rag.core.rag_service import rag_service
from loguru import logger
import threading
import traceback


class MCPDaemon:
    def __init__(self):
        self._lock = threading.Lock()
        logger.debug("MCPDaemon init")

    def generate_mcp_desc(self):
        try:
            asyncio.get_event_loop()
        except Exception as ex:
            logger.warning(f"No event loop found, will create new: {ex}")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        while True:
            if not hasattr(rag_service, "app"):
                logger.debug("MCP自动生成描述信息任务队列准备中...")
                time.sleep(2)
                continue
            if hasattr(rag_service, "app"):
                rag_config_value = rag_service.get_config()
                rag_config = RagConfig.model_validate(rag_config_value)
                if (
                    rag_config.mcp_servers is None
                    or len(rag_config.mcp_servers) == 0
                    or all(
                        mcp_server.description is not None
                        for mcp_server in rag_config.mcp_servers
                    )
                ):
                    logger.debug("MCP后台任务队列为空。sleeping...")
                    time.sleep(5)
                    continue
                default_llm = resolve_chat_llm(rag_config)
                if not default_llm:
                    logger.warning("MCP_LLM未成功初始化。sleeping...")
                    time.sleep(2)
                    continue
                else:
                    mcp_servers = rag_config.mcp_servers
                    for mcp_server in mcp_servers:
                        if mcp_server.description is None:
                            try:
                                logger.debug(
                                    f"MCP Server: {mcp_server.name} 没有描述信息, 将自动生成..."
                                )
                                messages = [
                                    ChatMessage(
                                        role=MessageRole.USER,
                                        content=f"您是一位帮助生成MCP服务器描述的助手。MCP服务器的名称为：{mcp_server.name}。请为该MCP服务器生成一个描述。描述的字数应少于100字。描述应该用中文书写。描述应该使用Markdown格式。描述的格式应如下所示：# MCP服务器描述\n\n## MCP服务器名称：{mcp_server.name} ##描述：",
                                    )
                                ]
                                try:
                                    response = default_llm.chat(messages)
                                except Exception as e:
                                    logger.error(f"描述信息生成失败: {e}")
                                    continue

                                mcp_server.description = response.message.content
                                logger.debug(
                                    f"描述信息生成完成: {mcp_server.name}: {mcp_server.description}"
                                )
                                with self._lock:
                                    rag_service.reload(rag_config.model_dump())
                            except Exception:
                                logger.error(
                                    f"后台生成描述信息出错: for MCP Server: {mcp_server.name}.  {traceback.format_exc()}"
                                )
                        else:
                            continue


mcp_daemon = MCPDaemon()
