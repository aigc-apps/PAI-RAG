import asyncio
import os
import traceback
from asgi_correlation_id import correlation_id
from pai_rag.core.models.errors import UserInputError
from pai_rag.core.rag_application import RagApplication, RagChatType
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.utils.oss_utils import get_oss_auth
from pai_rag.app.api.models import (
    RagQuery,
    RagResponse,
    RetrievalQuery,
)
from openinference.instrumentation import using_attributes
from typing import Dict, List
import logging

TASK_STATUS_FILE = "__upload_task_status.tmp"
logger = logging.getLogger(__name__)


def trace_correlation_id(function):
    def _trace_correlation_id(*args, **kwargs):
        session_id = correlation_id.get()
        with using_attributes(
            session_id=session_id,
        ):
            return function(*args, **kwargs)

    async def _a_trace_correlation_id(*args, **kwargs):
        session_id = correlation_id.get()
        with using_attributes(
            session_id=session_id,
        ):
            return await function(*args, **kwargs)

    if asyncio.iscoroutinefunction(function):
        return _a_trace_correlation_id
    else:
        return _trace_correlation_id


class RagService:
    def initialize(self, rag_configuration: RagConfigManager):
        self.rag_configuration = rag_configuration
        self.rag = RagApplication(config=rag_configuration.get_value())

        if os.path.exists(TASK_STATUS_FILE):
            open(TASK_STATUS_FILE, "w").close()

    def get_config(self):
        config = get_oss_auth(self.rag.config)
        return config.model_dump()

    def reload(self, new_config: Dict):
        self.rag_configuration.update(new_config)
        self.rag.refresh(self.rag_configuration.get_value())
        self.rag_configuration.persist()

    def add_knowledge(
        self,
        task_id: str,
        input_files: List[str] = None,
        filter_pattern: str = None,
        oss_path: str = None,
        index_name: str = None,
        enable_raptor: bool = False,
        enable_multimodal: bool = False,
        from_oss: bool = False,
        temp_file_dir: str = None,
    ):
        try:
            asyncio.get_event_loop()
        except Exception as ex:
            logger.warning(f"No event loop found, will create new: {ex}")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        with open(TASK_STATUS_FILE, "a") as f:
            f.write(f"{task_id}\tprocessing\n")
        try:
            self.rag.load_knowledge(
                input_files=input_files,
                filter_pattern=filter_pattern,
                from_oss=from_oss,
                index_name=index_name,
                oss_path=oss_path,
                enable_raptor=enable_raptor,
                enable_multimodal=enable_multimodal,
            )
            with open(TASK_STATUS_FILE, "a") as f:
                f.write(f"{task_id}\tcompleted\n")
        except Exception as ex:
            logger.error(f"Upload failed: {ex} {traceback.format_exc()}")
            with open(TASK_STATUS_FILE, "a") as f:
                detail = f"{ex}".replace("\t", " ").replace("\n", " ")
                f.write(f"{task_id}\tfailed\t{detail}\n")
            raise UserInputError(f"Upload knowledge failed: {ex}")
        finally:
            if temp_file_dir:
                os.rmdir(temp_file_dir)

    def get_task_status(self, task_id: str) -> str:
        status = "unknown"
        detail = None
        if not os.path.exists(TASK_STATUS_FILE):
            return status

        lines = open(TASK_STATUS_FILE).readlines()
        for line in lines[::-1]:
            if line.startswith(task_id):
                parts = line.strip().split("\t")
                status = parts[1]
                if len(parts) == 3:
                    detail = parts[2]
                break

        return status, detail

    async def aquery(self, query: RagQuery):
        try:
            return await self.rag.aquery(query, RagChatType.RAG)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery_search(self, query: RagQuery):
        try:
            return await self.rag.aquery(query, RagChatType.WEB)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query Search failed: {ex}")

    async def aquery_llm(self, query: RagQuery):
        try:
            return await self.rag.aquery(query, RagChatType.LLM)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery_retrieval(self, query: RetrievalQuery):
        try:
            return await self.rag.aretrieve(query)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery_agent(self, query: RagQuery) -> RagResponse:
        try:
            return await self.rag.aquery_agent(query)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG Agent failed: {ex}")

    async def aload_agent_config(self, agent_cfg_path: str):
        try:
            return await self.rag.aload_agent_config(agent_cfg_path)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Load agent config: {ex}")

    async def aquery_analysis(self, query: RagQuery):
        try:
            return await self.rag.aquery_analysis(query)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query Analysis failed: {ex}")


rag_service = RagService()
