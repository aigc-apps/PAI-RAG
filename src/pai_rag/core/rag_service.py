import asyncio
import os
import threading
import traceback
from asgi_correlation_id import correlation_id
from pai_rag.app.constants import DEFAULT_APPLICATION_CONFIG_FILE
from pai_rag.core.models.errors import UserInputError
from pai_rag.core.models.state import FileServiceState
from pai_rag.core.rag_application import RagApplication, RagChatType, SseVersion
from pai_rag.core.rag_config_manager import RagConfigManager, GENERATED_CONFIG_FILE_NAME
from pai_rag.utils.oss_utils import get_oss_auth
from pai_rag.app.api.models import (
    RagQuery,
    RagResponse,
)
from openinference.instrumentation import using_attributes
from typing import Dict, List
from loguru import logger
from pai_rag.knowledgebase.track_jobs import track_jobs

TASK_STATUS_FILE = "__upload_task_status.tmp"


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
    def initialize(self):
        self._state = FileServiceState(GENERATED_CONFIG_FILE_NAME)

        rag_configuration = RagConfigManager.from_file(DEFAULT_APPLICATION_CONFIG_FILE)
        if not os.path.exists(GENERATED_CONFIG_FILE_NAME):
            new_state = rag_configuration.persist()
            self._state.update_state(new_state)

        self.rag_configuration = rag_configuration
        self.rag = RagApplication(config=rag_configuration.get_value())

        self.reload_lock = threading.Lock()

        if os.path.exists(TASK_STATUS_FILE):
            open(TASK_STATUS_FILE, "w").close()

    def get_config(self):
        config = get_oss_auth(self.rag.config)
        return config.model_dump()

    def check_updates(self):
        new_state = self._state.check_state()
        if new_state != 0:
            logger.info(
                f"Detected changes for config file {self._state.state_key} {new_state}."
            )
            self.reload_from_file(GENERATED_CONFIG_FILE_NAME, new_state=new_state)

    def reload_from_file(self, config_file: str, new_state: int):
        with self.reload_lock:
            if self._state.state_value != new_state:
                logger.info(
                    f"Need reload configuration from background. {self._state.state_key}"
                )
                self.rag_configuration = RagConfigManager.from_file(config_file)
                self.rag.refresh(self.rag_configuration.get_value())
                self._state.update_state(new_state)
                logger.info(f"Reloaded rag configuration from {config_file}.")

    def reload(self, new_config: Dict):
        with self.reload_lock:
            logger.info("Reloading rag configuration from API.")
            self.rag_configuration.update(new_config)
            self.rag.refresh(self.rag_configuration.get_value())
            config_mtime = self.rag_configuration.persist()
            self._state.update_state(config_mtime)
            logger.info("Reloaded rag configuration from API.")

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

        track_jobs(task_id, index_name, input_files, status="start")
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
                task_id=task_id,
            )
            with open(TASK_STATUS_FILE, "a") as f:
                f.write(f"{task_id}\tcompleted\n")
        except Exception as ex:
            logger.error(f"Upload failed: {ex} {traceback.format_exc()}")
            with open(TASK_STATUS_FILE, "a") as f:
                detail = f"{ex}".replace("\t", " ").replace("\n", " ")
                f.write(f"{task_id}\tfailed\t{detail}\n")
                track_jobs(
                    task_id,
                    index_name,
                    input_files,
                    stage="parse",
                    status="failed",
                    detail=detail,
                )
            raise UserInputError(f"Upload knowledge failed: {ex}")

    def get_task_status(self, task_id: str):
        status = "unknown"
        detail = None
        if not os.path.exists(TASK_STATUS_FILE):
            return status, detail

        lines = open(TASK_STATUS_FILE).readlines()
        for line in lines[::-1]:
            if line.startswith(task_id):
                parts = line.strip().split("\t")
                status = parts[1]
                if len(parts) == 3:
                    detail = parts[2]
                break

        return status, detail

    async def aquery_v1(self, query: RagQuery):
        try:
            if query.search_web:
                return await self.rag.aquery(
                    query, RagChatType.WEB, sse_version=SseVersion.V1
                )
            return await self.rag.aquery(
                query, RagChatType.RAG, sse_version=SseVersion.V1
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def achat(self, query):
        try:
            return await self.rag.achat(query)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Chat RAG failed: {ex}")

    async def aquery_search_v1(self, query: RagQuery):
        try:
            return await self.rag.aquery(
                query, RagChatType.WEB, sse_version=SseVersion.V1
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query Search failed: {ex}")

    async def aquery_llm_v1(self, query: RagQuery):
        try:
            return await self.rag.aquery(
                query, RagChatType.LLM, sse_version=SseVersion.V1
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery(self, query: RagQuery):
        try:
            if query.search_web:
                return await self.rag.aquery(query, RagChatType.WEB)
            return await self.rag.aquery(query, RagChatType.RAG)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery_search(
        self, query: RagQuery, sse_version: SseVersion = SseVersion.V0
    ):
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

    async def aquery_retrieval(self, query: RagQuery):
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

    async def aquery_agent_v1(self, query: RagQuery) -> RagResponse:
        try:
            return await self.rag.aquery_agent(query, sse_version=SseVersion.V1)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG Agent failed: {ex}")

    async def aload_agent_config(self, agent_cfg_path: str):
        try:
            return await self.rag.aload_agent_config(agent_cfg_path)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Load agent config: {ex}")

    async def aload_db_info(self):
        try:
            return await self.rag.aload_db_info()
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Load DB info: {ex}")

    async def aquery_data_analysis_v1(self, query: RagQuery):
        try:
            return await self.rag.aquery(
                query, chat_type=RagChatType.NL2SQL, sse_version=SseVersion.V1
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query Data Analysis failed: {ex}")

    # async def aquery_analysis_v1(self, query: RagQuery):
    #     try:
    #         return await self.rag.aquery_analysis(query, sse_version=SseVersion.V1)
    #     except Exception as ex:
    #         logger.error(traceback.format_exc())
    #         raise UserInputError(f"Query Analysis failed: {ex}")


rag_service = RagService()
