import os
import threading
import traceback
from pai_rag.app.constants import (
    DEFAULT_APPLICATION_CONFIG_FILE,
)
from pai_rag.core.models.errors import UserInputError
from pai_rag.core.models.state import FileServiceState
from pai_rag.core.rag_application import PaiApp, RagChatType
from pai_rag.core.rag_config_manager import RagConfigManager, GENERATED_CONFIG_FILE_NAME
from pai_rag.core.utils.chat_utils import SseVersion
from pai_rag.utils.oss_utils import get_oss_auth
from pai_rag.app.api.models import (
    RagQuery,
    RagResponse,
    RetrievalRequest,
    NewRetrievalResponse,
)
from typing import Dict
from loguru import logger
from pai_rag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pai_rag.knowledgebase.rag_job_manager import job_manager

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

TASK_STATUS_FILE = "__upload_task_status.tmp"


class RagService:
    def initialize(self):
        self._state = FileServiceState(GENERATED_CONFIG_FILE_NAME)

        rag_configuration = RagConfigManager.from_file(DEFAULT_APPLICATION_CONFIG_FILE)
        if not os.path.exists(GENERATED_CONFIG_FILE_NAME):
            new_state = rag_configuration.persist()
            self._state.update_state(new_state)

        self.rag_configuration = rag_configuration

        knowledgebase_manager.compatible_init(rag_configuration.get_value())
        job_manager.update_config(new_config=rag_configuration.get_value())
        self.app = PaiApp(config=rag_configuration.get_value())

        self.reload_lock = threading.Lock()

        if os.path.exists(TASK_STATUS_FILE):
            open(TASK_STATUS_FILE, "w").close()

    def get_config(self):
        config = get_oss_auth(self.app.config)
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
                self.app.refresh(self.rag_configuration.get_value())
                self._state.update_state(new_state)
                logger.info(f"Reloaded rag configuration from {config_file}.")

    def reload(self, new_config: Dict):
        with self.reload_lock:
            logger.info("Reloading rag configuration from API.")
            self.rag_configuration.update(new_config)
            self.app.refresh(self.rag_configuration.get_value())
            job_manager.update_config(new_config=self.rag_configuration.get_value())
            config_mtime = self.rag_configuration.persist()
            self._state.update_state(config_mtime)
            logger.info("Reloaded rag configuration from API.")

    async def achat(self, query):
        try:
            return await self.app.achat(query)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Chat failed: {ex}")

    @dispatcher.span
    async def astream_chat(self, query):
        try:
            return await self.app.astream_chat(query)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Stream chat failed: {ex}")

    async def aquery_v1(self, query: RagQuery):
        try:
            if query.search_web:
                return await self.app.aquery(
                    query, RagChatType.WEB, sse_version=SseVersion.V1
                )
            return await self.app.aquery(
                query, RagChatType.RAG, sse_version=SseVersion.V1
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery_search_v1(self, query: RagQuery):
        try:
            return await self.app.aquery(
                query, RagChatType.WEB, sse_version=SseVersion.V1
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query Search failed: {ex}")

    async def aquery_llm_v1(
        self,
        query: RagQuery,
    ):
        try:
            return await self.app.aquery(
                query,
                chat_type=RagChatType.LLM,
                sse_version=SseVersion.V1,
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery(self, query: RagQuery):
        try:
            if query.search_web:
                return await self.app.aquery(
                    query, RagChatType.WEB, sse_version=SseVersion.V0
                )
            return await self.app.aquery(
                query, RagChatType.RAG, sse_version=SseVersion.V0
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery_search(self, query: RagQuery):
        try:
            return await self.app.aquery(
                query, RagChatType.WEB, sse_version=SseVersion.V0
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query Search failed: {ex}")

    async def aquery_llm(self, query: RagQuery):
        try:
            return await self.app.aquery(
                query, RagChatType.LLM, sse_version=SseVersion.V0
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery_retrieval(self, query: RagQuery):
        try:
            return await self.app.aretrieve(
                question=query.question, knowledgebase_name=query.index_name
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aknowledgebase_retrieval(
        self, retrieval_request: RetrievalRequest
    ) -> NewRetrievalResponse:
        try:
            return await self.app.aknowledgebase_retrieval(retrieval_request)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG failed: {ex}")

    async def aquery_agent(self, query: RagQuery) -> RagResponse:
        try:
            return await self.app.aquery(query, chat_type=RagChatType.Agent)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG Agent failed: {ex}")

    async def aquery_agent_v1(self, query: RagQuery) -> RagResponse:
        try:
            return await self.app.aquery(
                query, chat_type=RagChatType.Agent, sse_version=SseVersion.V1
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query RAG Agent failed: {ex}")

    async def aload_db_info(self):
        try:
            return await self.app.aload_db_info()
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Load DB info: {ex}")

    async def aquery_data_analysis_v1(self, query: RagQuery):
        try:
            return await self.app.aquery(
                query, chat_type=RagChatType.NL2SQL, sse_version=SseVersion.V1
            )
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Query Data Analysis failed: {ex}")


rag_service = RagService()
