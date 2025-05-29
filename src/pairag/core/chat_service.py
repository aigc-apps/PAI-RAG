import os
import threading
import traceback
from pairag.app.constants import (
    DEFAULT_APPLICATION_CONFIG_FILE,
)
from pairag.core.models.errors import UserInputError
from pairag.core.models.state import FileServiceState
from pairag.chat.chat_app import ChatApp
from pairag.core.rag_config_manager import RagConfigManager, GENERATED_CONFIG_FILE_NAME
from pairag.chat.models import (
    RetrievalRequest,
    NewRetrievalResponse,
)
from typing import Dict
from loguru import logger
from pairag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pairag.data_pipeline.job.rag_job_manager import job_manager


class ChatService:
    def initialize(self):
        self._state = FileServiceState(GENERATED_CONFIG_FILE_NAME)

        rag_configuration = RagConfigManager.from_file(DEFAULT_APPLICATION_CONFIG_FILE)
        if not os.path.exists(GENERATED_CONFIG_FILE_NAME):
            new_state = rag_configuration.persist()
            self._state.update_state(new_state)

        self.rag_configuration = rag_configuration

        knowledgebase_manager.compatible_init(rag_configuration.get_value())
        job_manager.update_config(new_config=rag_configuration.get_value())
        self.app = ChatApp(config=rag_configuration.get_value())

        self.reload_lock = threading.Lock()

    def get_config(self):
        return self.app.config.model_dump()

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

    async def astream_chat(self, query):
        try:
            return await self.app.astream_chat(query)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Stream chat failed: {ex}")

    async def aquery_retrieval(self, question: str, knowledgebase: str = None):
        try:
            return await self.app.aretrieve(
                question=question, knowledgebase_name=knowledgebase
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

    async def aload_db_info(self):
        try:
            return await self.app.aload_db_info()
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Load DB info: {ex}")

    async def astream_web_atomic(self, chat_request):
        try:
            return await self.app.astream_web_atomic(chat_request)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Chat web failed: {ex}")

    async def astream_llm_atomic(self, chat_request):
        try:
            return await self.app.astream_llm_atomic(chat_request)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Chat llm failed: {ex}")

    async def astream_news_agent_atomic(self, chat_request):
        try:
            return await self.app.astream_news_agent_atomic(chat_request)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Chat news failed: {ex}")

    async def astream_knowledgebase_atomic(self, chat_request):
        try:
            return await self.app.astream_knowledgebase_atomic(chat_request)
        except Exception as ex:
            logger.error(traceback.format_exc())
            raise UserInputError(f"Chat web failed: {ex}")


chat_service = ChatService()
