import asyncio
import os
from asgi_correlation_id import correlation_id
from pai_rag.core.rag_application import RagApplication
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.app.api.models import (
    RagQuery,
    LlmQuery,
    RetrievalQuery,
    RagResponse,
    LlmResponse,
)
from openinference.instrumentation import using_attributes
from typing import Any
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
    def initialize(self, config_file: str):
        self.config_file = config_file
        self.rag_configuration = RagConfiguration.from_file(config_file)
        self.config_dict_value = self.rag_configuration.get_value().to_dict()
        self.config_modified_time = self.rag_configuration.get_config_mtime()

        self.rag_configuration.persist()

        self.rag = RagApplication()
        self.rag.initialize(self.rag_configuration.get_value())

        if os.path.exists(TASK_STATUS_FILE):
            open(TASK_STATUS_FILE, "w").close()

    def get_config(self):
        self.check_updates()
        return self.config_dict_value

    def reload(self, new_config: Any = None):
        rag_snapshot = RagConfiguration.from_snapshot()
        if new_config:
            # 多worker模式，读取最新的setting
            rag_snapshot.update(new_config)
        config_snapshot = rag_snapshot.get_value()

        new_dict_value = config_snapshot.to_dict()
        if self.config_dict_value != new_dict_value:
            logger.debug("Config changed, reload")
            self.rag.reload(config_snapshot)
            self.config_dict_value = new_dict_value
            self.rag_configuration = rag_snapshot
            self.rag_configuration.persist()
        else:
            logger.debug("Config not changed, not reload")

    def check_updates(self):
        logger.info("Checking updates")
        new_modified_time = self.rag_configuration.get_config_mtime()
        if self.config_modified_time != new_modified_time:
            self.reload()
            self.config_modified_time = new_modified_time
        else:
            logger.info("No updates")

    def add_knowledge_async(
        self,
        task_id: str,
        file_dir: str,
        faiss_path: str = None,
        enable_qa_extraction: bool = False,
    ):
        self.check_updates()
        with open(TASK_STATUS_FILE, "a") as f:
            f.write(f"{task_id} processing\n")
        try:
            self.rag.load_knowledge(file_dir, faiss_path, enable_qa_extraction)
            with open(TASK_STATUS_FILE, "a") as f:
                f.write(f"{task_id} completed\n")
        except Exception as ex:
            logger.error(f"Upload failed: {ex}")
            with open(TASK_STATUS_FILE, "a") as f:
                f.write(f"{task_id} failed\n")
            raise

    def get_task_status(self, task_id: str) -> str:
        self.check_updates()
        default_status = "unknown"
        if not os.path.exists(TASK_STATUS_FILE):
            return default_status

        lines = open(TASK_STATUS_FILE).readlines()
        for line in lines[::-1]:
            if line.startswith(task_id):
                return line.strip().split(" ")[1]

        return default_status

    async def aquery(self, query: RagQuery) -> RagResponse:
        self.check_updates()
        return await self.rag.aquery(query)

    async def aquery_llm(self, query: LlmQuery) -> LlmResponse:
        self.check_updates()
        return await self.rag.aquery_llm(query)

    async def aquery_retrieval(self, query: RetrievalQuery):
        self.check_updates()
        return await self.rag.aquery_retrieval(query)

    async def aquery_agent(self, query: LlmQuery) -> LlmResponse:
        self.check_updates()
        return await self.rag.aquery_agent(query)

    async def batch_evaluate_retrieval_and_response(self, type):
        self.check_updates()
        return await self.rag.batch_evaluate_retrieval_and_response(type)


rag_service = RagService()
