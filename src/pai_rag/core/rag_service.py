import asyncio
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
from pai_rag.app.web.view_model import view_model
from openinference.instrumentation import using_attributes
from typing import Any, Dict


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
        self.rag_configuration = RagConfiguration.from_file(config_file)
        view_model.sync_app_config(self.rag_configuration.get_value())
        self.rag = RagApplication()
        self.rag.initialize(self.rag_configuration.get_value())
        self.tasks_status: Dict[str, str] = {}

    def reload(self, new_config: Any):
        self.rag_configuration.update(new_config)
        self.rag.reload(self.rag_configuration.get_value())
        self.rag_configuration.persist()

    def add_knowledge_async(
        self, task_id: str, file_dir: str, enable_qa_extraction: bool = False
    ):
        self.tasks_status[task_id] = "processing"
        try:
            self.rag.load_knowledge(file_dir, enable_qa_extraction)
            self.tasks_status[task_id] = "completed"
        except Exception:
            self.tasks_status[task_id] = "failed"

    def get_task_status(self, task_id: str) -> str:
        return self.tasks_status.get(task_id, "unknown")

    async def aquery(self, query: RagQuery) -> RagResponse:
        return await self.rag.aquery(query)

    async def aquery_llm(self, query: LlmQuery) -> LlmResponse:
        return await self.rag.aquery_llm(query)

    async def aquery_retrieval(self, query: RetrievalQuery):
        return await self.rag.aquery_retrieval(query)

    async def aquery_agent(self, query: LlmQuery) -> LlmResponse:
        return await self.rag.aquery_agent(query)

    async def batch_evaluate_retrieval_and_response(self, type):
        return await self.rag.batch_evaluate_retrieval_and_response(type)


rag_service = RagService()
