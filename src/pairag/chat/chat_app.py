from pairag.chat.chat_flow import ChatFlow
from pairag.core.rag_config import RagConfig
from pairag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pairag.core.rag_module import (
    resolve_data_analysis_loader,
    resolve_index_retriever_from_retrieval_settings,
    resolve_vector_index,
)

from pairag.chat.models import (
    ContextDoc,
    RetrievalResponse,
    RetrievalRequest,
    DocRecord,
    NewRetrievalResponse,
    ChatCompletionRequest,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import ImageNode
from loguru import logger
from enum import Enum
from pairag.integrations.trace.base import init_trace

DEFAULT_RAG_INDEX_FILE = "localdata/default_rag_indexes.json"


class RagChatType(str, Enum):
    LLM = "llm"
    RAG = "rag"
    WEB = "web"
    NL2SQL = "nl2sql"
    Agent = "agent"


class ChatApp:
    def __init__(self, config: RagConfig):
        self.config = config
        if self.config.trace.is_enabled():
            init_trace(self.config.trace)

        _ = resolve_vector_index(knowledgebase_manager.get_knowledgebase())

    def refresh(self, config: RagConfig):
        self.config = config
        if self.config.trace.is_enabled():
            init_trace(self.config.trace)

    async def achat(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.achat(chat_request)

    async def astream_chat(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.astream_chat(chat_request)

    async def aknowledgebase_retrieval(
        self, retrieval_request: RetrievalRequest
    ) -> NewRetrievalResponse:
        query_bundle = QueryBundle(retrieval_request.query)
        knowledgebase = knowledgebase_manager.get_knowledgebase(
            retrieval_request.knowledgebase_id
        )
        vector_index = resolve_vector_index(knowledgebase=knowledgebase)
        _retrieval_settings = {
            **knowledgebase.retrieval_settings,
            **retrieval_request.retrieval_settings,
        }
        logger.info(
            f"aknowledgebase_retrieval ==> query: {retrieval_request.query} to knowledgebase_id: {retrieval_request.knowledgebase_id} with retrieval_settings: {_retrieval_settings}"
        )

        retriever = resolve_index_retriever_from_retrieval_settings(
            vector_index=vector_index,
            retrieval_settings=_retrieval_settings,
        )
        node_results = await retriever.aretrieve(query_bundle)

        records = [
            DocRecord(
                content=score_node.node.get_content(),
                score=score_node.score,
                title=score_node.node.metadata.get("file_name", "null"),
                metadata=score_node.node.metadata,
            )
            for score_node in node_results
        ]

        return NewRetrievalResponse(records=records)

    async def aload_db_info(self):
        db_info_loader = resolve_data_analysis_loader(self.config)
        await db_info_loader.aload_db_info()

        return "Load database info successfully."
