from pairag.chat.chat_flow import ChatFlow
from pairag.core.rag_config import RagConfig
from pairag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pairag.core.rag_module import (
    resolve_data_analysis_loader,
    resolve_query_engine,
    resolve_vector_index,
    resolve_query_engine_from_retrieval_request,
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

        vector_index = resolve_vector_index(knowledgebase_manager.get_knowledgebase())
        _ = resolve_query_engine(self.config, vector_index=vector_index)

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

    async def aretrieve(
        self,
        question: str,
        knowledgebase_name: str = None,
    ) -> RetrievalResponse:
        query_bundle = QueryBundle(question)
        knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_name)
        vector_index = resolve_vector_index(knowledgebase=knowledgebase)
        query_engine = resolve_query_engine(self.config, vector_index=vector_index)
        node_results = await query_engine.aretrieve(query_bundle)

        docs = [
            ContextDoc(
                text=score_node.node.get_content(),
                metadata=score_node.node.metadata,
                score=score_node.score,
                image_url=score_node.node.image_url,
            )
            if isinstance(score_node.node, ImageNode)
            else ContextDoc(
                text=score_node.node.get_content(),
                metadata=score_node.node.metadata,
                score=score_node.score,
            )
            for score_node in node_results
        ]

        return RetrievalResponse(docs=docs)

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
        query_engine = resolve_query_engine_from_retrieval_request(
            self.config,
            vector_index=vector_index,
            retrieval_settings=_retrieval_settings,
        )
        node_results = await query_engine.aretrieve(query_bundle)

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
